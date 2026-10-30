import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from django.core.cache import cache

from apps.market import repository
from parity.cases import build_cases
from parity.engine_side import METRICS, run_engine
from parity.frozen import MarketSnapshot, engine_patches
from parity.kpis import derive_kpis
from parity.tests.synthetic import TODAY, make_hist, make_snapshot


def _w1():
    return build_cases(make_hist(), TODAY)[0]


def test_snapshot_load_matches_repository_contract():
    snap = make_snapshot()
    s = snap.load("bcb:12", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert s.name == "bcb:12"
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index[0] == pd.Timestamp("2024-01-02")
    assert s.index[-1] == pd.Timestamp("2024-01-05")  # fatia inclusiva
    assert s.dtype == float

    empty = snap.load("bcb:12", dt.date(1999, 1, 1), dt.date(1999, 12, 31))
    assert empty.empty
    assert isinstance(empty.index, pd.DatetimeIndex)
    assert empty.name == "bcb:12"


def test_snapshot_load_returns_copy():
    snap = make_snapshot()
    s = snap.load("bcb:12", dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    s.iloc[0] = -1.0
    assert snap.load("bcb:12", dt.date(2024, 1, 2), dt.date(2024, 1, 5)).iloc[0] != -1.0


def test_snapshot_load_unknown_key_raises():
    with pytest.raises(KeyError):
        make_snapshot().load("bcb:999", dt.date(2024, 1, 1), dt.date(2024, 2, 1))


def test_snapshot_missing_lists_empty_keys():
    snap = make_snapshot()
    snap.series["b3:IFIX"] = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert snap.missing() == ["b3:IFIX"]


def test_engine_patches_never_touch_repository(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("Postgres lido durante a camada 1")

    monkeypatch.setattr(repository, "load_series", boom)
    out = run_engine(_w1(), make_snapshot(), make_hist())
    assert out is not None
    assert set(out) == set(METRICS)
    assert repository.load_series is boom  # restaurado ao sair do patch


def test_engine_patches_use_dummy_cache():
    with engine_patches(make_snapshot(), make_hist()):
        cache.set("market:x", 1)
        assert cache.get("market:x") is None


def test_engine_patches_block_writes():
    with engine_patches(make_snapshot(), make_hist()):
        with pytest.raises(RuntimeError, match="escrita no Postgres proibida"):
            repository.upsert_points("bcb:12", "BCB", "rate", [])


def test_engine_patches_feed_frozen_history():
    from apps.market.services.sources import dlp

    hist = make_hist()
    with engine_patches(make_snapshot(), hist):
        got = dlp.buscar_historico("qualquer", classe="Ações")
        got.loc[0, "vlr_mercado"] = -1.0
        again = dlp.buscar_historico("qualquer")
    pd.testing.assert_frame_equal(again, hist)


def test_run_engine_outputs_contract():
    out = run_engine(_w1(), make_snapshot(), make_hist())
    assert set(out) == set(METRICS)
    assert set(out["twr"].columns) == {"Carteira", "CDI", "IPCA + 6%", "S&P 500 BRL", "IFIX"}
    assert list(out["irr"].columns) == ["TIR"]
    assert list(out["kpis"].columns) == ["valor"]
    assert "Total Acum." in out["yearly"].columns
    assert all(isinstance(c, str) for c in out["yearly"].columns)
    assert list(out["risk_return"].columns) == ["Volatilidade", "Retorno (CAGR)", "Sharpe"]
    assert list(out["portfolio"].columns) == ["vlr_mercado", "vlr_investido", "proventos", "fluxo"]
    assert not out["rolling_vol"].empty
    assert not out["rolling_sharpe"].empty


def test_run_engine_returns_none_for_empty_portfolio():
    empty = make_hist().iloc[0:0]
    assert run_engine(_w1(), make_snapshot(), empty) is None


def test_derive_kpis_matches_summary_rule():
    pdf = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "vlr_mercado": [100.0, 150.0, 180.0],
            "vlr_investido": [100.0, 140.0, 160.0],
            "proventos": [0.0, 0.0, 0.0],
            "fluxo": [100.0, 40.0, 20.0],
        }
    )
    idx = pdf["date"]
    twr = pd.DataFrame({"Carteira": [100.0, 105.0, 112.5], "CDI": [100.0, 100.1, 100.2]}, index=idx)
    rr = pd.DataFrame(
        {"Volatilidade": [0.2, 0.01], "Retorno (CAGR)": [0.3, 0.1], "Sharpe": [1.0, 0.0]},
        index=["Carteira", "CDI"],
    )
    dd = pd.DataFrame({"Carteira": [0.0, -0.05, -0.02]}, index=idx)
    irr = pd.Series([10.0, 12.5], index=idx[:2])

    k = derive_kpis(pdf, twr, rr, dd, irr)["valor"]
    assert list(k.index) == [
        "patrimonio_inicial", "aportes_periodo", "patrimonio", "ganho_periodo", "twr_total",
        "cagr", "volatilidade", "sharpe", "max_drawdown", "xirr",
    ]
    assert k["patrimonio_inicial"] == 100.0
    assert k["aportes_periodo"] == 60.0
    assert k["patrimonio"] == 180.0
    assert k["ganho_periodo"] == 20.0
    assert k["twr_total"] == pytest.approx(12.5)
    assert k["cagr"] == 0.3
    assert k["volatilidade"] == 0.2
    assert k["sharpe"] == 1.0
    assert k["max_drawdown"] == -0.05
    assert k["xirr"] == 12.5


def test_derive_kpis_missing_inputs_are_nan():
    pdf = pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-01"]), "vlr_mercado": [1.0], "vlr_investido": [1.0],
         "proventos": [0.0], "fluxo": [1.0]}
    )
    k = derive_kpis(pdf, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float))["valor"]
    for name in ("twr_total", "cagr", "volatilidade", "sharpe", "max_drawdown", "xirr"):
        assert math.isnan(k[name])
    assert not np.isnan(k["patrimonio"])
