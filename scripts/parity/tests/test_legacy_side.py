import logging.handlers

import numpy as np
import pandas as pd

from apps.market import catalog
from engine.compute import compute_yearly_summary
from parity.cases import BENCHMARKS, build_cases
from parity.compare import FALHA, OK, REL_TOL_L1, compare_frames
from parity.engine_side import run_engine
from parity.frozen import MarketSnapshot
from parity.legacy_side import legacy_patches, legacy_yearly, run_layer1, run_legacy
from parity.tests.synthetic import TODAY, make_hist, make_snapshot


def _hist_for(case, hist):
    if case.classe is None:
        return hist
    return hist[hist["classe"] == case.classe].reset_index(drop=True)


def test_benchmarks_exist_in_both_catalogs():
    from src.data.benchmarks_config import CATALOGO_B3, CATALOGO_BCB, CATALOGO_YF

    for name in BENCHMARKS:
        catalog.resolve(name)
    assert "CDI" in CATALOGO_BCB and "IPCA" in CATALOGO_BCB
    assert "IFIX" in CATALOGO_B3
    assert "S&P 500" in CATALOGO_YF


def test_isolate_legacy_disables_upstash_and_file_log():
    import os

    from src.data import cache as legacy_cache
    from src.utils.logger import logger as legacy_logger

    assert legacy_cache._get_redis_client() is None
    assert "UPSTASH_REDIS_REST_URL" not in os.environ
    assert "UPSTASH_REDIS_REST_TOKEN" not in os.environ
    assert not any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in legacy_logger.handlers
    )


def test_legacy_patches_restore_financial_report_on_first_import(monkeypatch):
    # Se financial_report só for importado dentro do patch, o seu
    # `from market_data import buscar_historico` captura o fake e o mock
    # "restaura" para o fake — a camada 2 passaria a usar o histórico congelado.
    import sys

    from src.data.sources import market_data

    monkeypatch.delitem(sys.modules, "src.engine.financial_report", raising=False)
    with legacy_patches(make_snapshot(), make_hist()):
        pass
    fr = sys.modules["src.engine.financial_report"]
    assert fr.buscar_historico is market_data.buscar_historico
    assert market_data.buscar_historico.__module__ == "src.data.sources.market_data"


def test_fake_bcb_rate_matches_legacy_accumulation():
    from src.data.sources import market_data

    snap = make_snapshot()
    with legacy_patches(snap, make_hist()):
        got = market_data.buscar_dados_bcb(12, "2024-01-02", "2024-03-01", nome="CDI")
    full = ((snap.series["bcb:12"] / 100) + 1).cumprod().fillna(1)
    expected = full.loc["2024-01-02":"2024-03-01"]
    assert got.index.name == "Date"
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=0, atol=0)
    assert list(got.index) == list(expected.index)


def test_fake_bcb_index_code_returns_level():
    from src.data.sources import market_data

    idx = pd.bdate_range("2024-01-01", periods=4)
    snap = MarketSnapshot(series={"bcb:12466": pd.Series([10.0, np.nan, 12.0, 13.0], index=idx)})
    with legacy_patches(snap, make_hist()):
        got = market_data.buscar_dados_bcb(12466, "2024-01-01", "2024-01-31")
    assert got.tolist() == [10.0, 10.0, 12.0, 13.0]


def test_fake_returns_none_when_empty():
    from src.data.sources import market_data

    snap = make_snapshot()
    with legacy_patches(snap, make_hist()):
        assert market_data.buscar_dados_b3("IFIX", "1999-01-01", "1999-12-31") is None
        assert market_data.buscar_dolar_bcb("1999-01-01", "1999-12-31") is None
        assert market_data.buscar_dados_benchmark("^GSPC", "1999-01-01", "1999-12-31") is None
        assert market_data.buscar_dados_tesouro("Tesouro IPCA+", "15/05/2035", "2024-01-01", "2024-12-31") is None


def test_legacy_yearly_matches_engine_on_ffilled_frame():
    idx = pd.bdate_range("2023-03-01", "2025-06-27")
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {"Carteira": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
         "CDI": 100 * np.cumprod(np.full(len(idx), 1.0004))},
        index=idx,
    )
    pd.testing.assert_frame_equal(legacy_yearly(df), compute_yearly_summary(df))


def test_layer1_parity_on_synthetic_data():
    hist = make_hist()
    snap = make_snapshot()
    cases = build_cases(hist, TODAY)
    assert len(cases) == 4
    for case in cases:
        diffs, legacy_out, new_out = run_layer1(case, snap, _hist_for(case, hist))
        assert legacy_out and new_out
        bad = [d for d in diffs if d.status != OK]
        assert not bad, (case.id, bad)


def test_layer1_detects_injected_difference():
    hist = make_hist()
    snap = make_snapshot()
    case = build_cases(hist, TODAY)[0]

    tampered = MarketSnapshot(series=dict(snap.series))
    cdi = snap.series["bcb:12"].copy()
    cdi.loc["2024-01-02":] *= 1.001
    tampered.series["bcb:12"] = cdi

    with legacy_patches(snap, hist):
        _, legacy_out = run_legacy("parity", case)
    new_out = run_engine(case, tampered, hist)
    diffs = {d.column: d for d in compare_frames("twr", legacy_out["twr"], new_out["twr"], REL_TOL_L1, FALHA)}
    assert diffs["CDI"].status == FALHA
    assert diffs["CDI"].first_date >= "2024-01-02"
    assert diffs["Carteira"].status == OK
