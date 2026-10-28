import datetime as dt

import pandas as pd
import pytest

from apps.market.synthetics import SyntheticSpec
from engine import dataset

pytestmark = pytest.mark.django_db


def _bench_series(start="2022-01-03", n=260, slope=0.03):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(
        [100 * (1 + slope) ** (i / n) for i in range(n)], index=idx, dtype=float
    )


def test_build_dataset_combines_user_and_benchmarks(monkeypatch):
    cdi = _bench_series(slope=0.10)
    selic = _bench_series(slope=0.11)

    def fake_get_series(name, s, e):
        return {"CDI": cdi, "SELIC": selic}.get(name)

    def fake_get_many(names, s, e):
        return {n: fake_get_series(n, s, e) for n in names if fake_get_series(n, s, e) is not None}

    monkeypatch.setattr(dataset.series, "get_series", fake_get_series)
    monkeypatch.setattr(dataset.series, "get_many", fake_get_many)

    user = _bench_series(slope=0.20)
    res = dataset.build_dataset(user_series=user, benchmarks=["CDI"])
    assert "Carteira" in res.df_combined.columns
    assert "CDI" in res.df_combined.columns
    # tudo rebased a 100 no início
    assert abs(res.df_combined["Carteira"].iloc[0] - 100) < 1e-6
    assert abs(res.df_combined["CDI"].iloc[0] - 100) < 1e-6
    assert res.selic_series is not None


def test_build_dataset_empty_does_not_raise_self_logger(monkeypatch, caplog):
    # regressão do bug financial_report.py:191 (self.logger inexistente)
    monkeypatch.setattr(dataset.series, "get_series", lambda *a, **k: None)
    monkeypatch.setattr(dataset.series, "get_many", lambda *a, **k: {})
    res = dataset.build_dataset(user_series=None, benchmarks=["CDI"],
                                start_date="2022-01-01", end_date="2022-12-31")
    assert res.df_combined.empty
    assert "Nenhum dado" in caplog.text  # logou via logging.getLogger, não quebrou


def test_build_dataset_extra_synthetic_spec(monkeypatch):
    ipca = _bench_series(slope=0.06)
    monkeypatch.setattr(dataset.series, "get_series",
                        lambda name, s, e: ipca if name == "IPCA" else None)
    monkeypatch.setattr(dataset.series, "get_many", lambda names, s, e: {})
    user = _bench_series(slope=0.15)
    spec = SyntheticSpec("IPCA", "+", 6.0)
    res = dataset.build_dataset(user_series=user, benchmarks=[], extra_specs=[spec])
    assert any("IPCA + 6" in c for c in res.df_combined.columns)


def test_fetch_user_portfolio_uses_dlp_and_twr(monkeypatch):
    hist = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "vlr_mercado": [100.0, 110.0, 104.5],
            "vlr_investido": [100.0, 100.0, 100.0],
            "proventos": [0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(dataset.dlp, "buscar_historico", lambda *a, **k: hist)
    twr, pdf = dataset.fetch_user_portfolio("DLP-tok")
    assert isinstance(twr, pd.Series) and not twr.empty
    assert isinstance(pdf, pd.DataFrame) and "fluxo" in pdf.columns


def test_fetch_user_portfolio_none_history_returns_none(monkeypatch):
    monkeypatch.setattr(dataset.dlp, "buscar_historico", lambda *a, **k: None)
    assert dataset.fetch_user_portfolio("DLP-tok") == (None, None)


@pytest.fixture
def mock_dlp_history(monkeypatch):
    hist = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]),
            "vlr_investido": [1000.0, 1000.0, 1500.0, 1500.0],
            "vlr_mercado": [1000.0, 1100.0, 1650.0, 1700.0],
            "lucro": [0.0, 0.0, 0.0, 0.0],
            "proventos": [0.0, 0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(dataset.dlp, "buscar_historico", lambda *a, **k: hist.copy())
    return hist


def test_end_date_cuts_portfolio_df(mock_dlp_history):
    twr, pdf = dataset.fetch_user_portfolio("t", end_date="2024-02-29")
    assert pdf is not None
    assert pdf["date"].max() == pd.Timestamp("2024-02-29") and len(pdf) == 2


def test_start_date_gap_sets_opening_flow_to_market_value(mock_dlp_history):
    twr, pdf = dataset.fetch_user_portfolio("t", start_date="2024-02-29")
    assert pdf is not None
    assert pdf["vlr_investido"].tolist() == [1100.0, 1600.0, 1600.0]
    assert pdf["fluxo"].tolist() == [1100.0, 500.0, 0.0]
    assert twr is not None
    assert twr.tolist() == pytest.approx([1.0, 1650 / 1600, 1650 / 1600 * 1700 / 1650])


def test_portfolio_df_carries_lucro(mock_dlp_history):
    twr, pdf = dataset.fetch_user_portfolio("t")
    assert pdf is not None
    assert "lucro" in pdf.columns


def test_window_outside_history_returns_none(mock_dlp_history):
    assert dataset.fetch_user_portfolio("t", start_date="2025-01-01") == (None, None)
    assert dataset.fetch_user_portfolio("t", end_date="2023-12-31") == (None, None)

