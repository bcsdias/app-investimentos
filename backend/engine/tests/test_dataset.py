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
    assert res.rf_series is not None


def test_build_dataset_fetches_cdi_as_rf(monkeypatch):
    called = []
    cdi = _bench_series(slope=0.10)

    def fake_get_series(name, s, e):
        called.append(name)
        return cdi if name == "CDI" else None

    def fake_get_many(names, s, e):
        return {}

    monkeypatch.setattr(dataset.series, "get_series", fake_get_series)
    monkeypatch.setattr(dataset.series, "get_many", fake_get_many)

    user = _bench_series(slope=0.20)
    res = dataset.build_dataset(user_series=user, benchmarks=[])
    assert "CDI" in called
    assert "SELIC" not in called
    assert res.rf_series is not None


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


def test_start_date_mid_month_opens_at_previous_close(mock_dlp_history):
    # Dado mensal: cada linha é um fechamento. "A partir de 01/03" parte do patrimônio
    # de 29/02 (último fechamento <= start), então o retorno de março entra no período.
    twr, pdf = dataset.fetch_user_portfolio("t", start_date="2024-03-01")
    assert pdf is not None
    assert pdf["date"].tolist() == list(pd.to_datetime(["2024-02-29", "2024-03-31", "2024-04-30"]))
    assert pdf["vlr_investido"].tolist() == [1100.0, 1600.0, 1600.0]
    assert pdf["fluxo"].tolist() == [1100.0, 500.0, 0.0]
    assert twr is not None
    assert twr.index[0] == pd.Timestamp("2024-02-29")
    assert twr.tolist() == pytest.approx([1.0, 1650 / 1600, 1700 / 1600])


def test_start_date_before_history_starts_at_first_row(mock_dlp_history):
    twr, pdf = dataset.fetch_user_portfolio("t", start_date="2023-06-01")
    assert pdf is not None
    assert len(pdf) == 4
    assert pdf["fluxo"].iloc[0] == 1000.0


def test_build_dataset_keeps_opening_close_before_start_date(monkeypatch):
    monkeypatch.setattr(dataset.series, "get_series", lambda *a, **k: None)
    monkeypatch.setattr(dataset.series, "get_many", lambda *a, **k: {})
    user = pd.Series([1.0, 1.05, 1.10], index=pd.to_datetime(["2024-02-29", "2024-03-31", "2024-04-30"]))
    res = dataset.build_dataset(user_series=user, start_date="2024-03-01", end_date="2024-04-30")
    assert res.df_combined.index[0] == pd.Timestamp("2024-02-29")
    assert res.df_combined["Carteira"].iloc[-1] == pytest.approx(110.0)


def test_portfolio_df_carries_lucro(mock_dlp_history):
    twr, pdf = dataset.fetch_user_portfolio("t")
    assert pdf is not None
    assert "lucro" in pdf.columns


def test_window_outside_history_returns_none(mock_dlp_history):
    assert dataset.fetch_user_portfolio("t", start_date="2025-01-01") == (None, None)
    assert dataset.fetch_user_portfolio("t", end_date="2023-12-31") == (None, None)


def test_benchmark_opens_at_last_value_before_portfolio_opening(monkeypatch):
    # CDI com pontos em 2023-12-29 (100.0), 2024-01-02 (101.0), 2024-01-31 (102.0);
    # carteira em 2023-12-31 e 2024-01-31; start_date="2024-01-01", end_date="2024-01-31"
    cdi_all = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-31"]),
    )

    def fake_get_many(names, s, e):
        s_ts = pd.to_datetime(s)
        e_ts = pd.to_datetime(e)
        return {"CDI": cdi_all[(cdi_all.index >= s_ts) & (cdi_all.index <= e_ts)]}

    monkeypatch.setattr(dataset.series, "get_many", fake_get_many)
    monkeypatch.setattr(dataset.series, "get_series", lambda *a, **k: None)

    user = pd.Series([1.0, 1.05], index=pd.to_datetime(["2023-12-31", "2024-01-31"]))
    res = dataset.build_dataset(
        user_series=user, benchmarks=["CDI"], start_date="2024-01-01", end_date="2024-01-31"
    )
    df = res.df_combined
    assert df.index[0] == pd.Timestamp("2023-12-31")
    assert df["CDI"].iloc[0] == pytest.approx(100.0)
    assert df["CDI"].iloc[-1] == pytest.approx(102.0)


def test_dataset_reports_benchmark_gap_inside_window(monkeypatch):
    from apps.market.gaps import SeriesGap

    idx1 = pd.date_range("2024-01-01", "2024-03-01", freq="D")
    idx2 = pd.date_range("2024-04-15", "2024-06-30", freq="D")
    cdi = pd.Series(range(len(idx1) + len(idx2)), index=idx1.union(idx2), dtype=float)

    monkeypatch.setattr(dataset.series, "get_many", lambda names, s, e: {"CDI": cdi})
    monkeypatch.setattr(dataset.series, "get_series", lambda *a, **k: None)

    res = dataset.build_dataset(benchmarks=["CDI"], start_date="2024-01-01", end_date="2024-06-30")
    assert res.gaps == [SeriesGap("CDI", dt.date(2024, 3, 1), dt.date(2024, 4, 15))]


def test_dataset_ignores_gap_outside_window(monkeypatch):
    idx1 = pd.date_range("2024-01-01", "2024-03-01", freq="D")
    idx2 = pd.date_range("2024-04-15", "2024-06-30", freq="D")
    cdi = pd.Series(range(len(idx1) + len(idx2)), index=idx1.union(idx2), dtype=float)

    monkeypatch.setattr(dataset.series, "get_many", lambda names, s, e: {"CDI": cdi})
    monkeypatch.setattr(dataset.series, "get_series", lambda *a, **k: None)

    res = dataset.build_dataset(benchmarks=["CDI"], start_date="2024-05-01", end_date="2024-06-30")
    assert res.gaps == []


def test_dataset_reports_gap_crossing_opening(monkeypatch):
    from apps.market.gaps import SeriesGap

    idx1 = pd.date_range("2024-01-01", "2024-03-01", freq="D")
    idx2 = pd.date_range("2024-04-15", "2024-06-30", freq="D")
    cdi = pd.Series(range(len(idx1) + len(idx2)), index=idx1.union(idx2), dtype=float)

    monkeypatch.setattr(dataset.series, "get_many", lambda names, s, e: {"CDI": cdi})
    monkeypatch.setattr(dataset.series, "get_series", lambda *a, **k: None)

    res = dataset.build_dataset(benchmarks=["CDI"], start_date="2024-04-01", end_date="2024-06-30")
    assert res.gaps == [SeriesGap("CDI", dt.date(2024, 3, 1), dt.date(2024, 4, 15))]


def test_dataset_does_not_include_portfolio_in_gaps(monkeypatch):
    user_idx = pd.to_datetime(["2024-01-01", "2024-06-30"])
    user = pd.Series([100.0, 110.0], index=user_idx)
    monkeypatch.setattr(dataset.series, "get_many", lambda names, s, e: {})
    monkeypatch.setattr(dataset.series, "get_series", lambda *a, **k: None)

    res = dataset.build_dataset(user_series=user, benchmarks=[], start_date="2024-01-01", end_date="2024-06-30")
    assert res.gaps == []


def _setup_dataset_mock(monkeypatch, series_map):
    calls = []

    def fake_get_series(name, s, e):
        s_date = s.date() if isinstance(s, (dt.datetime, pd.Timestamp)) else s
        e_date = e.date() if isinstance(e, (dt.datetime, pd.Timestamp)) else e
        calls.append((name, s_date, e_date))
        s_full = series_map.get(name)
        if s_full is None or s_full.empty:
            return None
        s_ts = pd.to_datetime(s_date)
        e_ts = pd.to_datetime(e_date)
        sliced = s_full[(s_full.index >= s_ts) & (s_full.index <= e_ts)]
        return sliced if not sliced.empty else None

    def fake_get_many(names, s, e):
        out = {}
        for n in names:
            val = fake_get_series(n, s, e)
            if val is not None and not val.empty:
                out[n] = val
        return out

    monkeypatch.setattr(dataset, "_today", lambda: dt.date(2026, 10, 29))
    monkeypatch.setattr(dataset.series, "get_series", fake_get_series)
    monkeypatch.setattr(dataset.series, "get_many", fake_get_many)
    return calls


def test_gap_crossing_opening_beyond_lookback_is_reported_and_ffilled(monkeypatch):
    from apps.market.gaps import SeriesGap

    idx = [pd.Timestamp("2023-05-22"), pd.Timestamp("2024-02-21")] + list(
        pd.date_range("2024-02-22", "2024-12-31", freq="D")
    )
    vals = [100.0, 101.0] + [101.0 + i * 0.01 for i in range(1, len(idx) - 1)]
    imas = pd.Series(vals, index=pd.to_datetime(idx))

    _setup_dataset_mock(monkeypatch, {"IMA-S": imas})
    carteira = pd.Series([1.0, 1.1], index=pd.to_datetime(["2023-12-31", "2024-12-31"]))

    res = dataset.build_dataset(
        user_series=carteira, benchmarks=["IMA-S"], start_date="2024-01-01", end_date="2024-12-31"
    )
    assert SeriesGap("IMA-S", dt.date(2023, 5, 22), dt.date(2024, 2, 21)) in res.gaps
    assert res.df_combined["IMA-S"].iloc[0] == pytest.approx(100.0)


def test_gap_crossing_window_end_is_reported(monkeypatch):
    from apps.market.gaps import SeriesGap

    idx1 = pd.date_range("2023-12-01", "2024-09-26", freq="D")
    idx2 = pd.date_range("2025-02-24", "2026-10-29", freq="D")
    idx = idx1.union(idx2)
    lfts = pd.Series(range(len(idx)), index=idx, dtype=float)

    _setup_dataset_mock(monkeypatch, {"LFTS11.SA": lfts})
    carteira = pd.Series([1.0, 1.1], index=pd.to_datetime(["2023-12-31", "2024-12-31"]))

    res = dataset.build_dataset(
        user_series=carteira, benchmarks=["LFTS11.SA"], start_date="2024-01-01", end_date="2024-12-31"
    )
    assert SeriesGap("LFTS11.SA", dt.date(2024, 9, 26), dt.date(2025, 2, 24)) in res.gaps


def test_series_stale_until_today_reports_tail(monkeypatch):
    from apps.market.gaps import SeriesGap

    idx = pd.date_range("2023-01-02", "2024-04-10", freq="D")
    imas = pd.Series(range(len(idx)), index=idx, dtype=float)

    _setup_dataset_mock(monkeypatch, {"IMA-S": imas})
    carteira = pd.Series([1.0, 1.1], index=pd.to_datetime(["2023-12-31", "2024-12-31"]))

    res = dataset.build_dataset(
        user_series=carteira, benchmarks=["IMA-S"], start_date="2024-01-01", end_date="2024-12-31"
    )
    assert SeriesGap("IMA-S", dt.date(2024, 4, 10), dt.date(2024, 12, 31)) in res.gaps


def test_series_starting_after_opening_has_no_gap_and_refetches_once(monkeypatch):
    idx = pd.date_range("2024-06-03", "2024-12-31", freq="D")
    etf = pd.Series(range(len(idx)), index=idx, dtype=float)

    calls = _setup_dataset_mock(monkeypatch, {"ETF": etf})
    carteira = pd.Series([1.0, 1.1], index=pd.to_datetime(["2023-12-31", "2024-12-31"]))

    res = dataset.build_dataset(
        user_series=carteira, benchmarks=["ETF"], start_date="2024-01-01", end_date="2024-12-31"
    )
    assert res.gaps == []
    etf_calls = [c for c in calls if c[0] == "ETF"]
    assert len(etf_calls) == 2
    from apps.market import catalog
    assert etf_calls[0][1] == dt.date(2023, 11, 21)
    assert etf_calls[0][2] == dt.date(2026, 10, 29)
    assert etf_calls[1][1] == catalog.EPOCH
    assert etf_calls[1][2] == dt.date(2026, 10, 29)




