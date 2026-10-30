import numpy as np
import pandas as pd
import pytest

from engine import compute


def _combined():
    idx = pd.bdate_range("2022-01-03", periods=520)  # ~2 anos úteis
    carteira = pd.Series(np.linspace(100, 130, len(idx)), index=idx, name="Carteira")
    cdi = pd.Series(np.linspace(100, 120, len(idx)), index=idx, name="CDI")
    return pd.concat([carteira, cdi], axis=1)


def test_twr_evolution_sorts_and_reports_final_returns():
    df = _combined()
    out = compute.compute_twr_evolution(df)
    assert list(out["series"].columns)[0] == "Carteira"  # maior último valor primeiro
    assert round(out["final_returns"]["Carteira"], 1) == 30.0
    assert round(out["final_returns"]["CDI"], 1) == 20.0


def test_drawdown_is_nonpositive_and_starts_at_zero():
    df = _combined()
    dd = compute.compute_drawdown(df)
    assert (dd.fillna(0) <= 1e-9).all().all()
    assert abs(dd["CDI"].iloc[0]) < 1e-9


def test_risk_return_columns_and_sharpe_formula():
    df = _combined()
    cdi = df["CDI"]
    rr = compute.compute_risk_return(df, rf_series=cdi)
    assert list(rr.columns) == ["Volatilidade", "Retorno (CAGR)", "Sharpe"]
    row = rr.loc["Carteira"]
    days = (df.index[-1] - df.index[0]).days
    cdi_cagr = (cdi.iloc[-1] / cdi.iloc[0]) ** (365.25 / days) - 1
    np.testing.assert_allclose(
        row["Sharpe"], (row["Retorno (CAGR)"] - cdi_cagr) / row["Volatilidade"], rtol=1e-9
    )


def test_risk_return_uses_rf_cagr():
    df = _combined()
    cdi = df["CDI"]
    rr = compute.compute_risk_return(df, rf_series=cdi)
    days = (df.index[-1] - df.index[0]).days
    cdi_cagr = (cdi.iloc[-1] / cdi.iloc[0]) ** (365.25 / days) - 1
    expected_sharpe = (rr.loc["Carteira", "Retorno (CAGR)"] - cdi_cagr) / rr.loc["Carteira", "Volatilidade"]
    np.testing.assert_allclose(rr.loc["Carteira", "Sharpe"], expected_sharpe, rtol=1e-9)


def test_risk_return_without_rf_keeps_row_with_nan_sharpe():
    df = _combined()
    rr = compute.compute_risk_return(df, rf_series=None)
    assert "Carteira" in rr.index and np.isnan(rr.loc["Carteira", "Sharpe"])


def test_risk_return_drops_constant_series_instead_of_infinite_sharpe():
    idx = pd.bdate_range("2022-01-03", periods=520)
    df = pd.concat(
        [
            pd.Series(np.linspace(100, 130, len(idx)), index=idx, name="Carteira"),
            pd.Series(100.0, index=idx, name="Flat"),  # volatilidade zero
        ],
        axis=1,
    )
    cdi = pd.Series(np.linspace(100, 120, len(idx)), index=idx, name="CDI")
    rr = compute.compute_risk_return(df, rf_series=cdi)
    assert "Flat" not in rr.index  # linha com vol=0 sai, não vira Sharpe = inf
    assert np.isfinite(rr["Sharpe"]).all()


def test_rolling_metrics_returns_both_frames():
    df = _combined()
    out = compute.compute_rolling_metrics(df, rf_series=None, window=252)
    assert set(out) == {"volatility", "sharpe"}
    assert isinstance(out["volatility"], pd.DataFrame)
    assert isinstance(out["sharpe"], pd.DataFrame)


def test_irr_evolution_positive_for_growing_portfolio():
    dates = pd.to_datetime(["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"])
    pdf = pd.DataFrame(
        {
            "date": dates,
            "vlr_mercado": [1000.0, 1600.0, 2200.0, 2900.0],
            "vlr_investido": [1000.0, 1500.0, 2000.0, 2500.0],
            "proventos": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pdf["fluxo"] = pdf["vlr_investido"].diff().fillna(pdf["vlr_investido"].iloc[0]) - pdf["proventos"]
    s = compute.compute_irr_evolution(pdf)
    assert isinstance(s, pd.Series)
    assert not s.empty
    assert s.iloc[-1] > 0


def test_shadow_has_real_and_invested_columns():
    df = _combined()
    dates = df.index[:: len(df) // 4][:4]
    pdf = pd.DataFrame(
        {
            "date": dates,
            "vlr_mercado": [1000.0, 1200.0, 1500.0, 1800.0],
            "vlr_investido": [1000.0, 1100.0, 1300.0, 1500.0],
            "proventos": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pdf["fluxo"] = pdf["vlr_investido"].diff().fillna(pdf["vlr_investido"].iloc[0]) - pdf["proventos"]
    sw = compute.compute_shadow(pdf, df)
    assert "Carteira Real" in sw.columns
    assert "Total Investido" in sw.columns
    assert "CDI" in sw.columns


def test_yearly_summary_has_years_and_total():
    df = _combined()
    ys = compute.compute_yearly_summary(df)
    assert "Total Acum." in ys.columns
    assert any(isinstance(c, int) or str(c).isdigit() for c in ys.columns)
    assert pd.api.types.is_numeric_dtype(ys["Total Acum."])


def test_yearly_drops_opening_only_year():
    df = pd.DataFrame(
        {"Carteira": [100.0, 105.0, 110.0]},
        index=pd.to_datetime(["2023-12-31", "2024-06-30", "2024-12-31"]),
    )
    out = compute.compute_yearly_summary(df)
    assert "2023" not in [str(c) for c in out.columns]
    assert out.loc["Carteira"].iloc[0] == pytest.approx(0.10)


def test_yearly_keeps_single_row_when_no_following_year():
    df = pd.DataFrame(
        {"Carteira": [100.0]},
        index=pd.to_datetime(["2023-12-31"]),
    )
    out = compute.compute_yearly_summary(df)
    assert "2023" in [str(c) for c in out.columns]
    assert out.loc["Carteira"].iloc[0] == pytest.approx(0.0)


PDF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
        "vlr_mercado": [1000.0, 700.0, 720.0],
        "vlr_investido": [1000.0, 600.0, 600.0],
        "lucro": [0.0, 200.0, 0.0],
        "proventos": [0.0, 0.0, 10.0],
        "fluxo": [1000.0, -600.0, -10.0],
    }
)


def test_summary_kpis_total_return():
    k = compute.compute_summary_kpis(PDF)
    # ganho = lucro 200 + valorização aberta 100 + 20 + provento 10
    assert k == pytest.approx(
        {
            "patrimonio_inicial": 1000.0,
            "aportes_periodo": -610.0,
            "patrimonio": 720.0,
            "ganho_periodo": 330.0,
        }
    )


def test_summary_kpis_empty_or_single_row():
    assert compute.compute_summary_kpis(pd.DataFrame()) == {
        "patrimonio_inicial": 0.0,
        "aportes_periodo": 0.0,
        "patrimonio": 0.0,
        "ganho_periodo": 0.0,
    }
    assert compute.compute_summary_kpis(PDF.iloc[:1]) == pytest.approx(
        {
            "patrimonio_inicial": 1000.0,
            "aportes_periodo": 0.0,
            "patrimonio": 1000.0,
            "ganho_periodo": 0.0,
        }
    )


def test_shadow_does_not_double_count_opening_balance():
    combined = pd.DataFrame(
        {"Carteira": [100.0, 130.0, 135.0], "CDI": [100.0] * 3}, index=PDF["date"]
    )
    sw = compute.compute_shadow(PDF, combined)
    assert sw["Total Investido"].tolist() == [1000.0, 400.0, 390.0]
    assert sw["CDI"].tolist() == pytest.approx([1000.0, 400.0, 390.0])


def test_irr_annual_after_constant():
    assert compute.IRR_ANNUAL_AFTER == pd.DateOffset(months=12)


def test_irr_period_series_converts_points():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2025-01-31"])
    opening = pd.to_datetime("2024-01-31")
    # 10% a.a. constante em porcentagem
    irr_series = pd.Series([0.0, 10.0, 10.0], index=idx)
    res = compute.irr_period_series(irr_series, opening)
    assert len(res) == 3
    # 2024-01-31: 0 dias -> 0%
    assert res.iloc[0] == 0.0
    # 2024-02-29: 29 dias de 10% a.a. -> (1.10)**(29/365) - 1 ≈ 0.76%
    assert res.iloc[1] == pytest.approx(0.76, abs=1e-2)
    # 2025-01-31: 366 dias (bissexto) -> próximo a 10.03%
    assert res.iloc[2] == pytest.approx(10.03, rel=1e-2)


def test_compute_annual_from():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-12-31", "2025-01-31", "2025-02-28"])
    s = pd.Series([1.0] * 5, index=dates)
    opening = pd.to_datetime("2024-01-31")
    # 12 meses após 2024-01-31 é 2025-01-31
    assert compute.compute_annual_from(s, opening) == "2025-01-31"

    # Janela curta (< 12 meses) -> None
    short_s = pd.Series([1.0] * 3, index=dates[:3])
    assert compute.compute_annual_from(short_s, opening) is None


def test_compute_summary_irr_short_vs_long():
    short_pdf = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "fluxo": [1000.0, 0.0],
            "vlr_mercado": [1000.0, 970.0],
        }
    )
    short_out = compute.compute_summary_irr(short_pdf)
    assert short_out["xirr_is_annual"] is False
    assert short_out["xirr"] is not None
    assert short_out["xirr_period"] == pytest.approx(-3.0, abs=1e-2)

    # 13 meses
    dates = pd.date_range("2024-01-31", periods=14, freq="ME")
    long_pdf = pd.DataFrame(
        {
            "date": dates,
            "fluxo": [1000.0] + [0.0] * 13,
            "vlr_mercado": [1000.0 * (1.01 ** i) for i in range(14)],
        }
    )
    long_out = compute.compute_summary_irr(long_pdf)
    assert long_out["xirr_is_annual"] is True
    assert long_out["xirr"] is not None
    assert long_out["xirr_period"] is not None


def test_compute_summary_keys():
    df = _combined()
    cdi = df["CDI"]
    dates = df.index[:: len(df) // 4][:4]
    pdf = pd.DataFrame(
        {
            "date": dates,
            "vlr_mercado": [1000.0, 1200.0, 1500.0, 1800.0],
            "vlr_investido": [1000.0, 1100.0, 1300.0, 1500.0],
            "proventos": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pdf["fluxo"] = pdf["vlr_investido"].diff().fillna(pdf["vlr_investido"].iloc[0]) - pdf["proventos"]
    out = compute.compute_summary(df, pdf, rf_series=cdi)
    expected_keys = {
        "patrimonio_inicial",
        "aportes_periodo",
        "patrimonio",
        "ganho_periodo",
        "twr_total",
        "cagr",
        "volatilidade",
        "sharpe",
        "max_drawdown",
        "xirr",
        "xirr_period",
        "xirr_is_annual",
    }
    assert set(out.keys()) == expected_keys
    assert out["patrimonio_inicial"] == 1000.0
    assert out["patrimonio"] == 1800.0
    assert out["twr_total"] is not None
    assert out["xirr"] is not None
    assert out["xirr_period"] is not None
    assert isinstance(out["xirr_is_annual"], (bool, np.bool_))

