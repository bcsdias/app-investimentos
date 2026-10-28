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

