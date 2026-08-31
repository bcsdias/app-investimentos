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
    rr = compute.compute_risk_return(df)
    assert list(rr.columns) == ["Volatilidade", "Retorno (CAGR)", "Sharpe"]
    row = rr.loc["Carteira"]
    np.testing.assert_allclose(
        row["Sharpe"], (row["Retorno (CAGR)"] - 0.10) / row["Volatilidade"], rtol=1e-9
    )


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
