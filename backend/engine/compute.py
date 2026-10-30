"""Funções de cálculo PURAS extraídas dos plot_* de src/engine/financial_report.py.

Cada função recebe um DataFrame/Series já pronto e devolve DataFrame/Series/dict
JSON-serializável. Zero matplotlib, zero I/O. A serialização para o formato do
ECharts é da Fase 3 (apps/api/serializers.py).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from engine.irr import calculate_xirr, period_return
from engine.metrics import (
    calculate_drawdown,
    calculate_rolling_sharpe,
    calculate_rolling_volatility,
)


def compute_twr_evolution(df_combined: pd.DataFrame) -> dict:
    if df_combined.empty:
        return {"series": df_combined, "final_returns": {}}
    order = df_combined.iloc[-1].sort_values(ascending=False).index
    series = df_combined[order]
    final_returns = {col: float(series[col].iloc[-1] - 100) for col in series.columns}
    return {"series": series, "final_returns": final_returns}


def compute_drawdown(df_combined: pd.DataFrame) -> pd.DataFrame:
    return calculate_drawdown(df_combined)


def compute_risk_return(
    df_combined: pd.DataFrame, rf_series: pd.Series | None = None
) -> pd.DataFrame:
    if df_combined.empty:
        return pd.DataFrame(columns=["Volatilidade", "Retorno (CAGR)", "Sharpe"])
    daily_ret = df_combined.pct_change(fill_method=None)
    volatility = daily_ret.std() * np.sqrt(252)
    days = (df_combined.index[-1] - df_combined.index[0]).days
    total_ret = df_combined.iloc[-1] / df_combined.iloc[0]
    cagr = (total_ret ** (365.25 / days)) - 1 if days > 0 else total_ret * 0

    cagr_rf = np.nan
    if rf_series is not None and not rf_series.empty and days > 0:
        rf = rf_series.copy()
        rf.index = pd.to_datetime(rf.index)
        rf = rf.sort_index()
        rf = rf.loc[(rf.index >= df_combined.index[0]) & (rf.index <= df_combined.index[-1])].dropna()
        if len(rf) >= 2 and rf.iloc[0] != 0:
            cagr_rf = (rf.iloc[-1] / rf.iloc[0]) ** (365.25 / days) - 1

    if np.isnan(cagr_rf):
        sharpe = pd.Series(np.nan, index=df_combined.columns)
    else:
        # volatilidade zero (série constante/achatada) -> Sharpe indefinido, não inf
        sharpe = (cagr - cagr_rf) / volatility.where(volatility != 0)

    res = pd.DataFrame(
        {
            "Volatilidade": volatility,
            "Retorno (CAGR)": cagr,
            "Sharpe": sharpe,
        }
    ).replace([np.inf, -np.inf], np.nan)
    valid = (res["Volatilidade"].notna()) & (res["Volatilidade"] != 0) & (res["Retorno (CAGR)"].notna())
    return res[valid]


def compute_rolling_metrics(
    df_combined: pd.DataFrame, rf_series: pd.Series | None, window: int = 252
) -> dict:
    return {
        "volatility": calculate_rolling_volatility(df_combined, window),
        "sharpe": calculate_rolling_sharpe(
            df_combined, rf_series, window=window
        ),
    }


def compute_irr_evolution(portfolio_df: pd.DataFrame) -> pd.Series:
    if portfolio_df is None or portfolio_df.empty:
        return pd.Series(dtype=float)
    df = portfolio_df.sort_values("date")
    dates_to_calc = df.set_index("date").resample("ME").last().index
    if df["date"].iloc[-1] not in dates_to_calc:
        dates_to_calc = dates_to_calc.union([df["date"].iloc[-1]])
    dates_to_calc = dates_to_calc[dates_to_calc >= df["date"].iloc[0]]

    all_dates = df["date"].values
    all_flows = -1 * df["fluxo"].values
    all_markets = df["vlr_mercado"].values

    irr_history, valid_dates = [], []
    for target in dates_to_calc:
        mask = all_dates <= target
        if not np.any(mask):
            continue
        last_idx = np.where(mask)[0][-1]
        if all_markets[last_idx] == 0 and np.sum(np.abs(all_flows[mask])) == 0:
            continue
        calc_flows = np.append(all_flows[mask], all_markets[last_idx])
        calc_dates = np.append(all_dates[mask], all_dates[last_idx])
        try:
            res = calculate_xirr(calc_flows, pd.to_datetime(calc_dates))
        except Exception:  # noqa: BLE001
            res = None
        if res is not None and -0.99 < res < 10.0:
            irr_history.append(res)
            valid_dates.append(target)

    if not irr_history:
        return pd.Series(dtype=float)
    return pd.Series(irr_history, index=valid_dates) * 100


IRR_ANNUAL_AFTER = pd.DateOffset(months=12)


def irr_period_series(irr: pd.Series, opening: pd.Timestamp | str) -> pd.Series:
    """Converte cada ponto da série de TIR anualizada para TIR acumulada no período."""
    if irr is None or irr.empty:
        return pd.Series(dtype=float)
    opening_dt = pd.to_datetime(opening)
    res = {}
    for d, val in irr.items():
        if pd.isna(val) or val is None:
            res[d] = np.nan
        else:
            p = period_return(float(val) / 100.0, opening_dt, pd.to_datetime(d))
            res[d] = float(p * 100.0) if p is not None else np.nan
    return pd.Series(res, index=irr.index, dtype=float)


def compute_annual_from(irr: pd.Series, opening: pd.Timestamp | str) -> str | None:
    """Retorna a 1ª data (AAAA-MM-DD) da série de TIR >= abertura + 12 meses, ou None."""
    if irr is None or irr.empty or opening is None:
        return None
    cutoff = pd.to_datetime(opening) + IRR_ANNUAL_AFTER
    for d in irr.index:
        d_dt = pd.to_datetime(d)
        if d_dt >= cutoff:
            return d_dt.strftime("%Y-%m-%d")
    return None


def compute_summary_irr(
    portfolio_df: pd.DataFrame, irr: pd.Series | None = None
) -> dict[str, Any]:
    """Calcula campos de TIR para o resumo do portfólio."""
    if portfolio_df is None or portfolio_df.empty:
        return {"xirr": None, "xirr_period": None, "xirr_is_annual": False}

    pdf = portfolio_df.sort_values("date") if "date" in portfolio_df.columns else portfolio_df
    if irr is None:
        irr = compute_irr_evolution(pdf)

    if irr.empty:
        return {"xirr": None, "xirr_period": None, "xirr_is_annual": False}

    opening = pd.to_datetime(pdf["date"].iloc[0])
    fim = pd.to_datetime(pdf["date"].iloc[-1])
    is_annual = fim >= (opening + IRR_ANNUAL_AFTER)
    xirr = float(irr.iloc[-1])
    p = period_return(xirr / 100.0, opening, fim)
    xirr_period = float(p * 100.0) if p is not None else None
    return {
        "xirr": xirr,
        "xirr_period": xirr_period,
        "xirr_is_annual": bool(is_annual),
    }



def compute_summary_kpis(portfolio_df: pd.DataFrame) -> dict[str, float]:
    """Calcula os KPIs de resumo do portfólio (retorno total).

    Retorna chaves:
        patrimonio_inicial: float
        aportes_periodo: float (aportes líquidos de resgates, proventos e lucro realizado)
        patrimonio: float
        ganho_periodo: float (retorno total = patrimonio - patrimonio_inicial - aportes_periodo)
    """
    if portfolio_df is None or portfolio_df.empty:
        return {
            "patrimonio_inicial": 0.0,
            "aportes_periodo": 0.0,
            "patrimonio": 0.0,
            "ganho_periodo": 0.0,
        }

    pdf = portfolio_df.sort_values("date") if "date" in portfolio_df.columns else portfolio_df
    pat_ini = float(pdf["vlr_mercado"].iloc[0])
    pat_fim = float(pdf["vlr_mercado"].iloc[-1])

    if len(pdf) > 1 and "fluxo" in pdf.columns:
        aportes = float(pdf["fluxo"].iloc[1:].sum())
    else:
        aportes = 0.0

    ganho = pat_fim - pat_ini - aportes
    return {
        "patrimonio_inicial": pat_ini,
        "aportes_periodo": aportes,
        "patrimonio": pat_fim,
        "ganho_periodo": ganho,
    }


def compute_summary(
    df_combined: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    rf_series: pd.Series | None = None,
) -> dict[str, Any]:
    """Calcula os indicadores resumidos da carteira (KPIs, TWR, risco/retorno, drawdown, TIR)."""
    summary_kpis = compute_summary_kpis(portfolio_df)

    twr_out = compute_twr_evolution(df_combined)
    twr_total = twr_out["final_returns"].get("Carteira")

    rr_df = compute_risk_return(df_combined, rf_series)
    if "Carteira" in rr_df.index:
        cagr = rr_df.loc["Carteira", "Retorno (CAGR)"]
        vol = rr_df.loc["Carteira", "Volatilidade"]
        sharpe = rr_df.loc["Carteira", "Sharpe"]
    else:
        cagr = vol = sharpe = None

    dd_df = compute_drawdown(df_combined)
    max_dd = dd_df["Carteira"].min() if "Carteira" in dd_df.columns else None

    irr_summary = compute_summary_irr(portfolio_df)
    xirr = irr_summary["xirr"]
    xirr_period = irr_summary["xirr_period"]
    xirr_is_annual = irr_summary["xirr_is_annual"]

    return {
        "patrimonio_inicial": summary_kpis["patrimonio_inicial"],
        "aportes_periodo": summary_kpis["aportes_periodo"],
        "patrimonio": summary_kpis["patrimonio"],
        "ganho_periodo": summary_kpis["ganho_periodo"],
        "twr_total": twr_total,
        "cagr": cagr,
        "volatilidade": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "xirr": xirr,
        "xirr_period": xirr_period,
        "xirr_is_annual": xirr_is_annual,
    }


def compute_shadow(portfolio_df: pd.DataFrame, df_combined: pd.DataFrame) -> pd.DataFrame:
    if df_combined.empty or portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame()

    pdf = portfolio_df.sort_values("date") if "date" in portfolio_df.columns else portfolio_df
    idx = df_combined.index

    # Saldo inicial no primeiro ponto da carteira
    initial_balance = float(pdf["vlr_mercado"].iloc[0])

    # Fluxos subsequentes (a partir da linha 1)
    if len(pdf) > 1 and "fluxo" in pdf.columns:
        sub_flows = pdf.iloc[1:].set_index("date")["fluxo"]
        sub_flows.index = pd.to_datetime(sub_flows.index)
        sub_flows_aligned = sub_flows.groupby(sub_flows.index).sum().reindex(idx, fill_value=0.0)
    else:
        sub_flows_aligned = pd.Series(0.0, index=idx)

    sw = pd.DataFrame(index=idx)

    # Carteira Real
    vlr_mkt = pdf.set_index("date")["vlr_mercado"]
    vlr_mkt.index = pd.to_datetime(vlr_mkt.index)
    sw["Carteira Real"] = vlr_mkt.groupby(vlr_mkt.index).last().reindex(idx).ffill()

    # Total Investido: vlr_mercado[0] + cumsum(fluxo[1:])
    sw["Total Investido"] = initial_balance + sub_flows_aligned.cumsum()

    # Benchmarks compram cotas com saldo inicial no t0 e depois só com fluxos subsequentes
    for col in df_combined.columns:
        if col == "Carteira":
            continue
        price = df_combined[col]
        initial_shares = initial_balance / price.iloc[0] if price.iloc[0] != 0 else 0.0
        shares_flow = sub_flows_aligned.div(price).fillna(0.0)
        sw[col] = (initial_shares + shares_flow.cumsum()) * price

    return sw


def compute_yearly_summary(df_combined: pd.DataFrame) -> pd.DataFrame:
    if df_combined.empty:
        return pd.DataFrame()
    df = df_combined
    total_ret = (df.iloc[-1] / df.iloc[0]) - 1
    yearly = df.resample("YE").last()
    yearly_ret = yearly.pct_change(fill_method=None)
    yearly_ret.iloc[0] = (yearly.iloc[0] / df.iloc[0]) - 1

    idx = pd.to_datetime(df.index)
    first_year = idx[0].year
    if (idx.year == first_year).sum() == 1 and (idx.year > first_year).any():
        yearly_ret = yearly_ret.iloc[1:]

    summary = yearly_ret.T
    summary.columns = [c.year for c in summary.columns]
    summary["Total Acum."] = total_ret
    return summary
