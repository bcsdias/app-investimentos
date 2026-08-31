"""Funções de cálculo PURAS extraídas dos plot_* de src/engine/financial_report.py.

Cada função recebe um DataFrame/Series já pronto e devolve DataFrame/Series/dict
JSON-serializável. Zero matplotlib, zero I/O. A serialização para o formato do
ECharts é da Fase 3 (apps/api/serializers.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.irr import calculate_xirr
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


def compute_risk_return(df_combined: pd.DataFrame) -> pd.DataFrame:
    if df_combined.empty:
        return pd.DataFrame(columns=["Volatilidade", "Retorno (CAGR)", "Sharpe"])
    daily_ret = df_combined.pct_change(fill_method=None)
    volatility = daily_ret.std() * np.sqrt(252)
    days = (df_combined.index[-1] - df_combined.index[0]).days
    total_ret = df_combined.iloc[-1] / df_combined.iloc[0]
    cagr = (total_ret ** (365.25 / days)) - 1 if days > 0 else total_ret * 0
    # volatilidade zero (série constante/achatada) -> Sharpe indefinido, não inf
    sharpe = (cagr - 0.10) / volatility.where(volatility != 0)
    return pd.DataFrame(
        {
            "Volatilidade": volatility,
            "Retorno (CAGR)": cagr,
            "Sharpe": sharpe,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()


def compute_rolling_metrics(
    df_combined: pd.DataFrame, rf_series: pd.Series | None, window: int = 252
) -> dict:
    return {
        "volatility": calculate_rolling_volatility(df_combined, window),
        "sharpe": calculate_rolling_sharpe(
            df_combined, rf_series, rf_constant=0.10, window=window
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


def compute_shadow(portfolio_df: pd.DataFrame, df_combined: pd.DataFrame) -> pd.DataFrame:
    if df_combined.empty or portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame()

    flows = portfolio_df.set_index("date")[["fluxo", "vlr_mercado", "vlr_investido"]]
    flows.index = pd.to_datetime(flows.index)
    prices = df_combined.copy()
    idx = prices.index

    flows_aligned = flows["fluxo"].groupby(flows.index).sum().reindex(idx, fill_value=0.0)
    sw = pd.DataFrame(index=idx)
    sw["Carteira Real"] = flows["vlr_mercado"].reindex(idx).ffill()
    sw["Total Investido"] = flows_aligned.cumsum() + (
        flows["vlr_investido"].iloc[0] if not flows.empty else 0
    )

    initial_balance = flows["vlr_mercado"].iloc[0] if not flows.empty else 0
    for col in prices.columns:
        if col == "Carteira":
            continue
        price = prices[col]
        shares_flow = flows_aligned.div(price).fillna(0)
        initial_shares = initial_balance / price.iloc[0] if price.iloc[0] != 0 else 0
        sw[col] = (shares_flow.cumsum() + initial_shares) * price
    return sw


def compute_yearly_summary(df_combined: pd.DataFrame) -> pd.DataFrame:
    if df_combined.empty:
        return pd.DataFrame()
    df = df_combined
    total_ret = (df.iloc[-1] / df.iloc[0]) - 1
    yearly = df.resample("YE").last()
    yearly_ret = yearly.pct_change(fill_method=None)
    yearly_ret.iloc[0] = (yearly.iloc[0] / df.iloc[0]) - 1
    summary = yearly_ret.T
    summary.columns = [c.year for c in summary.columns]
    summary["Total Acum."] = total_ret
    return summary
