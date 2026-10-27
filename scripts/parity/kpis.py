"""KPIs do /summary derivados das saídas do engine.

Mesma derivação de apps/api/routers/portfolio.py::summary; ausências viram NaN
(o router devolve null).
"""
from __future__ import annotations

import math

import pandas as pd

KPI_NAMES: tuple[str, ...] = (
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
)


def derive_kpis(
    portfolio_df: pd.DataFrame,
    twr: pd.DataFrame,
    risk_return: pd.DataFrame,
    drawdown: pd.DataFrame,
    irr: pd.Series,
) -> pd.DataFrame:
    nan = math.nan
    pat_ini = float(portfolio_df["vlr_mercado"].iloc[0])
    pat_fim = float(portfolio_df["vlr_mercado"].iloc[-1])
    aportes = float(portfolio_df["vlr_investido"].iloc[-1] - portfolio_df["vlr_investido"].iloc[0])

    twr_total = float(twr["Carteira"].iloc[-1] - 100) if "Carteira" in twr.columns and len(twr) else nan
    if "Carteira" in risk_return.index:
        row = risk_return.loc["Carteira"]
        cagr, vol, sharpe = (float(row["Retorno (CAGR)"]), float(row["Volatilidade"]), float(row["Sharpe"]))
    else:
        cagr = vol = sharpe = nan
    max_dd = float(drawdown["Carteira"].min()) if "Carteira" in drawdown.columns else nan
    xirr = float(irr.iloc[-1]) if irr is not None and not irr.empty else nan

    values = [pat_ini, aportes, pat_fim, pat_fim - pat_ini - aportes, twr_total, cagr, vol, sharpe, max_dd, xirr]
    return pd.DataFrame({"valor": values}, index=list(KPI_NAMES))
