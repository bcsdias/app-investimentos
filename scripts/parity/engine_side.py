"""Lado novo da camada 1: engine.dataset + engine.compute sobre o mercado congelado.

Reproduz apps.api.analysis.run_analysis sem HTTP, sem token e sem LAB. Sem src:
é o que a Fase 7 roda contra a linha de base.
"""
from __future__ import annotations

import pandas as pd

from engine.compute import (
    compute_drawdown,
    compute_irr_evolution,
    compute_risk_return,
    compute_rolling_metrics,
    compute_shadow,
    compute_twr_evolution,
    compute_yearly_summary,
)
from engine.dataset import DatasetResult, build_dataset, fetch_user_portfolio
from parity.cases import Case
from parity.frozen import MarketSnapshot, engine_patches
from parity.kpis import derive_kpis

METRICS: tuple[str, ...] = (
    "twr",
    "drawdown",
    "rolling_vol",
    "rolling_sharpe",
    "risk_return",
    "irr",
    "shadow",
    "yearly",
    "kpis",
    "portfolio",
)

PORTFOLIO_COLUMNS = ["vlr_mercado", "vlr_investido", "proventos", "fluxo"]
ROLLING_WINDOW = 252


def normalize_yearly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out


def irr_frame(irr: pd.Series | None) -> pd.DataFrame:
    if irr is None or irr.empty:
        return pd.DataFrame(columns=["TIR"], dtype=float)
    return irr.rename("TIR").to_frame()


def portfolio_frame(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    return portfolio_df.set_index("date")[PORTFOLIO_COLUMNS]


def engine_outputs(res: DatasetResult) -> dict[str, pd.DataFrame]:
    df = res.df_combined
    pdf = res.portfolio_df
    twr = compute_twr_evolution(df)["series"]
    drawdown = compute_drawdown(df)
    rolling = compute_rolling_metrics(df, res.rf_series, window=ROLLING_WINDOW)
    risk_return = compute_risk_return(df, res.rf_series)
    irr = compute_irr_evolution(pdf)
    return {
        "twr": twr,
        "drawdown": drawdown,
        "rolling_vol": rolling["volatility"],
        "rolling_sharpe": rolling["sharpe"],
        "risk_return": risk_return,
        "irr": irr_frame(irr),
        "shadow": compute_shadow(pdf, df),
        "yearly": normalize_yearly(compute_yearly_summary(df)),
        "kpis": derive_kpis(pdf, twr, risk_return, drawdown, irr),
        "portfolio": portfolio_frame(pdf),
    }


def run_engine(case: Case, snapshot: MarketSnapshot, hist: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    start, end = case.window.start.isoformat(), case.window.end.isoformat()
    with engine_patches(snapshot, hist):
        user_series, portfolio_df = fetch_user_portfolio(
            "parity", classe=case.classe, start_date=start, end_date=end
        )
        if user_series is None or portfolio_df is None or user_series.empty or portfolio_df.empty:
            return None
        res = build_dataset(
            user_series,
            benchmarks=list(case.benchmarks),
            extra_specs=[],
            start_date=start,
            end_date=end,
            portfolio_df=portfolio_df,
        )
        return engine_outputs(res)
