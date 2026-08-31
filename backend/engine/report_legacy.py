"""Export CLI dos gráficos históricos (Matplotlib). NÃO é importado pela API.

Cada plot_* chama a compute_* correspondente e desenha. Preservado para gerar os
PNGs antigos durante a coexistência e como conferência visual na paridade (Fase 6).
`import matplotlib` vive SÓ aqui.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from engine import compute
from engine.dataset import DatasetResult


def plot_twr_evolution(df_combined, title_suffix: str = ""):
    out = compute.compute_twr_evolution(df_combined)
    df = out["series"]
    fig, ax = plt.subplots(figsize=(12, 7))
    for col in df.columns:
        lw = 3 if col == "Carteira" else 1.5
        ax.plot(df.index, df[col], label=f"{col} ({out['final_returns'][col]:.1f}%)", linewidth=lw)
    ax.set_title(f"Evolução TWR (Base 100) - {title_suffix}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig


def plot_drawdown(df_combined, title_suffix: str = ""):
    dd = compute.compute_drawdown(df_combined)
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in dd.columns:
        ax.plot(dd.index, dd[col], label=col)
    ax.set_title(f"Drawdown - {title_suffix}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig


def plot_risk_return(df_combined, title_suffix: str = ""):
    rr = compute.compute_risk_return(df_combined)
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, row in rr.iterrows():
        ax.scatter(row["Volatilidade"], row["Retorno (CAGR)"], s=120 if name == "Carteira" else 70)
        ax.text(row["Volatilidade"], row["Retorno (CAGR)"], f"  {name}", fontsize=9, va="center")
    ax.set_title(f"Risco x Retorno - {title_suffix}")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def plot_rolling_sharpe(df_combined, rf_series=None, window: int = 252, title_suffix: str = ""):
    sharpe = compute.compute_rolling_metrics(df_combined, rf_series, window)["sharpe"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in sharpe.columns:
        ax.plot(sharpe.index, sharpe[col], label=col)
    ax.set_title(f"Sharpe Ratio Móvel ({window}d) - {title_suffix}")
    ax.grid(True)
    ax.legend()
    return fig


def plot_shadow(portfolio_df, df_combined, title_suffix: str = ""):
    sw = compute.compute_shadow(portfolio_df, df_combined)
    fig, ax = plt.subplots(figsize=(12, 7))
    for col in sw.columns:
        ax.plot(sw.index, sw[col], label=col, linewidth=3 if col == "Carteira Real" else 1.5)
    ax.set_title(f"Simulação de Aportes - {title_suffix}")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig


def export_all(result: DatasetResult, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    jobs = [
        ("twr_evolucao.png", plot_twr_evolution(result.df_combined)),
        ("drawdown.png", plot_drawdown(result.df_combined)),
        ("risco_retorno.png", plot_risk_return(result.df_combined)),
        ("sharpe_movel.png", plot_rolling_sharpe(result.df_combined, result.selic_series)),
    ]
    if result.portfolio_df is not None and not result.portfolio_df.empty:
        jobs.append(("simulacao_aportes.png", plot_shadow(result.portfolio_df, result.df_combined)))

    paths: list[str] = []
    for name, fig in jobs:
        path = os.path.join(out_dir, name)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
