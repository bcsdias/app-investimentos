"""Medidor de desempenho do engine sobre a linha de base congelada."""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# Garante backend e scripts no sys.path e Django configurado para execução CLI
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django

try:
    django.setup()
except RuntimeError:
    pass

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
from engine.dataset import build_dataset, fetch_user_portfolio
from parity.baseline import load_baseline
from parity.cases import Case
from parity.engine_side import ROLLING_WINDOW, normalize_yearly, portfolio_frame
from parity.frozen import MarketSnapshot, engine_patches
from parity.kpis import derive_kpis


def _run_single(case: Case) -> tuple[float, float, float, float]:
    start, end = case.window.start.isoformat(), case.window.end.isoformat()

    t_all_start = time.perf_counter()

    t_fetch_start = time.perf_counter()
    user_series, portfolio_df = fetch_user_portfolio(
        "parity", classe=case.classe, start_date=start, end_date=end
    )
    t_fetch = (time.perf_counter() - t_fetch_start) * 1000.0

    if user_series is None or portfolio_df is None or user_series.empty or portfolio_df.empty:
        t_all = (time.perf_counter() - t_all_start) * 1000.0
        return t_fetch, 0.0, 0.0, t_all

    t_build_start = time.perf_counter()
    res = build_dataset(
        user_series,
        benchmarks=list(case.benchmarks),
        extra_specs=[],
        start_date=start,
        end_date=end,
        portfolio_df=portfolio_df,
    )
    t_build = (time.perf_counter() - t_build_start) * 1000.0

    df = res.df_combined
    pdf = res.portfolio_df
    twr = compute_twr_evolution(df)["series"]
    drawdown = compute_drawdown(df)
    rolling = compute_rolling_metrics(df, res.rf_series, window=ROLLING_WINDOW)
    risk_return = compute_risk_return(df, res.rf_series)

    t_irr_start = time.perf_counter()
    irr = compute_irr_evolution(pdf)
    t_irr = (time.perf_counter() - t_irr_start) * 1000.0

    shadow = compute_shadow(pdf, df)
    yearly = normalize_yearly(compute_yearly_summary(df))
    kpis = derive_kpis(pdf, twr, risk_return, drawdown, irr)
    portfolio = portfolio_frame(pdf)

    t_all = (time.perf_counter() - t_all_start) * 1000.0
    return t_fetch, t_build, t_irr, t_all


def time_case(
    case: Case,
    snapshot: MarketSnapshot,
    hist: pd.DataFrame,
    repeat: int = 5,
) -> dict[str, float]:
    """Mede a mediana em ms das etapas do engine para um caso congelado.

    Uma execução de aquecimento antes é descartada.
    """
    with engine_patches(snapshot, hist):
        # Aquecimento
        _run_single(case)

        fetches: list[float] = []
        builds: list[float] = []
        irrs: list[float] = []
        totals: list[float] = []

        for _ in range(repeat):
            f, b, i, t = _run_single(case)
            fetches.append(f)
            builds.append(b)
            irrs.append(i)
            totals.append(t)

    return {
        "fetch_user_portfolio": statistics.median(fetches),
        "build_dataset": statistics.median(builds),
        "compute_irr_evolution": statistics.median(irrs),
        "total": statistics.median(totals),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Medidor de desempenho do engine.")
    parser.add_argument("--baseline", required=True, type=Path, help="Diretório da linha de base")
    parser.add_argument("--repeat", type=int, default=5, help="Número de repetições por caso")
    args = parser.parse_args(argv)

    baseline = load_baseline(args.baseline)

    for case in baseline.cases:
        hist = baseline.hists[case.id]
        res = time_case(case, baseline.snapshot, hist, repeat=args.repeat)
        print(
            f"{case.id:<25} "
            f"total={res['total']:>6.1f}ms  "
            f"fetch={res['fetch_user_portfolio']:>6.1f}ms  "
            f"build={res['build_dataset']:>6.1f}ms  "
            f"irr={res['compute_irr_evolution']:>6.1f}ms"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
