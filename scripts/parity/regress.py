"""Regressão numérica contra a linha de base e gravação de nova linha de base.

Compara série a série com tolerância 0 as saídas do engine novo contra a linha
de base gravada e reporta as colunas divergentes por métrica para cada caso.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
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

from parity.baseline import Baseline, load_baseline, save_baseline
from parity.cases import BENCHMARKS, Case, build_windows
from parity.compare import OK, compare_frames
from parity.engine_side import METRICS, run_engine


def changed_columns(
    old: dict[str, pd.DataFrame],
    new: dict[str, pd.DataFrame],
) -> dict[str, list[str]]:
    """Compara métricas entre old e new com tolerância 0.

    Retorna apenas as métricas que apresentaram colunas alteradas.
    """
    out: dict[str, list[str]] = {}
    seen = set()
    metrics: list[str] = []
    for m in METRICS:
        if m in old or m in new:
            metrics.append(m)
            seen.add(m)
    for m in (*old.keys(), *new.keys()):
        if m not in seen:
            metrics.append(m)
            seen.add(m)

    for metric in metrics:
        old_df = old.get(metric)
        new_df = new.get(metric)
        if old_df is None and new_df is None:
            continue
        diffs = compare_frames(
            metric, old_df, new_df, rel_tol=0.0, fail_status="CHANGED", abs_floor=0.0
        )
        changed = [d.column for d in diffs if d.status != OK]
        if changed:
            out[metric] = changed
    return out


def build_regression_grid(
    cases: list[Case],
    old_outputs: dict[str, dict[str, pd.DataFrame]],
    new_outputs: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Monta a grade caso × métrica → colunas alteradas / = / novo."""
    rows = []
    case_ids = [c.id for c in cases]
    for case in cases:
        row = {}
        old_case = old_outputs.get(case.id)
        new_case = new_outputs.get(case.id)
        if old_case is None:
            for m in METRICS:
                row[m] = "novo"
        else:
            diffs = changed_columns(old_case, new_case or {})
            for m in METRICS:
                if m in diffs:
                    row[m] = ", ".join(diffs[m])
                else:
                    row[m] = "="
        rows.append(row)
    return pd.DataFrame(rows, index=case_ids, columns=list(METRICS))


def format_markdown_table(grid: pd.DataFrame) -> str:
    """Formata o DataFrame da regressão em tabela Markdown alinhada."""
    headers = ["caso"] + list(grid.columns)
    rows = []
    for idx, row in grid.iterrows():
        rows.append([str(idx)] + [str(v) for v in row])

    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    data_lines = [
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(r)) + " |"
        for r in rows
    ]
    return "\n".join([header_line, sep_line, *data_lines])


def add_missing_windows(
    cases: list[Case], hists: dict[str, pd.DataFrame], today: dt.date
) -> tuple[list[Case], dict[str, pd.DataFrame]]:
    """Inclui as janelas de build_windows ausentes na linha de base, com o hist de w1-completo."""
    w1_hist = hists.get("w1-completo")
    if w1_hist is None:
        raise ValueError("Baseline não possui hist para w1-completo")
    cases, hists = list(cases), dict(hists)
    known = {c.id for c in cases}
    for win in build_windows(w1_hist["date"], today=today):
        if win.id in known:
            continue
        # Insere antes do caso com classe, mantendo ordem padrão
        idx = next((i for i, c in enumerate(cases) if c.classe is not None), len(cases))
        cases.insert(idx, Case(win.id, win, classe=None, benchmarks=BENCHMARKS))
        hists[win.id] = w1_hist
    return cases, hists


def run_regression(
    baseline_path: Path,
    save_root: Path | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    baseline = load_baseline(baseline_path)
    cases = list(baseline.cases)
    hists = dict(baseline.hists)

    cases, hists = add_missing_windows(cases, hists, dt.date.today())

    # Executa o engine para cada caso
    new_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for case in cases:
        out = run_engine(case, baseline.snapshot, hists[case.id])
        if out is None:
            raise RuntimeError(f"run_engine retornou None para o caso {case.id}")
        new_outputs[case.id] = out

    grid = build_regression_grid(cases, baseline.outputs, new_outputs)

    saved_path = None
    if save_root is not None:
        saved_path = save_baseline(
            root=save_root,
            cases=cases,
            snapshot=baseline.snapshot,
            hists=hists,
            outputs=new_outputs,
        )

    return grid, saved_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regressão contra a linha de base.")
    parser.add_argument("--baseline", required=True, type=Path, help="Diretório da linha de base")
    parser.add_argument("--save", type=Path, default=None, help="Raiz onde salvar a nova linha de base")
    args = parser.parse_args(argv)

    grid, saved_path = run_regression(args.baseline, args.save)
    print(format_markdown_table(grid))

    if saved_path is not None:
        print(f"\nNova linha de base salva em: {saved_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

