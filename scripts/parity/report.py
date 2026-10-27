"""Relatório de paridade: summary.md (sem R$) e CSVs de detalhe (locais, nunca versionados)."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from parity.cases import NOT_COVERED, Case
from parity.compare import ABS_FLOOR, DESVIO, FALHA, INVESTIGAR, OK, REL_TOL_L1, REL_TOL_L2, SeriesDiff

_LAYER_TITLE = {1: "Camada 1 — engine, mercado congelado", 2: "Camada 2 — ponta a ponta (API × legado ao vivo)"}


@dataclass
class CaseResult:
    case: Case
    layer: int
    diffs: list[SeriesDiff]
    notes: list[str]


def _rel(v: float) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3e}"


def _cell(text) -> str:
    return str(text if text not in (None, "") else "—").replace("|", "\\|")


def _case_rows(cases: list[Case]) -> list[str]:
    lines = ["| caso | início | fim | classe | benchmarks |", "|---|---|---|---|---|"]
    for c in cases:
        lines.append(
            f"| {c.id} | {c.window.start.isoformat()} | {c.window.end.isoformat()} | "
            f"{_cell(c.classe)} | {', '.join(c.benchmarks)} |"
        )
    return lines


def _result_block(r: CaseResult) -> list[str]:
    lines = [f"### {r.case.id} · {_LAYER_TITLE.get(r.layer, f'Camada {r.layer}')}", ""]
    for note in r.notes:
        lines.append(f"- Nota: {note}")
    if r.notes:
        lines.append("")

    ok_count = Counter(d.metric for d in r.diffs if d.status == OK)
    if ok_count:
        lines.append("OK (séries por métrica): " + ", ".join(f"{m}: {n}" for m, n in ok_count.items()))
        lines.append("")

    bad = [d for d in r.diffs if d.status != OK]
    if bad:
        lines += [
            "| métrica | coluna | status | max_rel | pontos fora | 1ª data divergente | origem | nota |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for d in bad:
            lines.append(
                f"| {d.metric} | {_cell(d.column)} | {d.status} | {_rel(d.max_rel)} | {d.n_out} | "
                f"{_cell(d.first_date)} | {_cell(d.origin)} | {_cell(d.note)} |"
            )
        lines.append("")
    elif not r.diffs and not r.notes:
        lines += ["Nada comparado.", ""]
    return lines


def write_summary(out_dir: Path, results: list[CaseResult], meta: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Relatório de paridade — Fase 6",
        "",
        f"- Execução: {meta.get('timestamp', '—')}",
        f"- Commit: {meta.get('commit', '—')}",
        f"- Camadas: {', '.join(str(x) for x in meta.get('layers', []))}",
        f"- Tolerâncias: camada 1 {REL_TOL_L1:g} relativo (piso absoluto {ABS_FLOOR:g}); "
        f"camada 2 {REL_TOL_L2:g} relativo (acima disso: {INVESTIGAR})",
        "- Sem valores em R$: só razões relativas, contagens e datas. "
        "Séries completas em `details/` (locais, nunca versionadas).",
        "",
        "## Casos",
        "",
        *_case_rows(meta.get("cases", [])),
        "",
        "## Fora da paridade",
        "",
        *(f"- {k}: {v}" for k, v in NOT_COVERED.items()),
        "",
        "## Séries ausentes no snapshot de mercado",
        "",
    ]
    missing = meta.get("missing") or []
    lines += [f"- {k}" for k in missing] if missing else ["- nenhuma"]
    lines += ["", "## Resultados", ""]
    for r in results:
        lines += _result_block(r)

    integration = meta.get("integration")
    lines += ["## Teste integration (Fase 3c)", "", f"- {integration or 'não executado'}", ""]

    totals = Counter(d.status for r in results for d in r.diffs)
    lines += [
        "## Totais",
        "",
        *(f"- {s}: {totals.get(s, 0)}" for s in (FALHA, DESVIO, INVESTIGAR, OK)),
        "",
        "## Aprovação",
        "",
        "- pendente",
        "",
    ]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_details(out_dir: Path, case_id: str, layer: int, legacy: dict, new: dict) -> None:
    base = Path(out_dir) / "details" / case_id / f"L{layer}"
    base.mkdir(parents=True, exist_ok=True)
    for side, outputs in (("legacy", legacy or {}), ("new", new or {})):
        for metric, df in outputs.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(base / f"{metric}.{side}.csv")


def exit_code(results: list[CaseResult]) -> int:
    failing = {FALHA, DESVIO}
    return int(any(r.layer == 1 and d.status in failing for r in results for d in r.diffs))
