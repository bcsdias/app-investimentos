"""Linha de base da Fase 7: entradas congeladas + saídas do engine novo.

Grava/recarrega só com backend/ (sem src) e sem pyarrow, em pickle do pandas.
Contém dados da carteira: fica em reports/parity/ (ignorado pelo git).
Pickle é seguro aqui só porque load_baseline lê arquivos que o próprio harness
gravou localmente; nunca apontar load_baseline para um diretório de terceiros.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from parity.cases import Case, Window
from parity.frozen import MarketSnapshot

VERSION = 1


@dataclass
class Baseline:
    cases: list[Case]
    snapshot: MarketSnapshot
    hists: dict[str, pd.DataFrame]
    outputs: dict[str, dict[str, pd.DataFrame]]


def _case_to_json(case: Case) -> dict:
    return {
        "id": case.id,
        "window": case.window.id,
        "start": case.window.start.isoformat(),
        "end": case.window.end.isoformat(),
        "classe": case.classe,
        "benchmarks": list(case.benchmarks),
    }


def _case_from_json(d: dict) -> Case:
    window = Window(d["window"], dt.date.fromisoformat(d["start"]), dt.date.fromisoformat(d["end"]))
    return Case(d["id"], window, classe=d["classe"], benchmarks=tuple(d["benchmarks"]))


def save_baseline(
    root: Path,
    cases: list[Case],
    snapshot: MarketSnapshot,
    hists: dict[str, pd.DataFrame],
    outputs: dict[str, dict[str, pd.DataFrame]],
) -> Path:
    base = Path(root) / "baseline"
    (base / "market").mkdir(parents=True, exist_ok=True)
    (base / "hist").mkdir(exist_ok=True)

    market_files = {}
    for key, s in snapshot.series.items():
        name = f"{quote(key, safe='')}.pkl"
        s.to_pickle(base / "market" / name)
        market_files[key] = name

    for cid, h in hists.items():
        h.to_pickle(base / "hist" / f"{cid}.pkl")

    for cid, metrics in outputs.items():
        if metrics is None:
            continue
        out_dir = base / "outputs" / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        for metric, df in metrics.items():
            df.to_pickle(out_dir / f"{metric}.pkl")

    manifest = {
        "version": VERSION,
        "cases": [_case_to_json(c) for c in cases],
        "market": market_files,
    }
    (base / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return base


def load_baseline(path: Path) -> Baseline:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("version") != VERSION:
        raise ValueError(f"versão de baseline não suportada: {manifest.get('version')!r}")
    cases = [_case_from_json(d) for d in manifest["cases"]]
    snapshot = MarketSnapshot(
        series={k: pd.read_pickle(path / "market" / f) for k, f in manifest["market"].items()}
    )
    hists = {c.id: pd.read_pickle(path / "hist" / f"{c.id}.pkl") for c in cases}
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for c in cases:
        out_dir = path / "outputs" / c.id
        if out_dir.is_dir():
            outputs[c.id] = {p.stem: pd.read_pickle(p) for p in sorted(out_dir.glob("*.pkl"))}
    return Baseline(cases=cases, snapshot=snapshot, hists=hists, outputs=outputs)
