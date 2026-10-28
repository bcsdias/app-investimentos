"""Matriz de casos da paridade: janelas, filtro de classe e benchmarks fixos."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Um benchmark por tipo de origem/transformação presente nos dois catálogos:
# BCB (taxa acumulada), sintético de taxa, Yahoo convertido por PTAX, B3.
BENCHMARKS: tuple[str, ...] = ("CDI", "IPCA + 6%", "S&P 500 BRL", "IFIX")

NOT_COVERED: dict[str, str] = {
    "Tesouro": "CATALOGO_TD do legado vazio e catálogo novo sem TD",
    "Carteiras compostas": "perda aceita (decisão 4 da spec)",
}

# Mesmas colunas, na mesma ordem, que CLASS_COLUMNS de apps/api/routers/portfolio.py.
_CLASS_COLUMNS = ("classe", "tipo")


@dataclass(frozen=True)
class Window:
    id: str
    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class Case:
    id: str
    window: Window
    classe: str | None = None
    benchmarks: tuple[str, ...] = BENCHMARKS

    def api_payload(self) -> dict:
        return {
            "start": self.window.start.isoformat(),
            "end": self.window.end.isoformat(),
            "ativo": None,
            "classe": self.classe,
            "benchmarks": list(self.benchmarks),
            "synthetics": [],
        }


def build_windows(dates: Iterable, today: dt.date) -> list[Window]:
    uniq = sorted(pd.to_datetime(pd.Series(list(dates))).dt.normalize().unique())
    first, last = pd.Timestamp(uniq[0]).date(), pd.Timestamp(uniq[-1]).date()
    return [
        Window("w1-completo", first, last),
        Window("w2-12m", (pd.Timestamp(last) - pd.DateOffset(years=1)).date(), last),
        Window("w3-meio", pd.Timestamp(uniq[len(uniq) // 2]).date(), today),
        Window("w4-fim-passado", first, pd.Timestamp(uniq[len(uniq) * 3 // 4]).date()),
        # Dia seguinte a um fechamento: exercita a abertura no fechamento anterior a start.
        Window("w5-inicio-mes", (pd.Timestamp(uniq[len(uniq) // 4]) + pd.Timedelta(days=1)).date(), last),
    ]


def pick_classe(hist: pd.DataFrame) -> str | None:
    lower_map = {str(c).lower(): c for c in hist.columns}
    col = next((lower_map[c] for c in _CLASS_COLUMNS if c in lower_map), None)
    if col is None or "date" not in hist.columns:
        return None
    df = hist[["date", col]].dropna()
    df = df.assign(_classe=df[col].astype(str).str.strip())
    df = df[df["_classe"] != ""]
    if df.empty:
        return None
    counts = df.groupby("_classe")["date"].nunique()
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0]


def build_cases(hist: pd.DataFrame, today: dt.date) -> list[Case]:
    windows = build_windows(hist["date"], today)
    cases = [Case(w.id, w) for w in windows]
    classe = pick_classe(hist)
    if classe is not None:
        cases.append(Case(f"{windows[0].id}-classe", windows[0], classe=classe))
    return cases
