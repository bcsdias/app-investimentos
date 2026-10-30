"""Comparação numérica série a série entre o lado legado e o lado novo.

Puro pandas/numpy — sem Django, sem src. Sobrevive à remoção do legado e é o
comparador que a Fase 7 usa contra a linha de base.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

REL_TOL_L1 = 1e-6
REL_TOL_L2 = 1e-4
ABS_FLOOR = 1e-9

FALHA = "FALHA"
INVESTIGAR = "INVESTIGAR"
OK = "OK"
DESVIO = "DESVIO CONHECIDO (ADR 0012)"

ADR0012_METRICS = frozenset({"rolling_vol", "rolling_sharpe", "risk_return", "yearly"})

_CALC_METRICS = frozenset({"irr", "kpis", "portfolio"})
_CALC_COLUMNS = frozenset({"Carteira", "Carteira Real", "Total Investido", "TIR"})


@dataclass(frozen=True)
class SeriesDiff:
    metric: str
    column: str
    status: str
    max_rel: float
    n_out: int
    first_date: str | None
    only_legacy: int
    only_new: int
    note: str = ""
    origin: str = ""


def _label(x) -> str:
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    return str(x)


def _as_float_series(col: pd.Series) -> pd.Series:
    s = pd.to_numeric(col, errors="coerce").astype(float)
    s.index = [_label(i) for i in col.index]
    return s


def _compare_column(
    metric: str,
    column: str,
    legacy: pd.Series,
    new: pd.Series,
    rel_tol: float,
    fail_status: str,
    abs_floor: float,
) -> SeriesDiff:
    a = _as_float_series(legacy)
    b = _as_float_series(new)
    only_l = sorted(set(a.index) - set(b.index))
    only_n = sorted(set(b.index) - set(a.index))
    common = [i for i in a.index if i in set(b.index)]

    av = a.loc[common].to_numpy()
    bv = b.loc[common].to_numpy()
    a_nan, b_nan = np.isnan(av), np.isnan(bv)
    both = ~a_nan & ~b_nan
    scale = np.maximum(np.abs(av), np.abs(bv))
    delta = np.abs(av - bv)
    with np.errstate(invalid="ignore"):
        out_val = both & (delta > np.maximum(rel_tol * scale, abs_floor))
    diverging = (a_nan ^ b_nan) | out_val

    rel_mask = both & (scale > 0)
    max_rel = float(np.max(delta[rel_mask] / scale[rel_mask])) if rel_mask.any() else 0.0

    bad_labels = [lbl for lbl, bad in zip(common, diverging) if bad] + only_l + only_n
    n_out = len(bad_labels)
    return SeriesDiff(
        metric=metric,
        column=column,
        status=OK if n_out == 0 else fail_status,
        max_rel=max_rel,
        n_out=n_out,
        first_date=min(bad_labels) if bad_labels else None,
        only_legacy=len(only_l),
        only_new=len(only_n),
    )


def compare_frames(
    metric: str,
    legacy: pd.DataFrame,
    new: pd.DataFrame,
    rel_tol: float,
    fail_status: str,
    abs_floor: float = ABS_FLOOR,
) -> list[SeriesDiff]:
    legacy = legacy if legacy is not None else pd.DataFrame()
    new = new if new is not None else pd.DataFrame()
    columns = list(dict.fromkeys([*map(str, legacy.columns), *map(str, new.columns)]))
    leg_cols = {str(c): c for c in legacy.columns}
    new_cols = {str(c): c for c in new.columns}

    diffs: list[SeriesDiff] = []
    for col in columns:
        in_l, in_n = col in leg_cols, col in new_cols
        if in_l and in_n:
            diffs.append(
                _compare_column(
                    metric, col, legacy[leg_cols[col]], new[new_cols[col]],
                    rel_tol, fail_status, abs_floor,
                )
            )
            continue
        present = legacy if in_l else new
        diffs.append(
            SeriesDiff(
                metric=metric,
                column=col,
                status=fail_status,
                max_rel=math.nan,
                n_out=len(present),
                first_date=None,
                only_legacy=0,
                only_new=0,
                note="só no legado" if in_l else "só no novo",
            )
        )
    return diffs


def classify_origin(diff: SeriesDiff) -> str:
    if diff.only_legacy or diff.only_new:
        return "calendário/datas"
    if diff.column.endswith(" BRL"):
        return "conversão de câmbio"
    if diff.metric in _CALC_METRICS or diff.column in _CALC_COLUMNS:
        return "cálculo"
    return "revisão de cotação na fonte"


def apply_adr0012_recheck(
    diffs: list[SeriesDiff],
    legacy_fill_none: dict[str, pd.DataFrame],
    new: dict[str, pd.DataFrame],
) -> list[SeriesDiff]:
    bad = [d for d in diffs if d.status != OK]
    if not bad or any(d.metric not in ADR0012_METRICS for d in bad):
        return diffs
    for metric in sorted({d.metric for d in bad}):
        if metric not in legacy_fill_none or metric not in new:
            return diffs
        recheck = compare_frames(metric, legacy_fill_none[metric], new[metric], REL_TOL_L1, FALHA)
        if any(r.status != OK for r in recheck):
            return diffs
    return [
        dataclasses.replace(
            d,
            status=DESVIO,
            note=f"some com fill_method=None; max_rel original {d.max_rel:.3e}",
        )
        if d.status != OK
        else d
        for d in diffs
    ]
