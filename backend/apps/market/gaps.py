"""Detecção de lacunas em séries temporais de mercado e inferência de cadência.
Módulo puro Python / pandas — zero I/O, zero Django.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

TOLERANCE_DAYS: dict[str, int] = {
    "daily": 10,
    "monthly": 45,
}

# Tolerância da cauda (último ponto até `until`). A série mensal é datada no 1º dia do
# mês de referência e publicada ~10 dias depois do fim dele (IPCA: ~dia 10 do mês
# seguinte), então o último ponto chega a ter ~70 dias sem nada estar parado.
TAIL_TOLERANCE_DAYS: dict[str, int] = {
    "daily": 10,
    "monthly": 75,
}


@dataclass(frozen=True)
class Gap:
    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class SeriesGap:
    series: str
    start: dt.date
    end: dt.date


def _to_date(val: Any) -> dt.date:
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    return pd.to_datetime(val).date()


def _clean_dates(dates: Iterable) -> pd.DatetimeIndex:
    if not isinstance(dates, (pd.Index, pd.Series)):
        dates = list(dates)
    if len(dates) == 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()


def find_gaps(dates: Iterable, periodicity: str) -> list[Gap]:
    """Ordena, deduplica e devolve os pares consecutivos com diferença > TOLERANCE_DAYS[periodicity].
    Aceita dt.date, pd.Timestamp ou DatetimeIndex.
    """
    tolerance = TOLERANCE_DAYS[periodicity]
    idx = _clean_dates(dates)
    if len(idx) < 2:
        return []

    diffs = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    gap_indices = np.where(diffs > tolerance)[0]
    return [Gap(start=idx[i].date(), end=idx[i + 1].date()) for i in gap_indices]


def stale_tail(dates: Iterable, periodicity: str, until: dt.date) -> Gap | None:
    """Devolve Gap(last, until) se (until - last).days > TAIL_TOLERANCE_DAYS[periodicity]; senão None.
    Série vazia -> None.
    """
    idx = _clean_dates(dates)
    if len(idx) == 0:
        return None
    last = idx[-1].date()
    until_d = _to_date(until)
    tolerance = TAIL_TOLERANCE_DAYS[periodicity]
    if (until_d - last).days > tolerance:
        return Gap(start=last, end=until_d)
    return None


def infer_periodicity(index: pd.Index) -> str:
    """Infere se o índice temporal é 'monthly' (mediana de passos >= 20 dias) ou 'daily'."""
    if len(index) < 2:
        return "daily"
    s = pd.Series(index)
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s)
    gap_days = s.sort_values().diff().dropna().dt.days.median()
    if pd.isna(gap_days):
        return "daily"
    if gap_days >= 20:
        return "monthly"
    return "daily"
