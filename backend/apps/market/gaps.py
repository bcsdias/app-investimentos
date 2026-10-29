"""Detecção de lacunas em séries temporais de mercado e inferência de cadência.
Módulo puro Python / pandas — zero I/O, zero Django.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

TOLERANCE_DAYS: dict[str, int] = {
    "daily": 10,
    "monthly": 45,
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


def find_gaps(dates: Iterable, periodicity: str) -> list[Gap]:
    """Ordena, deduplica e devolve os pares consecutivos com diferença > TOLERANCE_DAYS[periodicity].
    Aceita dt.date, pd.Timestamp ou DatetimeIndex.
    """
    tolerance = TOLERANCE_DAYS[periodicity]
    clean_dates = sorted({_to_date(d) for d in dates})
    if len(clean_dates) < 2:
        return []

    gaps: list[Gap] = []
    for d1, d2 in zip(clean_dates[:-1], clean_dates[1:]):
        if (d2 - d1).days > tolerance:
            gaps.append(Gap(start=d1, end=d2))
    return gaps


def stale_tail(dates: Iterable, periodicity: str, until: dt.date) -> Gap | None:
    """Devolve Gap(last, until) se (until - last).days > TOLERANCE_DAYS[periodicity]; senão None.
    Série vazia -> None.
    """
    clean_dates = sorted({_to_date(d) for d in dates})
    if not clean_dates:
        return None
    last = clean_dates[-1]
    until_d = _to_date(until)
    tolerance = TOLERANCE_DAYS[periodicity]
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
