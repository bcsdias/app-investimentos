"""Único ponto de acesso a core.MarketSeries. Postgres <-> pandas, sem derivação."""
from __future__ import annotations

import datetime as dt
import logging
import math
from decimal import Decimal
from typing import Iterable

import pandas as pd
from django.db.models import Max

from apps.core.models import MarketSeries
from apps.market import cache

logger = logging.getLogger(__name__)

_Q = Decimal("1e-8")


def upsert_points(series_key: str, source: str, kind: str, points: Iterable) -> int:
    rows: list[MarketSeries] = []
    dropped = 0
    for p in points:
        v = float(p.value)
        if math.isnan(v) or math.isinf(v):
            dropped += 1
            continue
        rows.append(
            MarketSeries(
                series_key=series_key,
                source=source,
                kind=kind,
                reference_date=p.date,
                value=Decimal(str(v)).quantize(_Q),
            )
        )
    if dropped:
        logger.warning("%s: %d ponto(s) descartado(s) (NaN/inf)", series_key, dropped)

    # bulk_create(update_conflicts=True) levanta "cannot affect row a second time"
    # se o mesmo reference_date vier repetido no lote — dedup mantendo o último.
    by_date = {row.reference_date: row for row in rows}
    if len(by_date) != len(rows):
        logger.warning(
            "%s: %d data(s) repetida(s) no lote — mantido o último ponto de cada",
            series_key,
            len(rows) - len(by_date),
        )
    rows = list(by_date.values())

    if not rows:
        return 0
    MarketSeries.objects.bulk_create(
        rows,
        update_conflicts=True,
        unique_fields=["series_key", "reference_date"],
        update_fields=["value", "kind", "source", "updated_at"],
    )
    cache.bump_series_generation()
    return len(rows)


def load_series(series_key: str, start: dt.date, end: dt.date) -> pd.Series:
    qs = (
        MarketSeries.objects.filter(
            series_key=series_key, reference_date__range=(start, end)
        )
        .order_by("reference_date")
        .values_list("reference_date", "value")
    )
    pairs = list(qs)
    if not pairs:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]), name=series_key)
    idx = pd.to_datetime([d for d, _ in pairs])
    vals = [float(v) for _, v in pairs]
    return pd.Series(vals, index=idx, name=series_key, dtype=float)


def last_date(series_key: str) -> dt.date | None:
    return (
        MarketSeries.objects.filter(series_key=series_key)
        .aggregate(m=Max("reference_date"))
        .get("m")
    )


def available_keys() -> list[str]:
    return sorted(
        MarketSeries.objects.order_by().values_list("series_key", flat=True).distinct()
    )
