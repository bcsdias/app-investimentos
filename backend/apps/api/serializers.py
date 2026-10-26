"""Serialização pura de Series / DataFrames do engine para JSON (formato ECharts)."""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pandas as pd


def clean(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _date_iso(idx: Any) -> str:
    if isinstance(idx, pd.Timestamp):
        return idx.date().isoformat()
    if isinstance(idx, dt.datetime):
        return idx.date().isoformat()
    if isinstance(idx, dt.date):
        return idx.isoformat()
    return str(idx)[:10]


def series_to_json(s: pd.Series, name: str | None = None) -> dict:
    resolved_name = name if name is not None else str(s.name or "")
    data = [[_date_iso(idx), clean(val)] for idx, val in s.items()]
    return {"name": resolved_name, "data": data}


def frame_to_series_list(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return [series_to_json(df[col], name=str(col)) for col in df.columns]


def frame_to_rows(df: pd.DataFrame, columns: dict[str, str]) -> list[dict]:
    if df is None or df.empty:
        return []
    rows: list[dict] = []
    for idx, row in df.iterrows():
        item: dict[str, Any] = {"name": str(idx)}
        for orig_col, target_col in columns.items():
            item[target_col] = clean(row.get(orig_col))
        rows.append(item)
    return rows


def yearly_to_json(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"years": [], "rows": []}
    years = [int(c) for c in df.columns if c != "Total Acum."]
    rows: list[dict] = []
    for idx, row in df.iterrows():
        values = {str(y): clean(row.get(y)) for y in years}
        rows.append(
            {
                "name": str(idx),
                "values": values,
                "total": clean(row.get("Total Acum.")),
            }
        )
    return {"years": years, "rows": rows}
