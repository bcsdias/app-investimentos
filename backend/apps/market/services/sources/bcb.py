"""Fonte BCB (SGS). Valor CRU: taxa % para 11/12/433; número-índice para 12466...

Porta de src/data/sources/market_data.py::buscar_dados_bcb — SEM cumprod, SEM
proxy splice, SEM cache. Só busca a janela pedida (com chunking de 5 anos porque
o SGS limita o range).
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from bcb import sgs

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_CHUNK_YEARS = 5


def fetch_sgs(codigo: int, start: dt.date, end: dt.date) -> list[RawPoint]:
    frames: list[pd.DataFrame] = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    try:
        while cur <= stop:
            chunk_end = min(cur + pd.DateOffset(years=_CHUNK_YEARS), stop)
            df = sgs.get({str(codigo): codigo}, start=cur.date(), end=chunk_end.date())
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                frames.append(df)
            cur = chunk_end + pd.DateOffset(days=1)
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"BCB SGS {codigo}: {exc}") from exc

    if not frames:
        return []
    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="last")].sort_index()
    col = str(codigo)
    series = pd.to_numeric(full[col], errors="coerce").dropna()
    return [RawPoint(ts.date(), float(v)) for ts, v in series.items()]
