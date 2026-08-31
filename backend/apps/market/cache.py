"""Cache de séries de mercado — envelope pandas sobre django.core.cache.

Porta do envelope de serialização de src/data/cache.py (mesmo formato JSON), mas
backed pelo django-redis configurado na Fase 1. O cache é uma OTIMIZAÇÃO: qualquer
falha é logada e engolida, nunca propaga.
"""
from __future__ import annotations

import io
import json
import logging

import pandas as pd
from django.core.cache import cache

logger = logging.getLogger(__name__)

TTL_SERIES = 86400  # 24h — séries derivadas
TTL_DLP = 600       # 10min — dados por-usuário da DLP

_SERIES = "pd.Series"
_FRAME = "pd.DataFrame"


def make_key(*parts) -> str:
    return "market:" + ":".join(str(p) for p in parts)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else str(col)
            for col in df.columns.to_flat_index()
        ]
    return df


def set_df(key: str, value, ttl: int | None) -> None:
    try:
        payload = value
        if isinstance(value, (pd.Series, pd.DataFrame)):
            frame = value.to_frame() if isinstance(value, pd.Series) else value
            frame = _flatten_columns(frame)
            payload = {
                "_type": _SERIES if isinstance(value, pd.Series) else _FRAME,
                "payload": json.loads(frame.to_json(orient="table")),
            }
        cache.set(key, payload, timeout=ttl)
    except Exception as exc:  # cache é otimização — não propaga
        logger.warning("cache set falhou para %s: %s", key, exc)


def get_df(key: str):
    try:
        raw = cache.get(key)
    except Exception as exc:
        logger.warning("cache get falhou para %s: %s", key, exc)
        return None
    if raw is None:
        return None
    if isinstance(raw, dict) and raw.get("_type") in (_SERIES, _FRAME):
        frame = pd.read_json(io.StringIO(json.dumps(raw["payload"])), orient="table")
        if raw["_type"] == _SERIES:
            return frame.iloc[:, 0]
        return frame
    return raw
