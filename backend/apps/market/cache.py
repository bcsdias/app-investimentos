"""Cache de séries de mercado — envelope pandas sobre django.core.cache.

Porta do envelope de serialização de src/data/cache.py (mesmo formato JSON), mas
backed pelo django-redis configurado na Fase 1. O cache é uma OTIMIZAÇÃO: qualquer
falha é logada e engolida, nunca propaga.
"""
from __future__ import annotations

import io
import json
import logging
import time
from contextlib import contextmanager

import pandas as pd
from django.core.cache import cache

logger = logging.getLogger(__name__)

TTL_SERIES = 86400  # 24h — séries derivadas
TTL_DLP = 600       # 10min — dados por-usuário da DLP

_SERIES = "pd.Series"
_FRAME = "pd.DataFrame"
# Rótulo para série sem nome: no orient="table" a coluna 0 (int) volta como "0" (str)
# e o read_json devolve tudo NaN — caso de to_brl/splice_proxy (achado da Fase 6).
_UNNAMED = "__unnamed__"


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
            if isinstance(value, pd.Series):
                frame = value.to_frame(name=_UNNAMED if value.name is None else value.name)
            else:
                frame = value
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
            out = frame.iloc[:, 0]
            if out.name == _UNNAMED:
                out.name = None
            return out
        return frame
    return raw


@contextmanager
def single_flight(key: str, timeout: float = 90, poll: float = 0.1):
    """Serializa quem calcula/busca o mesmo `key` (lock via cache.add, atômico).

    Quem não obtém o lock espera até obtê-lo ou estourar `timeout`; neste caso
    segue sem o lock (com warning) e não libera um lock que não é seu.
    """
    lock_key = key + ":lock"
    acquired = False
    deadline = time.monotonic() + timeout
    while True:
        try:
            acquired = bool(cache.add(lock_key, 1, timeout))
        except Exception as exc:  # cache é otimização — segue sem lock
            logger.warning("single_flight add falhou para %s: %s", lock_key, exc)
            break
        if acquired:
            break
        if time.monotonic() >= deadline:
            logger.warning("single_flight estourou %.1fs para %s; seguindo sem lock", timeout, key)
            break
        time.sleep(poll)
    try:
        yield
    finally:
        if acquired:
            try:
                cache.delete(lock_key)
            except Exception as exc:
                logger.warning("single_flight delete falhou para %s: %s", lock_key, exc)
