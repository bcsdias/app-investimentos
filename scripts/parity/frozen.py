"""Mercado congelado em memória e isolamento do lado novo em relação ao LAB.

O snapshot é lido do Postgres UMA vez e alimenta os dois lados da camada 1, de
modo que a ingestão rodando em paralelo (cron do lab-appinvest-ingestion) não
muda os números no meio da execução. Sem src: serve à Fase 7.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import warnings
from dataclasses import dataclass, field
from typing import Iterable
from unittest import mock

import pandas as pd
from django.core.cache.backends.base import CacheKeyWarning
from django.test import override_settings

SNAPSHOT_KEYS: tuple[str, ...] = (
    "bcb:11",
    "bcb:12",
    "bcb:433",
    "yf:^GSPC",
    "ptax:USDBRL",
    "b3:IFIX",
)

_DUMMY_CACHES = {"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}}


def _empty(series_key: str) -> pd.Series:
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]), name=series_key)


@dataclass
class MarketSnapshot:
    series: dict[str, pd.Series] = field(default_factory=dict)

    @classmethod
    def from_repository(cls, end: dt.date, keys: Iterable[str] = SNAPSHOT_KEYS) -> MarketSnapshot:
        from apps.market import catalog, repository

        return cls(series={k: repository.load_series(k, catalog.EPOCH, end) for k in keys})

    def load(self, series_key: str, start: dt.date, end: dt.date) -> pd.Series:
        """Mesmo contrato de apps.market.repository.load_series."""
        if series_key not in self.series:
            raise KeyError(f"série fora do snapshot: {series_key!r}")
        s = self.series[series_key]
        if s.empty:
            return _empty(series_key)
        s = s.sort_index()
        mask = (s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))
        out = s[mask].astype(float).copy()
        if out.empty:
            return _empty(series_key)
        out.name = series_key
        return out

    def missing(self) -> list[str]:
        return [k for k, s in self.series.items() if s is None or s.empty]


def _forbidden_write(*args, **kwargs):
    raise RuntimeError("parity: escrita no Postgres proibida")


@contextlib.contextmanager
def engine_patches(snapshot: MarketSnapshot, hist: pd.DataFrame):
    """Lado novo lendo só do snapshot e do histórico congelado, sem Redis nem escrita."""
    with contextlib.ExitStack() as stack:
        # DummyCache valida chaves para memcached; espaços em 'IPCA + 6%' são inofensivos aqui.
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore", CacheKeyWarning)
        stack.enter_context(override_settings(CACHES=_DUMMY_CACHES))
        stack.enter_context(mock.patch("apps.market.repository.load_series", snapshot.load))
        stack.enter_context(mock.patch("apps.market.repository.upsert_points", _forbidden_write))
        stack.enter_context(mock.patch("apps.market.series._ensure_fresh", lambda *a, **k: None))
        stack.enter_context(
            mock.patch(
                "apps.market.services.sources.dlp.buscar_historico",
                lambda *a, **k: hist.copy(deep=True),
            )
        )
        yield
