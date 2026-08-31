"""Fachada de leitura de séries de mercado.

get_series(name, start, end): Redis -> repository + transforms (conforme catálogo)
-> Redis. Substitui o processar_benchmarks de 80 linhas do legado. Recursão rasa
(Composite -> membros -> talvez ToBRL -> base + PTAX); sem DAG genérico.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from apps.market import cache, catalog, repository, transforms

logger = logging.getLogger(__name__)


def get_series(name: str, start: dt.date, end: dt.date) -> pd.Series | None:
    key = cache.make_key("series", name, start, end)
    hit = cache.get_df(key)
    if hit is not None:
        return hit
    sdef = catalog.resolve(name)
    series = _materialize(sdef, start, end)
    if series is None or series.empty:
        logger.info("série '%s' vazia no período %s..%s", name, start, end)
        return None
    cache.set_df(key, series, ttl=cache.TTL_SERIES)
    return series


def get_many(names: list[str], start: dt.date, end: dt.date) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for name in names:
        s = get_series(name, start, end)
        if s is not None and not s.empty:
            out[name] = s
        else:
            logger.info("get_many: '%s' ausente/vazia — fora do resultado", name)
    return out


def _materialize(sdef: catalog.SeriesDef, start: dt.date, end: dt.date) -> pd.Series | None:
    d = sdef.derivation

    if d is None:
        raw = repository.load_series(sdef.series_key, start, end)
        if raw.empty:
            return None
        if sdef.kind == "rate":
            return transforms.accumulate(raw, sdef.periodicity)
        return raw

    if isinstance(d, catalog.ProxySplice):
        bcb = repository.load_series(d.bcb_key, start, end)
        etf = repository.load_series(d.etf_key, start, end)
        if bcb.empty and etf.empty:
            return None
        return transforms.splice_proxy(bcb, etf)

    if isinstance(d, catalog.ToBRL):
        base = get_series(d.base_name, start, end)
        ptax = get_series("USD/BRL", start, end)
        if base is None or ptax is None:
            return None
        return transforms.to_brl(base, ptax)

    if isinstance(d, catalog.Synthetic):
        base = get_series(d.spec.base, start, end)
        if base is None:
            return None
        return transforms.synthetic_from_spec(d.spec, base)

    if isinstance(d, catalog.Composite):
        members = {m: get_series(m, start, end) for m in d.weights}
        if any(v is None for v in members.values()):
            return None
        return transforms.composite(members, d.weights)

    raise TypeError(f"derivação não suportada: {type(d).__name__}")
