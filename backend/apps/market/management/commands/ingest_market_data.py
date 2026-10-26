"""Ingestão AO VIVO de séries de mercado -> core.MarketSeries (fonte de verdade).

Percorre catalog.ingestible(); para cada série busca da última data gravada até
hoje (incremental), grava o valor CRU via repository e, ao fim, aquece o cache
dos DEFAULT_BENCHMARKS. Uma série falhar NÃO aborta o lote; exit 1 se houve falha.
Agendar este comando é da Fase 5 (ofelia/cron).
"""
from __future__ import annotations

import datetime as dt
import logging
import sys

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import DatabaseError

from apps.market import catalog, repository, series
from apps.market.services.sources import b3, bcb, ptax, tesouro, yahoo
from apps.market.services.sources.base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_EPOCH = catalog.EPOCH


def _fetch(sdef: catalog.SeriesDef, start: dt.date, end: dt.date) -> list[RawPoint]:
    prefix, _, ident = sdef.series_key.partition(":")
    if prefix == "bcb":
        return bcb.fetch_sgs(int(ident), start, end)
    if prefix == "yf":
        return yahoo.fetch_yahoo(ident, start, end)
    if prefix == "ptax":
        return ptax.fetch_ptax(start, end)
    if prefix == "td":
        titulo, _, venc = ident.partition(":")
        return tesouro.fetch_tesouro(titulo, dt.date.fromisoformat(venc), start, end)
    if prefix == "b3":
        if settings.B3_INGEST_USE_SELENIUM:
            anos = list(range(start.year, end.year + 1))
            return b3.fetch_b3_selenium(ident, anos)
        return b3.read_b3_static(ident, start, end)
    raise FetchError(f"prefixo sem fonte: {prefix}")


def _on_demand_defs() -> list[catalog.SeriesDef]:
    """Chaves yf:* gravadas sob demanda (fora do catálogo), para manter atualizadas."""
    known = {d.series_key for d in catalog.ingestible()}
    return [
        catalog.SeriesDef(key[3:], "YF", key, "level")
        for key in sorted(repository.available_keys())
        if key.startswith("yf:") and key not in known
    ]


def _warm_cache(today: dt.date) -> None:
    five_years_ago = today - dt.timedelta(days=5 * 365)  # janela quente ~5 anos
    for name in catalog.DEFAULT_BENCHMARKS:
        for start in (five_years_ago, _EPOCH):
            try:
                series.get_series(name, start, today)
            except Exception as exc:  # noqa: BLE001
                logger.warning("warm cache falhou para %s: %s", name, exc)


class Command(BaseCommand):
    help = "Ingere séries de mercado ao vivo em core.MarketSeries."

    def add_arguments(self, parser):
        parser.add_argument("--only", default="")
        parser.add_argument("--full", action="store_true")
        parser.add_argument("--since", default="")

    def handle(self, *args, **opts):
        only = {k.strip() for k in opts["only"].split(",") if k.strip()}
        today = dt.date.today()
        ok = failed = 0
        failed_keys: list[str] = []

        for sdef in [*catalog.ingestible(), *_on_demand_defs()]:
            if only and sdef.series_key not in only:
                continue
            if opts["full"]:
                start = _EPOCH
            elif opts["since"]:
                start = dt.date.fromisoformat(opts["since"])
            else:
                last = repository.last_date(sdef.series_key)
                start = (last + dt.timedelta(days=1)) if last else _EPOCH
            if start > today:
                self.stdout.write(f"  {sdef.series_key}: já atualizado")
                continue
            try:
                points = _fetch(sdef, start, today)
                n = repository.upsert_points(
                    sdef.series_key, sdef.source, sdef.kind, points
                )
            except (FetchError, DatabaseError) as exc:
                failed += 1
                failed_keys.append(sdef.series_key)
                logger.error("ingest %s falhou: %s", sdef.series_key, exc)
                continue
            ok += 1
            if n:
                self.stdout.write(f"  ok {sdef.series_key}: +{n} pontos")
            else:
                self.stdout.write(f"  {sdef.series_key}: sem novos pontos")

        _warm_cache(today)

        self.stdout.write(
            self.style.SUCCESS(f">> ingest: {ok} ok, {failed} falha(s).")
        )
        if failed_keys:
            self.stderr.write("falharam: " + ", ".join(failed_keys))
            sys.exit(1)
