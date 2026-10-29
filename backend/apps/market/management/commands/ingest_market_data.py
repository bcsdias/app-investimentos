"""Ingestão AO VIVO de séries de mercado -> core.MarketSeries (fonte de verdade).

Percorre catalog.ingestible(); para cada série busca da última data gravada até
hoje (incremental), grava o valor CRU via repository e, ao fim, aquece o cache
dos DEFAULT_BENCHMARKS. Uma série falhar NÃO aborta o lote; exit 1 se houve falha.
Agendado via supercronic no container de ingestão (Fase 5).
"""
from __future__ import annotations

import datetime as dt
import logging
import sys

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import DatabaseError

from apps.market import catalog, gaps, repository, series
from apps.market.services.sources import b3, bcb, ptax, tesouro, yahoo
from apps.market.services.sources.base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_EPOCH = catalog.EPOCH


class _PartialFetch(FetchError):
    def __init__(self, message: str, points: list[RawPoint]) -> None:
        super().__init__(message)
        self.points = points


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
            try:
                return b3.fetch_b3_selenium(ident, anos)
            except FetchError as exc:
                logger.error(
                    "Selenium falhou para %s (%s); caindo no CSV estático",
                    sdef.series_key,
                    exc,
                )
                fallback_pts = b3.read_b3_static(ident, start, end)
                raise _PartialFetch(str(exc), fallback_pts) from exc
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


class RefillResult(tuple):
    total_gaps: int
    points_written: int
    remaining_gaps: list[gaps.Gap]

    def __new__(
        cls,
        points_written: int,
        remaining_gaps: list[gaps.Gap],
        total_gaps: int = 0,
    ) -> RefillResult:
        instance = super().__new__(cls, (points_written, remaining_gaps))
        instance.points_written = points_written
        instance.remaining_gaps = remaining_gaps
        instance.total_gaps = total_gaps
        return instance


def _refill_gaps(
    sdef: catalog.SeriesDef,
    gap_list: list[gaps.Gap] | None = None,
) -> tuple[int, list[gaps.Gap]]:
    if gap_list is None:
        dates = repository.load_dates(sdef.series_key)
        gap_list = gaps.find_gaps(dates, sdef.periodicity)
    if not gap_list:
        return RefillResult(0, [], 0)

    points_written = 0
    remaining: list[gaps.Gap] = []
    for gap in gap_list:
        start = gap.start + dt.timedelta(days=1)
        end = gap.end - dt.timedelta(days=1)
        if start > end:
            remaining.append(gap)
            logger.warning(
                "lacuna sem dado na fonte em %s: %s..%s",
                sdef.series_key,
                gap.start,
                gap.end,
            )
            continue
        try:
            pts = _fetch(sdef, start, end)
        except _PartialFetch as exc:
            pts = exc.points
        except FetchError:
            pts = []
        if pts:
            n = repository.upsert_points(
                sdef.series_key, sdef.source, sdef.kind, pts
            )
            points_written += n
        else:
            remaining.append(gap)
            logger.warning(
                "lacuna sem dado na fonte em %s: %s..%s",
                sdef.series_key,
                gap.start,
                gap.end,
            )
    return RefillResult(points_written, remaining, len(gap_list))


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
            else:
                try:
                    points = _fetch(sdef, start, today)
                    n = repository.upsert_points(
                        sdef.series_key, sdef.source, sdef.kind, points
                    )
                except _PartialFetch as exc:
                    if exc.points:
                        repository.upsert_points(
                            sdef.series_key, sdef.source, sdef.kind, exc.points
                        )
                    failed += 1
                    failed_keys.append(sdef.series_key)
                    logger.error("ingest %s falhou: %s", sdef.series_key, exc)
                    continue
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

            res = _refill_gaps(sdef)
            total_gaps = getattr(res, "total_gaps", 0)
            if total_gaps:
                remaining_gaps = res[1] if isinstance(res, tuple) else []
                self.stdout.write(
                    f"  {sdef.series_key}: {total_gaps} lacuna(s) re-buscada(s), {len(remaining_gaps)} sem dado na fonte"
                )

        _warm_cache(today)

        self.stdout.write(
            self.style.SUCCESS(f">> ingest: {ok} ok, {failed} falha(s).")
        )
        if failed_keys:
            self.stderr.write("falharam: " + ", ".join(failed_keys))
            sys.exit(1)
