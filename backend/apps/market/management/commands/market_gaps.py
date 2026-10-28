"""Diagnóstico de lacunas em séries de mercado em core.MarketSeries.

Percorre catalog.ingestible() e as chaves yf:* sob demanda (reutilizando
_on_demand_defs de ingest_market_data). Para cada série, carrega as datas
gravadas e detecta lacunas conforme a tolerância da cadência (diária >10 dias,
mensal >45 dias). Sempre exit 0.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand

from apps.market import catalog, repository
from apps.market.gaps import find_gaps
from apps.market.management.commands.ingest_market_data import _on_demand_defs


class Command(BaseCommand):
    help = "Diagnostica lacunas nas séries de mercado gravadas no banco."

    def add_arguments(self, parser):
        parser.add_argument(
            "--only",
            default="",
            help="Filtrar por chaves específicas separadas por vírgula",
        )

    def handle(self, *args, **opts):
        only = {k.strip() for k in opts["only"].split(",") if k.strip()}

        for sdef in [*catalog.ingestible(), *_on_demand_defs()]:
            if only and sdef.series_key not in only:
                continue

            dates = repository.load_dates(sdef.series_key)
            last = dates[-1] if dates else None
            gaps = find_gaps(dates, sdef.periodicity)

            self.stdout.write(
                f"{sdef.series_key} ({sdef.periodicity}) última={last} lacunas={len(gaps)}"
            )
            for gap in gaps:
                dias = (gap.end - gap.start).days
                self.stdout.write(f"  {gap.start} → {gap.end} ({dias} dias)")
