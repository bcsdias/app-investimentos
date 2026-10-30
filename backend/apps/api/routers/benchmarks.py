from __future__ import annotations

from collections import defaultdict

from django.http import HttpRequest
from ninja import Router

from apps.api.schemas import BenchmarkCatalogOut
from apps.market import catalog

router = Router(tags=["benchmarks"])


def _group_label(sdef: catalog.SeriesDef) -> str:
    if sdef.source == "ANBIMA":
        return "ANBIMA / Tesouro"
    if isinstance(sdef.derivation, catalog.ToBRL):
        return "Internacional / Cripto (BRL)"
    if isinstance(sdef.derivation, catalog.Synthetic):
        return "Sintéticos"
    if sdef.source == "BCB":
        return "Renda Fixa / Índices BCB"
    if sdef.source == "B3":
        return "Índices B3"
    if sdef.source == "YF":
        return "Mercado / ETFs"
    if sdef.source == "PTAX":
        return "Câmbio"
    return "Outros"


@router.get("/catalog", response=BenchmarkCatalogOut)
def benchmarks_catalog(request: HttpRequest):
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in catalog.all_names():
        sdef = catalog.resolve(name)
        grouped[_group_label(sdef)].append(name)

    groups = [{"name": gname, "items": items} for gname, items in grouped.items() if items]
    return {
        "groups": groups,
        "defaults": list(catalog.DEFAULT_BENCHMARKS),
        "synthetic": {
            "bases": ["IPCA", "CDI"],
            "operadores": ["+", "%"],
        },
    }
