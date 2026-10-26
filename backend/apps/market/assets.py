"""Universo de tickers (ativos.csv) que o usuário pode comparar com a carteira.

Identificador público = ticker Yahoo. Ativos INTL também têm a variante
"<ticker> BRL" (conversão por PTAX via catalog.ToBRL).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache

from django.conf import settings

from apps.market.catalog import SeriesDef, ToBRL

_BR = {"ACAO", "ETF", "ETF RF", "FI_INFRA", "FI_SET", "FIDC", "FII", "FIP", "FIP_IE", "BDR", "BDR ETF"}
_INTL = {"ETF_GB", "ETF_US", "REIT", "STOCK"}
_BRL_SUFFIX = " BRL"


@dataclass(frozen=True)
class Asset:
    sigla: str
    nome: str
    classe: str
    mercado: str  # "BR" | "INTL"
    yf_ticker: str

    @property
    def brl_variant(self) -> bool:
        return self.mercado == "INTL"


def _yf_ticker(sigla: str, market_cod: str) -> str:
    prefix = market_cod.split(":", 1)[0].upper()
    if prefix == "BVMF":
        return f"{sigla}.SA"
    if prefix in ("LON", "LSE"):
        return f"{sigla}.L"
    return sigla


@lru_cache(maxsize=1)
def load_universe() -> dict[str, Asset]:
    universe: dict[str, Asset] = {}
    with open(settings.MARKET_ASSETS_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            sigla = (row.get("sigla") or "").strip()
            classe = (row.get("classe") or "").strip()
            if not sigla:
                continue
            if classe in _BR:
                mercado = "BR"
            elif classe in _INTL:
                mercado = "INTL"
            else:
                continue
            yf = _yf_ticker(sigla, (row.get("market_cod") or "").strip())
            universe.setdefault(
                yf, Asset(sigla, (row.get("razao_social") or "").strip(), classe, mercado, yf)
            )
    return universe


def resolve(name: str) -> SeriesDef | None:
    universe = load_universe()
    if name in universe:
        return SeriesDef(name, "YF", f"yf:{name}", "level")
    if name.endswith(_BRL_SUFFIX):
        base = name[: -len(_BRL_SUFFIX)]
        asset = universe.get(base)
        if asset is not None and asset.brl_variant:
            return SeriesDef(name, None, None, "level", derivation=ToBRL(base))
    return None


def search(q: str, limit: int = 20) -> list[Asset]:
    q = q.strip().lower()
    if not q:
        return []
    exact, prefix, name = [], [], []
    for a in load_universe().values():
        sigla = a.sigla.lower()
        if sigla == q:
            exact.append(a)
        elif sigla.startswith(q):
            prefix.append(a)
        elif q in a.nome.lower():
            name.append(a)
    return (exact + prefix + name)[:limit]
