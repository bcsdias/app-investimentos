"""Catálogo de séries lógicas. Python puro — porta de src/data/benchmarks_config.py.

Mapeia um nome lógico ('CDI', 'IMA-B', 'S&P 500 BRL') para: onde buscar o dado
cru (source/series_key) e como derivar a série servível (derivation).
"""
from __future__ import annotations

from dataclasses import dataclass

from apps.market.synthetics import SyntheticSpec


@dataclass(frozen=True)
class ProxySplice:
    bcb_key: str
    etf_key: str


@dataclass(frozen=True)
class ToBRL:
    base_name: str


@dataclass(frozen=True)
class Synthetic:
    spec: SyntheticSpec


@dataclass(frozen=True)
class Composite:
    weights: dict[str, float]


@dataclass(frozen=True)
class SeriesDef:
    name: str
    source: str | None
    series_key: str | None
    kind: str                  # "rate" | "level"
    periodicity: str = "daily"  # "daily" | "monthly"
    derivation: object | None = None


# --- Séries CRUAS (têm source/series_key; a ingestão grava estas) ---
_RAW: list[SeriesDef] = [
    SeriesDef("CDI", "BCB", "bcb:12", "rate"),
    SeriesDef("SELIC", "BCB", "bcb:11", "rate"),
    SeriesDef("IPCA", "BCB", "bcb:433", "rate", periodicity="monthly"),
    SeriesDef("IMA-B (BCB)", "BCB", "bcb:12466", "level"),
    SeriesDef("IMA-B 5 (BCB)", "BCB", "bcb:12467", "level"),
    SeriesDef("IMA-B 5+ (BCB)", "BCB", "bcb:12468", "level"),
    SeriesDef("IRF-M (BCB)", "BCB", "bcb:12461", "level"),
    SeriesDef("IMA-S (BCB)", "BCB", "bcb:12469", "level"),
    SeriesDef("S&P 500", "YF", "yf:^GSPC", "level"),
    SeriesDef("IVVB11", "YF", "yf:IVVB11.SA", "level"),
    SeriesDef("BOVA11", "YF", "yf:BOVA11.SA", "level"),
    SeriesDef("IMID", "YF", "yf:IMID.L", "level"),
    SeriesDef("Bitcoin", "YF", "yf:BTC-USD", "level"),
    SeriesDef("IMAB11 (proxy)", "YF", "yf:IMAB11.SA", "level"),
    SeriesDef("IRFM11 (proxy)", "YF", "yf:IRFM11.SA", "level"),
    SeriesDef("B5P211 (proxy)", "YF", "yf:B5P211.SA", "level"),
    SeriesDef("IB5M11 (proxy)", "YF", "yf:IB5M11.SA", "level"),
    SeriesDef("LFTS11 (proxy)", "YF", "yf:LFTS11.SA", "level"),
    SeriesDef("IFIX", "B3", "b3:IFIX", "level"),
    SeriesDef("SMLL", "B3", "b3:SMLL", "level"),
    SeriesDef("IDIV", "B3", "b3:IDIV", "level"),
    SeriesDef("USD/BRL", "PTAX", "ptax:USDBRL", "level"),
]

# --- Séries DERIVADAS (calculadas na leitura; a ingestão NÃO grava estas) ---
_DERIVED: list[SeriesDef] = [
    SeriesDef("IMA-B", None, None, "level",
              derivation=ProxySplice("bcb:12466", "yf:IMAB11.SA")),
    SeriesDef("IMA-B 5", None, None, "level",
              derivation=ProxySplice("bcb:12467", "yf:B5P211.SA")),
    SeriesDef("IMA-B 5+", None, None, "level",
              derivation=ProxySplice("bcb:12468", "yf:IB5M11.SA")),
    SeriesDef("IRF-M", None, None, "level",
              derivation=ProxySplice("bcb:12461", "yf:IRFM11.SA")),
    SeriesDef("IMA-S", None, None, "level",
              derivation=ProxySplice("bcb:12469", "yf:LFTS11.SA")),
    SeriesDef("S&P 500 BRL", None, None, "level", derivation=ToBRL("S&P 500")),
    SeriesDef("IMID BRL", None, None, "level", derivation=ToBRL("IMID")),
    SeriesDef("Bitcoin BRL", None, None, "level", derivation=ToBRL("Bitcoin")),
    SeriesDef("IPCA + 6%", None, None, "level",
              derivation=Synthetic(SyntheticSpec("IPCA", "+", 6.0))),
]

_BY_NAME: dict[str, SeriesDef] = {d.name: d for d in (_RAW + _DERIVED)}

DEFAULT_BENCHMARKS: list[str] = ["CDI", "IMA-B", "IMA-B 5", "IMA-B 5+", "IPCA + 6%"]


def all_names() -> list[str]:
    return list(_BY_NAME)


def resolve(name: str) -> SeriesDef:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"série desconhecida no catálogo: {name!r}") from exc


def ingestible() -> list[SeriesDef]:
    return [d for d in _RAW if d.series_key]


def legacy_csv_name(sdef: SeriesDef) -> str | None:
    key = sdef.series_key
    if not key:
        return None
    prefix, _, ident = key.partition(":")
    if prefix == "bcb":
        return f"BCB_{ident}.csv"
    if prefix == "yf":
        return "YF_" + ident.lstrip("^").replace(".", "_") + ".csv"
    if prefix == "b3":
        return f"{ident}_all.csv"
    if prefix == "td":
        titulo, _, venc = ident.partition(":")
        return f"TD_{titulo.replace(' ', '_').replace('+', 'mais')}_{venc[8:10]}-{venc[5:7]}-{venc[0:4]}.csv"
    return None  # ptax não tem CSV legado
