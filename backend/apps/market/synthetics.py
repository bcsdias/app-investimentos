"""Contrato estruturado dos benchmarks sintéticos.

Substitui as 3 regexes espalhadas em financial_report.build_dataset. A API da
Fase 3 aceita SyntheticSpec direto; parse_legacy_string é só para migrar strings
antigas e como açúcar de frontend.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SPREAD = re.compile(r"^\s*(IPCA|CDI)\s*\+\s*(\d+(?:\.\d+)?)\s*%\s*$", re.IGNORECASE)
_PCT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*do\s*(CDI)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class SyntheticSpec:
    base: str        # "IPCA" | "CDI"
    operador: str    # "+" (spread a.a.) | "%" (percentual do índice)
    valor: float


def display_name(spec: SyntheticSpec) -> str:
    if spec.operador == "+":
        return f"{spec.base} + {spec.valor:g}%"
    return f"{spec.valor:g}% do {spec.base}"



def parse_legacy_string(s: str) -> SyntheticSpec | None:
    if not s:
        return None
    m = _SPREAD.match(s)
    if m:
        return SyntheticSpec(m.group(1).upper(), "+", float(m.group(2)))
    m = _PCT.match(s)
    if m:
        return SyntheticSpec(m.group(2).upper(), "%", float(m.group(1)))
    return None
