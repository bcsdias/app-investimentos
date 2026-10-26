"""Regra de isenção por classe DLP e estimativa de IR (puro, sem Django).

Simplificações deliberadas: sem taxas/emolumentos, sem compensação de prejuízo, sem day-trade.
"""
from __future__ import annotations

from typing import Literal

TaxTreatment = Literal["exempt_20k", "taxed", "not_estimated"]

EXEMPTION_LIMIT = 20_000.0  # R$ de venda de ações por mês
ACAO_OVER_LIMIT_RATE = 0.15

# classe -> (tratamento, alíquota sobre o lucro). Ausente da tabela = not_estimated.
_TABLE: dict[str, tuple[TaxTreatment, float | None]] = {
    "ACAO": ("exempt_20k", None),
    "FII": ("taxed", 0.20),
    "ETF": ("taxed", 0.15),
    "BDR": ("taxed", 0.15),
    "BDR ETF": ("taxed", 0.15),
}


def tax_treatment(classe: str) -> tuple[TaxTreatment, float | None]:
    return _TABLE.get(str(classe or "").strip().upper(), ("not_estimated", None))


def estimate_profit(price: float, pm: float | None, qtd: float) -> float | None:
    if pm is None:
        return None
    return (price - pm) * qtd


def estimate_ir(lucro: float | None, rate: float) -> float | None:
    if lucro is None:
        return None
    return max(0.0, lucro) * rate
