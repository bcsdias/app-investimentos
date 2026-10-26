from __future__ import annotations

from typing import Literal

from ninja import Schema
from pydantic import Field


class TargetIn(Schema):
    ticker: str
    weight: float


class HoldingActionIn(Schema):
    ativo: str
    action: Literal["sell", "count_as", "ignore"]
    target: str | None = None
    saldo: float | None = None


class PlanRequest(Schema):
    targets: list[TargetIn]
    holdings: list[HoldingActionIn] = Field(default_factory=list)
    monthly_cap: float = 20000.0
    initial_cash: float = 0.0
    already_sold_this_month: float = 0.0
    contributions_only: bool = False


class WalletItemOut(Schema):
    ativo: str
    classe: str
    qtd: float
    price: float | None
    pm: float | None
    saldo: float
    saldo_source: str
    vlr_investido: float
    market_cod: str
    moeda: str
    tax_treatment: str
    tax_rate: float | None = None


class WalletMetaOut(Schema):
    total: float
    warnings: list[str]


class WalletOut(Schema):
    data: list[WalletItemOut]
    meta: WalletMetaOut


class CountedOut(Schema):
    ativo: str
    saldo: float
    saldo_source: str


class InitialBalanceOut(Schema):
    ticker: str
    direct: float
    counted: list[CountedOut]
    total: float


class PriceOut(Schema):
    price: float
    source: str


class SaleOut(Schema):
    ativo: str
    classe: str
    qtd: float
    valor: float
    tax_treatment: str
    lucro_estimado: float | None
    ir_estimado: float | None


class PurchaseOut(Schema):
    ticker: str
    valor: float
    price: float
    qtd_sugerida: int
    saldo_final: float
    participacao_final: float


class MonthOut(Schema):
    month: int
    sales: list[SaleOut]
    exempt_sales_total: float
    taxed_sales_total: float
    cash_in: float
    purchase_total: float
    purchases: list[PurchaseOut]
    ir_estimado_total: float
    warnings: list[str]


class TotalsOut(Schema):
    months: int
    sold: float
    bought: float
    ir_estimado: float
    ir_not_estimated: list[str]


class PlanDataOut(Schema):
    initial_balances: list[InitialBalanceOut]
    prices: dict[str, PriceOut]
    months: list[MonthOut]
    totals: TotalsOut


class PlanMetaOut(Schema):
    monthly_cap: float
    disclaimer: str
    warnings: list[str]


class PlanOut(Schema):
    data: PlanDataOut
    meta: PlanMetaOut
