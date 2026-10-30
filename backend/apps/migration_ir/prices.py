"""Cotação (BRL) dos ativos-alvo: price_fx da carteira, senão último fechamento em apps.market.

Nunca inventa preço: alvo sem cotação vira PlanError("price_unavailable").
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Literal

from apps.market import assets, series
from apps.migration_ir.planner import PlanError, PlanInput, target_holding

logger = logging.getLogger(__name__)

_WINDOW_DAYS = 10  # cobre feriados/fins de semana até o último fechamento


@dataclass(frozen=True)
class TargetPrice:
    price: float
    source: Literal["wallet", "market"]


def _last_close(name: str, today: dt.date) -> float | None:
    try:
        s = series.get_series(name, today - dt.timedelta(days=_WINDOW_DAYS), today)
    except KeyError:
        return None
    if s is None:
        return None
    s = s.dropna()
    if s.empty:
        return None
    price = float(s.iloc[-1])
    return price if price > 0 else None


def resolve_target_prices(
    inp: PlanInput, today: dt.date | None = None
) -> tuple[dict[str, TargetPrice], list[str]]:
    ref = today or dt.date.today()
    prices: dict[str, TargetPrice] = {}
    warnings: list[str] = []
    unknown: list[str] = []
    unavailable: list[str] = []

    for ticker in inp.targets:
        h = target_holding(inp, ticker)
        if h is not None and h.price:
            prices[ticker] = TargetPrice(h.price, "wallet")
            continue
        asset = assets.lookup(ticker, h.market_cod if h else None)
        if asset is None:
            (unavailable if h is not None else unknown).append(ticker)
            continue
        name = asset.yf_ticker if asset.mercado == "BR" else f"{asset.yf_ticker} BRL"
        price = _last_close(name, ref)
        if price is None:
            logger.info("alvo %s sem cotação em %s", ticker, name)
            unavailable.append(ticker)
            continue
        prices[ticker] = TargetPrice(price, "market")
        if asset.classe == "ETF_GB" or (h is not None and h.moeda.upper() == "GBP"):
            warnings.append(f"fx_approx:{ticker}")

    if unknown:
        raise PlanError(
            "invalid_allocation",
            "Um ou mais alvos não existem na carteira nem no universo de ativos.",
            {"tickers": unknown},
        )
    if unavailable:
        raise PlanError(
            "price_unavailable",
            "Não foi possível obter a cotação de um ou mais alvos.",
            {"tickers": unavailable},
        )
    return prices, warnings
