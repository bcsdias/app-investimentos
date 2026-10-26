import datetime as dt

import pytest

from apps.migration_ir.planner import PlanError, validate_plan_input
from apps.migration_ir.prices import TargetPrice, resolve_target_prices
from apps.migration_ir.tests.fixtures import RESUMO
from apps.migration_ir.wallet import Holding, normalize_wallet

WALLET, _ = normalize_wallet(RESUMO)
TODAY = dt.date(2026, 10, 26)


def _inp(targets, wallet=WALLET):
    return validate_plan_input(targets=targets, actions=[], wallet=wallet, initial_cash=100)


def test_wallet_price_first_then_market(fake_series):
    prices, warns = resolve_target_prices(_inp([("IMAB11", 60), ("VWRA11", 35), ("BITH11", 5)]), today=TODAY)
    assert prices["IMAB11"] == TargetPrice(100.0, "wallet")
    assert prices["VWRA11"] == TargetPrice(50.0, "market") and prices["BITH11"].price == 20.0
    assert "IMAB11.SA" not in fake_series.calls and warns == []


def test_etf_gb_uses_brl_variant_and_warns(fake_series):
    wallet = [Holding("GDX", "ETF_GB", 10, None, None, 1000.0, "LON:GDX", "GBP")]
    prices, warns = resolve_target_prices(_inp([("GDX", 100)], wallet), today=TODAY)
    assert prices["GDX"] == TargetPrice(7.0, "market")
    assert fake_series.calls == ["GDX.L BRL"] and warns == ["fx_approx:GDX"]


def test_unknown_target_is_invalid_allocation(fake_series):
    with pytest.raises(PlanError) as ei:
        resolve_target_prices(_inp([("ZZZZ11", 50), ("AAPL", 50)]), today=TODAY)
    assert ei.value.code == "invalid_allocation" and ei.value.details == {"tickers": ["ZZZZ11"]}


def test_missing_quote_is_price_unavailable(fake_series):
    with pytest.raises(PlanError) as ei:
        resolve_target_prices(_inp([("AAPL", 100)]), today=TODAY)
    assert ei.value.code == "price_unavailable" and ei.value.details == {"tickers": ["AAPL"]}
    assert fake_series.calls == ["AAPL BRL"]
