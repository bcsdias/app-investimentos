import pytest

from apps.migration_ir.planner import (
    Action,
    PlanError,
    calcular_compras_rebalanceadas,
    target_holding,
    validate_plan_input,
)
from apps.migration_ir.tests.fixtures import RESUMO
from apps.migration_ir.wallet import normalize_wallet

WALLET, _ = normalize_wallet(RESUMO)
T3 = [("IMAB11", 60), ("VWRA11", 35), ("BITH11", 5)]
SELL_PETR = [Action("PETR4", "sell")]


def _v(**kw):
    base = dict(targets=T3, actions=SELL_PETR, wallet=WALLET)
    return validate_plan_input(**{**base, **kw})


def test_rebalance_legacy_60_35_5():
    out = calcular_compras_rebalanceadas(10000, {"A": 60, "B": 35, "C": 5}, {"A": 10000, "B": 0, "C": 2000})
    assert out["A"] == pytest.approx(2631.578947, rel=1e-6)
    assert out["B"] == pytest.approx(7368.421053, rel=1e-6)
    assert out["C"] == 0.0 and sum(out.values()) == pytest.approx(10000)


@pytest.mark.parametrize("kw,code", [
    (dict(targets=[("IMAB11", 60), ("VWRA11", 39)]), "invalid_allocation"),
    (dict(targets=[("IMAB11", 50), ("imab11 ", 50)]), "invalid_allocation"),
    (dict(targets=[]), "invalid_allocation"),
    (dict(targets=[(f"T{i}", 100 / 11) for i in range(11)]), "invalid_allocation"),
    (dict(targets=[("IMAB11", 100), ("VWRA11", 0)]), "invalid_allocation"),
    (dict(monthly_cap=0), "invalid_parameters"),
    (dict(monthly_cap=20000.01), "invalid_parameters"),
    (dict(initial_cash=-1), "invalid_parameters"),
    (dict(already_sold=-1), "invalid_parameters"),
    (dict(actions=[Action("XPTO3", "sell")]), "invalid_holding"),
    (dict(actions=[Action("CDB-XP-2027", "sell")]), "invalid_holding"),
    (dict(actions=[Action("PETR4", "sell", saldo=100)]), "invalid_holding"),
    (dict(actions=[Action("IMAB11", "sell")]), "invalid_holding"),
    (dict(actions=[Action("IMAB11", "count_as", target="VWRA11")]), "invalid_holding"),
    (dict(actions=[Action("CDB-XP-2027", "count_as", target="BOVA11")]), "invalid_holding"),
    (dict(actions=[Action("CDB-XP-2027", "count_as")]), "invalid_holding"),
    (dict(actions=[Action("CDB-XP-2027", "count_as", target="IMAB11", saldo=0)]), "invalid_holding"),
    (dict(actions=[Action("PETR4", "sell"), Action("PETR4", "sell")]), "invalid_holding"),
    (dict(actions=[]), "nothing_to_plan"),
    (dict(contributions_only=True), "nothing_to_plan"),
])
def test_validation_errors(kw, code):
    with pytest.raises(PlanError) as ei:
        _v(**kw)
    assert ei.value.code == code


def test_invalid_holding_details_name_the_ativo():
    with pytest.raises(PlanError) as ei:
        _v(actions=[Action("XPTO3", "sell")])
    assert ei.value.details == {"ativo": "XPTO3"}


def test_target_ticker_is_normalized_and_matches_wallet():
    inp = _v(targets=[(" imab11", 100)])
    assert list(inp.targets) == ["IMAB11"]
    assert target_holding(inp, "IMAB11").ativo == "IMAB11"


def test_contributions_only_drops_sells_keeps_count_as():
    inp = _v(contributions_only=True, initial_cash=1000,
             actions=[Action("PETR4", "sell"), Action("CDB-XP-2027", "count_as", target="IMAB11")])
    assert list(inp.actions) == ["CDB-XP-2027"]


def test_cash_only_plan_is_valid():
    assert _v(actions=[], initial_cash=500).initial_cash == 500
