import pytest

from apps.migration_ir.planner import Action, build_plan, validate_plan_input
from apps.migration_ir.wallet import Holding


def _h(ativo, classe, qtd, price, pm=None, vlr=0.0):
    return Holding(ativo, classe, qtd, price, pm, vlr, "", "BRL")


def _inp(holdings, actions, **kw):
    kw.setdefault("targets", [("A", 100)])
    return validate_plan_input(actions=actions, wallet=holdings, **kw)


def test_greedy_fills_cap_then_spills():
    hs = [_h("AAA3", "ACAO", 1500, 10.0, 5.0), _h("BBB3", "ACAO", 1000, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell"), Action("BBB3", "sell")]), {"A": 10.0})
    m1, m2 = plan["months"]
    assert [(s["ativo"], s["qtd"]) for s in m1["sales"]] == [("BBB3", 1000), ("AAA3", 1000)]
    assert m1["exempt_sales_total"] == 20000 and m1["ir_estimado_total"] == 0 and m1["warnings"] == []
    assert [(s["ativo"], s["qtd"]) for s in m2["sales"]] == [("AAA3", 500)]
    assert plan["totals"]["months"] == 2 and plan["totals"]["sold"] == 25000


def test_already_sold_reduces_month1_cap():
    hs = [_h("AAA3", "ACAO", 1000, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")], already_sold=15000), {"A": 10.0})
    assert [m["exempt_sales_total"] for m in plan["months"]] == [5000, 5000]


def test_already_sold_fills_month1_cap_without_exemption_warning():
    hs = [_h("AAA3", "ACAO", 100, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")], already_sold=20000), {"A": 10.0})
    m1, m2 = plan["months"]
    assert m1["sales"] == [] and m1["warnings"] == [] and m2["exempt_sales_total"] == 1000


def test_already_over_exemption_warning():
    hs = [_h("AAA3", "ACAO", 100, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")], already_sold=21000), {"A": 10.0})
    assert plan["months"][0]["warnings"] == ["already_over_exemption"]
    assert plan["months"][0]["sales"] == []


def test_unit_price_above_cap_sells_one_alone_with_ir():
    hs = [_h("CARA3", "ACAO", 2, 25000.0, 20000.0)]
    plan, _ = build_plan(_inp(hs, [Action("CARA3", "sell")]), {"A": 10.0})
    assert len(plan["months"]) == 2
    m1 = plan["months"][0]
    assert m1["sales"][0]["qtd"] == 1 and m1["sales"][0]["ir_estimado"] == pytest.approx(750.0)
    assert set(m1["warnings"]) == {"exceeds_cap", "exceeds_exemption"}


def test_unit_price_above_cap_skips_month1_with_outside_sales():
    hs = [_h("CARA3", "ACAO", 2, 25000.0, 20000.0)]
    plan, _ = build_plan(_inp(hs, [Action("CARA3", "sell")], already_sold=1000), {"A": 10.0})
    assert [len(m["sales"]) for m in plan["months"]] == [0, 1, 1]


def test_user_cap_below_20k_only_warns_exceeds_cap():
    hs = [_h("CARA3", "ACAO", 1, 16000.0, 10000.0)]
    plan, _ = build_plan(_inp(hs, [Action("CARA3", "sell")], monthly_cap=15000), {"A": 10.0})
    m1 = plan["months"][0]
    assert m1["warnings"] == ["exceeds_cap"] and m1["ir_estimado_total"] == 0


def test_fractional_residual_rides_last_sale():
    hs = [_h("AAA3", "ACAO", 10.5, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")]), {"A": 10.0})
    (m1,) = plan["months"]
    assert [s["qtd"] for s in m1["sales"]] == [10.5] and m1["exempt_sales_total"] == 105.0


def test_taxed_and_not_estimated_go_to_month1():
    hs = [_h("AAA3", "ACAO", 3000, 10.0, 5.0), _h("HGLG11", "FII", 50, 160.0, 150.0),
          _h("BOVA11", "ETF", 10, 100.0, 120.0), _h("BTC", "CRIPTO", 0.1, 300000.0, 200000.0)]
    acts = [Action(a, "sell") for a in ("AAA3", "HGLG11", "BOVA11", "BTC")]
    plan, _ = build_plan(_inp(hs, acts), {"A": 10.0})
    m1 = plan["months"][0]
    by = {s["ativo"]: s for s in m1["sales"]}
    assert by["HGLG11"]["ir_estimado"] == 100.0 and by["BOVA11"]["ir_estimado"] == 0.0
    assert by["BTC"]["ir_estimado"] is None and "not_estimated:BTC" in m1["warnings"]
    assert m1["taxed_sales_total"] == pytest.approx(8000 + 1000 + 30000)
    assert [s["ativo"] for s in m1["sales"]][:3] == ["HGLG11", "BOVA11", "BTC"]
    assert plan["totals"]["ir_not_estimated"] == ["BTC"] and plan["totals"]["ir_estimado"] == 100.0


def test_taxed_sale_without_pm_is_not_estimated():
    hs = [_h("HGLG11", "FII", 50, 160.0, None)]
    plan, _ = build_plan(_inp(hs, [Action("HGLG11", "sell")]), {"A": 10.0})
    s = plan["months"][0]["sales"][0]
    assert s["lucro_estimado"] is None and s["ir_estimado"] is None
    assert plan["totals"]["ir_not_estimated"] == ["HGLG11"]


def test_count_as_balances_and_purchases_accumulate():
    hs = [_h("AAA3", "ACAO", 3000, 10.0, 5.0), _h("IMAB11", "ETF RF", 50, 100.0, 90.0),
          _h("CDB1", "RF", 1, None, None, vlr=12000.0), _h("CDB2", "RF", 1, None, None, vlr=1000.0)]
    acts = [Action("AAA3", "sell"), Action("CDB1", "count_as", target="IMAB11"),
            Action("CDB2", "count_as", target="IMAB11", saldo=1500.0)]
    inp = _inp(hs, acts, targets=[("IMAB11", 50), ("VWRA11", 50)])
    plan, warns = build_plan(inp, {"IMAB11": 100.0, "VWRA11": 50.0})
    ib = {b["ticker"]: b for b in plan["initial_balances"]}
    assert ib["IMAB11"]["direct"] == 5000 and ib["IMAB11"]["total"] == 18500
    assert [(c["ativo"], c["saldo_source"]) for c in ib["IMAB11"]["counted"]] == [("CDB1", "cost"), ("CDB2", "user")]
    assert warns == ["cost_basis:CDB1"]
    m1, m2 = plan["months"]
    assert m2["purchases"][0]["saldo_final"] + m2["purchases"][1]["saldo_final"] == pytest.approx(18500 + 30000)
    assert m1["purchase_total"] == 20000 and m2["purchase_total"] == 10000


def test_contributions_only_single_month():
    hs = [_h("IMAB11", "ETF RF", 50, 100.0, 90.0)]
    inp = _inp(hs, [], targets=[("IMAB11", 100)], contributions_only=True, initial_cash=1000)
    plan, _ = build_plan(inp, {"IMAB11": 100.0})
    (m1,) = plan["months"]
    assert m1["sales"] == [] and m1["cash_in"] == 1000
    p = m1["purchases"][0]
    assert (p["ticker"], p["valor"], p["qtd_sugerida"], p["participacao_final"]) == ("IMAB11", 1000.0, 10, 1.0)
