import pytest

from apps.migration_ir.planner import Action, PlanError, build_plan, validate_plan_input
from apps.migration_ir.wallet import Holding


def _h(ativo: str, classe: str, qtd: float, price: float, pm: float | None = None) -> Holding:
    return Holding(ativo, classe, qtd, price, pm, 0.0, "", "BRL")


def _inp(holdings: list[Holding], actions: list[Action], **kw) -> validate_plan_input:
    kw.setdefault("targets", [("A", 100)])
    return validate_plan_input(actions=actions, wallet=holdings, **kw)


def test_plan_with_exactly_max_months_is_accepted():
    """Entrada com cronograma normal de exatamente 240 meses é aceita."""
    hs = [_h("AAA3", "ACAO", 240, 100.0, 50.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")], monthly_cap=100.0), {"A": 10.0})
    assert len(plan["months"]) == 240
    assert plan["totals"]["months"] == 240


def test_plan_with_max_months_plus_one_is_rejected():
    """Entrada com 241 meses no cronograma normal é rejeitada com PlanError."""
    hs = [_h("AAA3", "ACAO", 241, 100.0, 50.0)]
    with pytest.raises(PlanError) as exc_info:
        build_plan(_inp(hs, [Action("AAA3", "sell")], monthly_cap=100.0), {"A": 10.0})
    assert exc_info.value.code == "invalid_parameters"
    assert "240" in exc_info.value.message


def test_plan_with_exactly_max_months_unit_above_cap_is_accepted():
    """Unidade acima do teto com exatamente 240 meses é aceita (close não conta mês vazio no fim)."""
    hs = [_h("CARA3", "ACAO", 240, 150.0, 100.0)]
    plan, _ = build_plan(_inp(hs, [Action("CARA3", "sell")], monthly_cap=100.0), {"A": 10.0})
    assert len(plan["months"]) == 240
    assert plan["totals"]["months"] == 240


def test_plan_with_max_months_plus_one_unit_above_cap_is_rejected():
    """Unidade acima do teto com 241 meses é rejeitada com PlanError."""
    hs = [_h("CARA3", "ACAO", 241, 150.0, 100.0)]
    with pytest.raises(PlanError) as exc_info:
        build_plan(_inp(hs, [Action("CARA3", "sell")], monthly_cap=100.0), {"A": 10.0})
    assert exc_info.value.code == "invalid_parameters"
    assert "240" in exc_info.value.message


def test_plan_with_already_sold_and_unit_above_cap_exactly_240_is_accepted():
    """Mês 1 com already_sold + 239 unidades acima do teto = 240 meses aceitos."""
    hs = [_h("CARA3", "ACAO", 239, 150.0, 100.0)]
    plan, _ = build_plan(_inp(hs, [Action("CARA3", "sell")], monthly_cap=100.0, already_sold=100.0), {"A": 10.0})
    assert len(plan["months"]) == 240
    assert plan["totals"]["months"] == 240


def test_plan_with_already_sold_and_unit_above_cap_241_is_rejected():
    """Mês 1 com already_sold + 240 unidades acima do teto = 241 meses rejeitados."""
    hs = [_h("CARA3", "ACAO", 240, 150.0, 100.0)]
    with pytest.raises(PlanError) as exc_info:
        build_plan(_inp(hs, [Action("CARA3", "sell")], monthly_cap=100.0, already_sold=100.0), {"A": 10.0})
    assert exc_info.value.code == "invalid_parameters"
    assert "240" in exc_info.value.message


def test_plan_with_fractional_residual_spilling_into_241_is_rejected():
    """240 meses completos + resíduo fracionário que não cabe no teto abre mês 241 e é rejeitado."""
    hs = [_h("AAA3", "ACAO", 240.5, 100.0, 50.0)]
    with pytest.raises(PlanError) as exc_info:
        build_plan(_inp(hs, [Action("AAA3", "sell")], monthly_cap=100.0), {"A": 10.0})
    assert exc_info.value.code == "invalid_parameters"
    assert "240" in exc_info.value.message


def test_plan_with_fractional_residual_fitting_in_month_240_is_accepted():
    """239 meses completos + mês 240 com fração que cabe no teto é aceito."""
    hs = [_h("AAA3", "ACAO", 239.5, 100.0, 50.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")], monthly_cap=100.0), {"A": 10.0})
    assert len(plan["months"]) == 240
    assert plan["totals"]["months"] == 240

