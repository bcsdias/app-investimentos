"""Planejador de migração com isenção de IR (puro, sem Django).

validate_plan_input -> PlanInput; build_plan(PlanInput, preços) -> cronograma mês a mês.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from apps.migration_ir.tax import (
    ACAO_OVER_LIMIT_RATE,
    EXEMPTION_LIMIT,
    estimate_ir,
    estimate_profit,
    tax_treatment,
)
from apps.migration_ir.wallet import Holding

MAX_TARGETS = 10
MAX_CAP = 20_000.0
MIN_CAP = 100.0
MAX_MONTHS = 240
_WEIGHT_TOL = 1e-6
_EPS = 1e-6  # folga para comparar R$ em float


class PlanError(Exception):
    """Erro de negócio do planejador; a API converte em ApiError 422."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class Action:
    ativo: str
    action: Literal["sell", "count_as", "ignore"]
    target: str | None = None
    saldo: float | None = None


@dataclass(frozen=True)
class PlanInput:
    targets: dict[str, float]  # ticker normalizado -> peso, na ordem do request
    wallet: dict[str, Holding]  # por ativo
    actions: dict[str, Action]  # só sell / count_as
    monthly_cap: float
    initial_cash: float
    already_sold: float
    contributions_only: bool


def _norm_ticker(t: str) -> str:
    return str(t or "").strip().upper()


def _target_holdings(targets, wallet: dict[str, Holding]) -> dict[str, Holding]:
    by_upper = {a.strip().upper(): h for a, h in wallet.items()}
    return {t: by_upper[t] for t in targets if t in by_upper}


def target_holding(inp: PlanInput, ticker: str) -> Holding | None:
    return _target_holdings([ticker], inp.wallet).get(ticker)


def _validate_targets(targets: list[tuple[str, float]]) -> dict[str, float]:
    if not 1 <= len(targets) <= MAX_TARGETS:
        raise PlanError("invalid_allocation", f"Informe de 1 a {MAX_TARGETS} ativos-alvo.")
    out: dict[str, float] = {}
    for ticker, weight in targets:
        t = _norm_ticker(ticker)
        if not t:
            raise PlanError("invalid_allocation", "Ticker de alvo vazio.")
        if t in out:
            raise PlanError("invalid_allocation", f"O alvo {t} aparece mais de uma vez.")
        if not weight > 0:
            raise PlanError("invalid_allocation", f"O peso do alvo {t} deve ser maior que zero.")
        out[t] = float(weight)
    if abs(sum(out.values()) - 100.0) > _WEIGHT_TOL:
        raise PlanError("invalid_allocation", "Os pesos dos alvos devem somar 100.")
    return out


def _validate_params(monthly_cap: float, initial_cash: float, already_sold: float) -> None:
    if not all(math.isfinite(v) for v in (monthly_cap, initial_cash, already_sold)):
        raise PlanError("invalid_parameters", "Valores monetários devem ser números finitos.")
    if not MIN_CAP <= monthly_cap <= MAX_CAP:
        raise PlanError("invalid_parameters", "O teto mensal deve estar entre R$ 100 e R$ 20.000.")
    if initial_cash < 0 or already_sold < 0:
        raise PlanError("invalid_parameters", "Valores monetários não podem ser negativos.")


def _bad_holding(ativo: str, message: str) -> PlanError:
    return PlanError("invalid_holding", message, {"ativo": ativo})


def _validate_actions(
    actions: list[Action], wallet: dict[str, Holding], targets: dict[str, float]
) -> dict[str, Action]:
    target_ativos = {h.ativo for h in _target_holdings(targets, wallet).values()}
    seen: set[str] = set()
    out: dict[str, Action] = {}
    for a in actions:
        if a.ativo in seen:
            raise _bad_holding(a.ativo, f"O papel {a.ativo} aparece mais de uma vez.")
        seen.add(a.ativo)
        h = wallet.get(a.ativo)
        if h is None:
            raise _bad_holding(a.ativo, f"O papel {a.ativo} não está na carteira.")
        if a.saldo is not None and a.action != "count_as":
            raise _bad_holding(a.ativo, f"Saldo informado só vale para papéis associados a um alvo ({a.ativo}).")
        if a.action == "ignore":
            continue
        if a.ativo in target_ativos:
            raise _bad_holding(a.ativo, f"O alvo {a.ativo} não pode ser vendido nem associado a outro alvo.")
        if a.action == "sell":
            if h.price is None:
                raise _bad_holding(a.ativo, f"O papel {a.ativo} não pode ser vendido: não tem cotação.")
            out[a.ativo] = a
            continue
        target = _norm_ticker(a.target or "")
        if target not in targets:
            raise _bad_holding(a.ativo, f"O papel {a.ativo} deve ser associado a um dos alvos.")
        if a.saldo is not None and not a.saldo > 0:
            raise _bad_holding(a.ativo, f"O saldo informado para {a.ativo} deve ser maior que zero.")
        out[a.ativo] = Action(a.ativo, "count_as", target, a.saldo)
    return out


def validate_plan_input(
    *,
    targets: list[tuple[str, float]],
    actions: list[Action],
    wallet: list[Holding],
    monthly_cap: float = 20000.0,
    initial_cash: float = 0.0,
    already_sold: float = 0.0,
    contributions_only: bool = False,
) -> PlanInput:
    tgt = _validate_targets(targets)
    _validate_params(monthly_cap, initial_cash, already_sold)
    by_ativo = {h.ativo: h for h in wallet}
    valid = _validate_actions(actions, by_ativo, tgt)
    if contributions_only:
        valid = {k: a for k, a in valid.items() if a.action != "sell"}
        if initial_cash <= 0:
            raise PlanError("nothing_to_plan", "No modo apenas aportes, informe um caixa maior que zero.")
    elif initial_cash <= 0 and not any(a.action == "sell" for a in valid.values()):
        raise PlanError("nothing_to_plan", "Nada a planejar: marque papéis para vender ou informe caixa.")
    return PlanInput(
        targets=tgt,
        wallet=by_ativo,
        actions=valid,
        monthly_cap=float(monthly_cap),
        initial_cash=float(initial_cash),
        already_sold=float(already_sold),
        contributions_only=contributions_only,
    )


def calcular_compras_rebalanceadas(total_aporte, pesos_alvo, valores_atuais):
    """
    Calcula o valor de compra para cada ativo alvo baseado em rebalanceamento.
    Evita vendas de ativos alvo (rebalanceamento apenas com aportes).

    Cópia literal de src/ui/pages/3_migracao_ir.py::calcular_compras_rebalanceadas.
    """
    ativos = list(pesos_alvo.keys())
    soma_pesos = sum(pesos_alvo.values())
    if soma_pesos == 0:
        return {a: 0.0 for a in ativos}

    w_alvo = {a: pesos_alvo[a] / soma_pesos for a in ativos}
    valores_pos = {a: valores_atuais.get(a, 0.0) for a in ativos}

    ativos_ativos = set(ativos)
    while True:
        soma_w_ativos = sum(w_alvo[a] for a in ativos_ativos)
        if soma_w_ativos == 0:
            for a in ativos:
                valores_pos[a] = valores_atuais.get(a, 0.0)
            break

        soma_v_ativos = sum(valores_atuais.get(a, 0.0) for a in ativos_ativos)
        total_alocado_ativos = soma_v_ativos + total_aporte

        diferencas = {}
        for a in ativos_ativos:
            v_alvo = (w_alvo[a] / soma_w_ativos) * total_alocado_ativos
            diferencas[a] = v_alvo - valores_atuais.get(a, 0.0)

        excesso = [a for a, diff in diferencas.items() if diff < 0]
        if not excesso:
            for a in ativos:
                if a in ativos_ativos:
                    valores_pos[a] = valores_atuais.get(a, 0.0) + diferencas[a]
                else:
                    valores_pos[a] = valores_atuais.get(a, 0.0)
            break
        else:
            ativos_ativos -= set(excesso)

    compras = {}
    for a in ativos:
        compras[a] = max(0.0, valores_pos[a] - valores_atuais.get(a, 0.0))

    return compras


# --- build_plan --------------------------------------------------------------


@dataclass
class _Lot:
    holding: Holding
    qtd: float

    @property
    def valor(self) -> float:
        return self.qtd * self.holding.price


def _initial_balances(inp: PlanInput) -> tuple[list[dict], dict[str, float], list[str]]:
    warnings: list[str] = []
    out: list[dict] = []
    saldos: dict[str, float] = {}
    for t in inp.targets:
        h = target_holding(inp, t)
        direct = h.saldo if h else 0.0
        if h and h.saldo_source == "cost":
            warnings.append(f"cost_basis:{h.ativo}")
        counted = []
        for a in inp.actions.values():
            if a.action != "count_as" or a.target != t:
                continue
            ch = inp.wallet[a.ativo]
            if a.saldo is not None:
                saldo, source = float(a.saldo), "user"
            else:
                saldo, source = ch.saldo, ch.saldo_source
                if source == "cost":
                    warnings.append(f"cost_basis:{a.ativo}")
            counted.append({"ativo": a.ativo, "saldo": saldo, "saldo_source": source})
        total = direct + sum(c["saldo"] for c in counted)
        out.append({"ticker": t, "direct": direct, "counted": counted, "total": total})
        saldos[t] = total
    return out, saldos, warnings


def _schedule_stocks(inp: PlanInput, stocks: list[Holding]) -> list[list[_Lot]]:
    """Guloso: enche cada mês até o teto com unidades inteiras (spec §7.2, D3, D12)."""
    cap = inp.monthly_cap
    months: list[list[_Lot]] = [[]]
    remaining = max(0.0, cap - inp.already_sold)

    def is_empty(i: int) -> bool:
        return not months[i] and (i > 0 or inp.already_sold == 0)

    def close() -> None:
        nonlocal remaining
        months.append([])
        remaining = cap
        if len(months) > MAX_MONTHS:
            raise PlanError(
                "invalid_parameters",
                f"O plano passaria de {MAX_MONTHS} meses; aumente o teto mensal ou venda menos papéis.",
            )

    for h in sorted(stocks, key=lambda x: (x.saldo, x.ativo)):
        price, q, last = h.price, h.qtd, None
        while q >= 1:
            n = min(math.floor((remaining + _EPS) / price), math.floor(q))
            cur = len(months) - 1
            if n >= 1:
                last = _Lot(h, n)
                months[cur].append(last)
                remaining -= n * price
                q -= n
            elif is_empty(cur) and price > cap:
                last = _Lot(h, 1)
                months[cur].append(last)
                q -= 1
                close()
            else:
                close()
        if q > _EPS:
            # A fração vai junto da última venda só se couber no teto; senão, vira lote no próximo mês.
            extra = q * price
            if extra > remaining + _EPS:
                close()
            if last is not None and any(lot is last for lot in months[-1]):
                last.qtd += q
            else:
                months[-1].append(_Lot(h, q))
            remaining -= extra

    while len(months) > 1 and not months[-1]:
        months.pop()
    return months


def _sale_record(lot: _Lot, over_limit: bool) -> dict:
    h = lot.holding
    treatment, rate = tax_treatment(h.classe)
    lucro = estimate_profit(h.price, h.pm, lot.qtd)
    if treatment == "taxed":
        ir = estimate_ir(lucro, rate)
    elif treatment == "exempt_20k":
        ir = estimate_ir(lucro, ACAO_OVER_LIMIT_RATE) if over_limit else 0.0
    else:
        ir = None
    return {
        "ativo": h.ativo,
        "classe": h.classe,
        "qtd": lot.qtd,
        "valor": lot.valor,
        "tax_treatment": treatment,
        "lucro_estimado": lucro,
        "ir_estimado": ir,
    }


def _purchases(inp: PlanInput, valor: float, saldos: dict[str, float], prices: dict[str, float]) -> list[dict]:
    compras = calcular_compras_rebalanceadas(valor, inp.targets, saldos)
    for t in inp.targets:
        saldos[t] += compras[t]
    total = sum(saldos.values())
    return [
        {
            "ticker": t,
            "valor": compras[t],
            "price": prices[t],
            "qtd_sugerida": math.floor(compras[t] / prices[t] + 1e-9),
            "saldo_final": saldos[t],
            "participacao_final": saldos[t] / total if total else 0.0,
        }
        for t in inp.targets
    ]


def build_plan(inp: PlanInput, prices: dict[str, float]) -> tuple[dict, list[str]]:
    """Cronograma mês a mês (spec §5.2/§7). Devolve (data sem `prices`, warnings globais)."""
    initial_balances, saldos, warnings = _initial_balances(inp)

    sells = [inp.wallet[a.ativo] for a in inp.actions.values() if a.action == "sell"]
    taxed = [_Lot(h, h.qtd) for h in sells if tax_treatment(h.classe)[0] != "exempt_20k"]
    stock_months = _schedule_stocks(inp, [h for h in sells if tax_treatment(h.classe)[0] == "exempt_20k"])

    months: list[dict] = []
    ir_not_estimated: list[str] = []
    for i, exempt in enumerate(stock_months):
        month_no = i + 1
        outside = inp.already_sold if month_no == 1 else 0.0
        exempt_total = sum(lot.valor for lot in exempt)
        stock_total = exempt_total + outside
        month_warnings: list[str] = []
        if month_no == 1 and inp.already_sold > EXEMPTION_LIMIT:
            month_warnings.append("already_over_exemption")
        over = bool(exempt) and stock_total > EXEMPTION_LIMIT + _EPS
        if exempt and stock_total > inp.monthly_cap + _EPS:
            month_warnings.append("exceeds_cap")
        if over:
            month_warnings.append("exceeds_exemption")

        month_taxed = taxed if month_no == 1 else []
        sales = [_sale_record(lot, False) for lot in month_taxed] + [_sale_record(lot, over) for lot in exempt]
        for s in sales:
            if s["ir_estimado"] is None:
                month_warnings.append(f"not_estimated:{s['ativo']}")
                if s["ativo"] not in ir_not_estimated:
                    ir_not_estimated.append(s["ativo"])

        taxed_total = sum(lot.valor for lot in month_taxed)
        cash_in = inp.initial_cash if month_no == 1 else 0.0
        purchase_total = taxed_total + exempt_total + cash_in
        months.append({
            "month": month_no,
            "sales": sales,
            "exempt_sales_total": exempt_total,
            "taxed_sales_total": taxed_total,
            "cash_in": cash_in,
            "purchase_total": purchase_total,
            "purchases": _purchases(inp, purchase_total, saldos, prices),
            "ir_estimado_total": sum(s["ir_estimado"] or 0.0 for s in sales),
            "warnings": month_warnings,
        })

    totals = {
        "months": len(months),
        "sold": sum(s["valor"] for m in months for s in m["sales"]),
        "bought": sum(p["valor"] for m in months for p in m["purchases"]),
        "ir_estimado": sum(m["ir_estimado_total"] for m in months),
        "ir_not_estimated": ir_not_estimated,
    }
    return {"initial_balances": initial_balances, "months": months, "totals": totals}, warnings
