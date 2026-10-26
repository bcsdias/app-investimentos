"""Planejador de migração com isenção de IR (puro, sem Django).

validate_plan_input -> PlanInput; build_plan(PlanInput, preços) -> cronograma mês a mês.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from apps.migration_ir.wallet import Holding

MAX_TARGETS = 10
MAX_CAP = 20_000.0
_WEIGHT_TOL = 1e-6


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
    if not 0 < monthly_cap <= MAX_CAP:
        raise PlanError("invalid_parameters", "O teto mensal deve estar entre 0 (exclusivo) e 20.000.")
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
