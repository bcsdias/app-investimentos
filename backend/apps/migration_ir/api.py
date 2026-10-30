"""Router fino do planejador: token -> /resumo -> wallet -> validação -> preços -> plano."""
from __future__ import annotations

import contextlib

from django.http import HttpRequest, HttpResponse
from django.utils import timezone
from ninja import Router

from apps.api.analysis import dlp_errors, get_token
from apps.api.errors import ApiError
from apps.market.services.sources import dlp
from apps.migration_ir.pdf import render_plan_pdf
from apps.migration_ir.planner import Action, PlanError, build_plan, validate_plan_input
from apps.migration_ir.prices import resolve_target_prices
from apps.migration_ir.schemas import PlanOut, PlanRequest, WalletOut
from apps.migration_ir.tax import DISCLAIMER, tax_treatment
from apps.migration_ir.wallet import Holding, normalize_wallet

router = Router(tags=["migration"])


@contextlib.contextmanager
def plan_errors():
    try:
        yield
    except PlanError as exc:
        raise ApiError(422, exc.code, exc.message, exc.details) from exc


def _load_wallet(user) -> tuple[list[Holding], list[str]]:
    token = get_token(user)
    with dlp_errors():
        payload = dlp.buscar_resumo_carteira(token)
    holdings, warnings = normalize_wallet(payload or {})
    if not holdings:
        raise ApiError(404, "portfolio_empty", "Nenhum papel com saldo encontrado na carteira.")
    return holdings, warnings


def compute_plan(user, payload: PlanRequest) -> dict:
    holdings, wallet_warnings = _load_wallet(user)
    with plan_errors():
        inp = validate_plan_input(
            targets=[(t.ticker, t.weight) for t in payload.targets],
            actions=[Action(h.ativo, h.action, h.target, h.saldo) for h in payload.holdings],
            wallet=holdings,
            monthly_cap=payload.monthly_cap,
            initial_cash=payload.initial_cash,
            already_sold=payload.already_sold_this_month,
            contributions_only=payload.contributions_only,
        )
        prices, price_warnings = resolve_target_prices(inp)
        data, plan_warnings = build_plan(inp, {t: p.price for t, p in prices.items()})
    data["prices"] = {t: {"price": p.price, "source": p.source} for t, p in prices.items()}
    return {
        "data": data,
        "meta": {
            "monthly_cap": inp.monthly_cap,
            "disclaimer": DISCLAIMER,
            "warnings": [*wallet_warnings, *price_warnings, *plan_warnings],
        },
    }


@router.get("/wallet", response=WalletOut)
def wallet(request: HttpRequest):
    holdings, warnings = _load_wallet(request.user)
    items = []
    for h in holdings:
        treatment, rate = tax_treatment(h.classe)
        items.append({
            "ativo": h.ativo,
            "classe": h.classe,
            "qtd": h.qtd,
            "price": h.price,
            "pm": h.pm,
            "saldo": h.saldo,
            "saldo_source": h.saldo_source,
            "vlr_investido": h.vlr_investido,
            "market_cod": h.market_cod,
            "moeda": h.moeda,
            "tax_treatment": treatment,
            "tax_rate": rate,
        })
    return {"data": items, "meta": {"total": sum(h.saldo for h in holdings), "warnings": warnings}}


@router.post("/plan", response=PlanOut)
def plan(request: HttpRequest, payload: PlanRequest):
    return compute_plan(request.user, payload)


@router.post("/plan.pdf")
def plan_pdf(request: HttpRequest, payload: PlanRequest):
    envelope = compute_plan(request.user, payload)
    now = timezone.localtime()
    resp = HttpResponse(render_plan_pdf(envelope, payload, generated_at=now), content_type="application/pdf")
    resp["Content-Disposition"] = f'attachment; filename="plano_migracao_ir_{now:%Y%m%d}.pdf"'
    resp["Access-Control-Expose-Headers"] = "Content-Disposition"
    return resp
