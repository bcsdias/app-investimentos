import re

import pytest
from django.test import Client

from apps.core.models import UserToken
from apps.market.services.sources.base import DLPAuthError, DLPUnavailableError
from apps.migration_ir.tests.fixtures import PLAN_BODY

M = "/api/v1/migration"


def post(c, path, body):
    return c.post(f"{M}/{path}", body, content_type="application/json")


def test_wallet_lists_holdings(client_resumo):
    r = client_resumo.get(f"{M}/wallet")
    body = r.json()
    assert r.status_code == 200 and body["meta"]["total"] == pytest.approx(66430)
    first = {i["ativo"]: i for i in body["data"]}
    assert first["HGLG11"]["tax_treatment"] == "taxed" and first["HGLG11"]["tax_rate"] == 0.20
    assert (first["CDB-XP-2027"]["price"], first["CDB-XP-2027"]["saldo_source"]) == (None, "cost")
    assert [i["ativo"] for i in body["data"]][0] == "BTC"  # saldo decrescente


def test_plan_end_to_end(client_resumo):
    r = post(client_resumo, "plan", PLAN_BODY)
    assert r.status_code == 200, r.json()
    data, meta = r.json()["data"], r.json()["meta"]
    (m1,) = data["months"]
    assert m1["taxed_sales_total"] == 8000 and m1["exempt_sales_total"] == pytest.approx(11430)
    assert m1["ir_estimado_total"] == pytest.approx(100.0)
    buy = {p["ticker"]: p for p in m1["purchases"]}
    assert buy["IMAB11"]["valor"] == pytest.approx(4518) and buy["VWRA11"]["valor"] == pytest.approx(13048)
    assert buy["BITH11"]["valor"] == pytest.approx(1864) and buy["VWRA11"]["qtd_sugerida"] == 260
    assert buy["IMAB11"]["participacao_final"] == pytest.approx(0.6)
    assert data["prices"]["IMAB11"] == {"price": 100.0, "source": "wallet"}
    assert data["initial_balances"][0]["total"] == pytest.approx(17850)
    assert meta["disclaimer"].startswith("Valores de IR são estimativas")
    assert meta["monthly_cap"] == 20000.0 and meta["warnings"] == []


@pytest.mark.parametrize("patch,code", [
    ({"targets": [{"ticker": "IMAB11", "weight": 50}]}, "invalid_allocation"),
    ({"holdings": [{"ativo": "XPTO3", "action": "sell"}]}, "invalid_holding"),
    ({"monthly_cap": 25000}, "invalid_parameters"),
    ({"holdings": []}, "nothing_to_plan"),
    ({"targets": [{"ticker": "AAPL", "weight": 100}], "holdings": [{"ativo": "PETR4", "action": "sell"}]},
     "price_unavailable"),
])
def test_plan_business_errors(client_resumo, patch, code):
    r = post(client_resumo, "plan", {**PLAN_BODY, **patch})
    assert r.status_code == 422 and r.json()["code"] == code


def test_invalid_holding_details(client_resumo):
    r = post(client_resumo, "plan", {**PLAN_BODY, "holdings": [{"ativo": "XPTO3", "action": "sell"}]})
    assert r.json()["details"] == {"ativo": "XPTO3"}


def test_without_token_409(auth_client):
    assert auth_client.get(f"{M}/wallet").json()["code"] == "dlp_token_missing"


@pytest.mark.parametrize("exc,status,code", [
    (DLPAuthError("x"), 422, "dlp_token_rejected"),
    (DLPUnavailableError("x"), 502, "dlp_unavailable"),
])
def test_dlp_errors_map(user_with_token_client, monkeypatch, exc, status, code):
    def boom(tok):
        raise exc

    monkeypatch.setattr("apps.migration_ir.api.dlp.buscar_resumo_carteira", boom)
    r = user_with_token_client.get(f"{M}/wallet")
    assert r.status_code == status and r.json()["code"] == code


@pytest.mark.parametrize("payload", [None, {}, {"wallet": []}])
def test_empty_wallet_404(user_with_token_client, monkeypatch, payload):
    monkeypatch.setattr("apps.migration_ir.api.dlp.buscar_resumo_carteira", lambda tok: payload)
    r = post(user_with_token_client, "plan", PLAN_BODY)
    assert r.status_code == 404 and r.json()["code"] == "portfolio_empty"


def test_requires_session(client, db):
    assert client.get(f"{M}/wallet").status_code == 401


def test_post_requires_csrf(user_with_token, resumos):
    c = Client(enforce_csrf_checks=True)
    c.force_login(user_with_token)
    assert post(c, "plan", PLAN_BODY).status_code == 403


def test_users_are_isolated(user_with_token_client, other_user, resumos):
    ut = UserToken(user=other_user)
    ut.set_token("tok-ana")
    ut.save()
    ana = Client()
    ana.force_login(other_user)
    assert [i["ativo"] for i in ana.get(f"{M}/wallet").json()["data"]] == ["VALE3"]
    assert "VALE3" not in [i["ativo"] for i in user_with_token_client.get(f"{M}/wallet").json()["data"]]


def test_plan_pdf_endpoint(client_resumo):
    r = post(client_resumo, "plan.pdf", PLAN_BODY)
    assert r.status_code == 200 and r["Content-Type"] == "application/pdf"
    assert re.fullmatch(r'attachment; filename="plano_migracao_ir_\d{8}\.pdf"', r["Content-Disposition"])
    assert r["Access-Control-Expose-Headers"] == "Content-Disposition"
    assert r.content.startswith(b"%PDF")


def test_plan_pdf_errors_are_json(client_resumo):
    r = post(client_resumo, "plan.pdf", {**PLAN_BODY, "monthly_cap": 0})
    assert r.status_code == 422 and r.json()["code"] == "invalid_parameters"
