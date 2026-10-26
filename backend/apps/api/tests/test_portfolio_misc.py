import pandas as pd
import pytest

from apps.api.tests.conftest import PORTFOLIO_DF
from apps.market.services.sources.base import DLPAuthError

P = "/api/v1/portfolio"
BODY = {"start": "2024-01-02", "end": "2025-06-30", "benchmarks": ["CDI"]}


def post(c, path, body=BODY):
    return c.post(f"{P}/{path}", body, content_type="application/json")


def test_summary_period_values(portfolio_env):
    r = post(portfolio_env, "summary")
    assert r.status_code == 200
    d = r.json()["data"]
    df = PORTFOLIO_DF
    assert d["patrimonio_inicial"] == pytest.approx(df["vlr_mercado"].iloc[0])
    assert d["aportes_periodo"] == pytest.approx(
        df["vlr_investido"].iloc[-1] - df["vlr_investido"].iloc[0]
    )
    assert d["patrimonio"] == pytest.approx(df["vlr_mercado"].iloc[-1])
    assert (
        d["patrimonio_inicial"] + d["aportes_periodo"] + d["ganho_periodo"]
        == pytest.approx(d["patrimonio"])
    )
    assert d["max_drawdown"] <= 0
    assert d["data_inicio"] == "2024-03-01"
    assert d["data_fim"] == "2025-06-30"
    assert d["twr_total"] is not None
    assert d["cagr"] is not None
    assert d["volatilidade"] is not None
    assert d["sharpe"] is not None
    assert d["xirr"] is not None


def test_filters_lists_distinct_sorted(user_with_token_client, monkeypatch):
    df = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 3,
            "ativo": ["PETR4", "HGLG11", "PETR4"],
            "classe": ["ACAO", "FII", "ACAO"],
        }
    )
    monkeypatch.setattr("apps.api.routers.portfolio.dlp.buscar_historico", lambda tok: df)
    r = user_with_token_client.get(f"{P}/filters")
    assert r.status_code == 200
    assert r.json() == {"ativos": ["HGLG11", "PETR4"], "classes": ["ACAO", "FII"]}


def test_filters_unknown_columns_returns_empty(user_with_token_client, monkeypatch, caplog):
    df = pd.DataFrame({"date": ["2024-01-02"], "vlr_mercado": [100.0]})
    monkeypatch.setattr("apps.api.routers.portfolio.dlp.buscar_historico", lambda tok: df)
    r = user_with_token_client.get(f"{P}/filters")
    assert r.status_code == 200
    assert r.json() == {"ativos": [], "classes": []}
    assert caplog.records


def test_filters_dlp_rejected_422(user_with_token_client, monkeypatch):
    def boom(tok):
        raise DLPAuthError("401")

    monkeypatch.setattr("apps.api.routers.portfolio.dlp.buscar_historico", boom)
    r = user_with_token_client.get(f"{P}/filters")
    assert r.status_code == 422 and r.json()["code"] == "dlp_token_rejected"


def test_refresh_invalidates_user_cache(user_with_token_client, monkeypatch):
    seen = []
    monkeypatch.setattr("apps.api.routers.portfolio.dlp.invalidate_user_cache", seen.append)
    assert user_with_token_client.post(f"{P}/refresh").status_code == 204
    assert seen == ["tok-teste"]


def test_refresh_without_token_409(auth_client):
    r = auth_client.post(f"{P}/refresh")
    assert r.status_code == 409 and r.json()["code"] == "dlp_token_missing"


def test_extract_distinct_dedups_strips_and_drops_blanks():
    from apps.api.routers.portfolio import _extract_distinct

    df = pd.DataFrame({"Classe": ["ACAO", " ACAO", "FII ", None, float("nan"), "", "  ", "ETF", "ACAO"]})
    assert _extract_distinct(df, ("classe",), "classe") == ["ACAO", "ETF", "FII"]
