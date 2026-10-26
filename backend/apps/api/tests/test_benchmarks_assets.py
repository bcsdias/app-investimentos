import pytest

from apps.market import catalog


def test_catalog_groups_only_known_names(auth_client):
    r = auth_client.get("/api/v1/benchmarks/catalog")
    assert r.status_code == 200
    d = r.json()
    names = [n for g in d["groups"] for n in g["items"]]
    assert names and set(names) <= set(catalog.all_names())
    assert set(catalog.DEFAULT_BENCHMARKS) <= set(names)
    assert d["defaults"] == catalog.DEFAULT_BENCHMARKS
    assert d["synthetic"] == {"bases": ["IPCA", "CDI"], "operadores": ["+", "%"]}


def test_search_returns_assets(auth_client, assets_csv):
    r = auth_client.get("/api/v1/assets/search", {"q": "aapl"})
    assert r.status_code == 200
    assert r.json()[0] == {
        "sigla": "AAPL",
        "nome": "APPLE INC",
        "classe": "STOCK",
        "mercado": "INTL",
        "yf_ticker": "AAPL",
        "brl_variant": True,
    }


@pytest.mark.parametrize(
    "params",
    [
        {"q": "a"},
        {"q": "  "},
        {"q": "pe", "limit": 51},
        {"q": "pe", "limit": 0},
    ],
)
def test_search_invalid_query(auth_client, assets_csv, params):
    r = auth_client.get("/api/v1/assets/search", params)
    assert r.status_code == 422 and r.json()["code"] == "invalid_query"


def test_catalog_and_search_require_session(client, db):
    assert client.get("/api/v1/benchmarks/catalog").status_code == 401
    assert client.get("/api/v1/assets/search", {"q": "pe"}).status_code == 401
