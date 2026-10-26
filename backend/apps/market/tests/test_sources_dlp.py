import threading
import time

import pandas as pd
import pytest
import requests

from apps.market.services.sources import dlp
from apps.market.services.sources.base import DLPAuthError, DLPError, DLPUnavailableError


@pytest.fixture(autouse=True)
def _locmem_cache(settings):
    settings.CACHES = {
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}
    }
    from django.core.cache import cache

    cache.clear()


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


def test_buscar_historico_returns_dataframe_and_caches(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(url)
        return _Resp({"historico": [{"data": "2024-01-02", "vlr_mercado": 100}]})

    monkeypatch.setattr(dlp.requests, "get", fake_get)
    df1 = dlp.buscar_historico("DLP-tok")
    df2 = dlp.buscar_historico("DLP-tok")  # segundo hit vem do cache
    assert isinstance(df1, pd.DataFrame)
    assert "date" in df1.columns  # 'data' normalizado para 'date'
    assert len(calls) == 1


def test_buscar_historico_empty_returns_none(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({"historico": []}))
    assert dlp.buscar_historico("DLP-tok") is None


def test_buscar_historico_http_error_raises_dlperror(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({}, status=500))
    with pytest.raises(DLPError):
        dlp.buscar_historico("DLP-tok")


def test_buscar_resumo_carteira_roundtrip(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({"wallet": [1, 2]}))
    assert dlp.buscar_resumo_carteira("DLP-tok") == {"wallet": [1, 2]}


@pytest.mark.parametrize("status", [401, 403])
def test_auth_status_raises_dlpautherror(monkeypatch, status):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({}, status=status))
    with pytest.raises(DLPAuthError) as ei:
        dlp.buscar_historico("SEGREDO-123")
    assert "SEGREDO-123" not in str(ei.value)


@pytest.mark.parametrize("exc", [requests.Timeout("t"), requests.ConnectionError("c")])
def test_network_errors_raise_unavailable(monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(dlp.requests, "get", boom)
    with pytest.raises(DLPUnavailableError):
        dlp.buscar_historico("t")


def test_5xx_raises_unavailable(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({}, status=503))
    with pytest.raises(DLPUnavailableError):
        dlp.buscar_historico("t")


def test_invalid_json_raises_unavailable(monkeypatch):
    class Bad(_Resp):
        def json(self):
            raise ValueError("no json")

    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: Bad({}))
    with pytest.raises(DLPUnavailableError):
        dlp.buscar_resumo_carteira("t")


def test_other_4xx_raises_plain_dlperror(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({}, status=404))
    with pytest.raises(DLPError) as ei:
        dlp.buscar_historico("t")
    assert type(ei.value) is DLPError


_OK = {"historico": [{"data": "2024-01-02", "vlr_mercado": 1}]}


def test_invalidate_user_cache_forces_refetch(monkeypatch):
    calls = []
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: calls.append(1) or _Resp(_OK))
    dlp.buscar_historico("A")
    dlp.buscar_historico("A")
    assert len(calls) == 1
    dlp.invalidate_user_cache("A")
    dlp.buscar_historico("A")
    assert len(calls) == 2


def test_invalidate_is_per_user(monkeypatch):
    calls = []
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: calls.append(1) or _Resp(_OK))
    dlp.buscar_historico("A")
    dlp.buscar_historico("B")
    assert len(calls) == 2
    dlp.invalidate_user_cache("A")
    dlp.buscar_historico("B")
    assert len(calls) == 2
    dlp.buscar_historico("A")
    assert len(calls) == 3


def test_concurrent_calls_hit_dlp_once(monkeypatch):
    calls = []

    def slow_get(*a, **k):
        calls.append(1)
        time.sleep(0.3)
        return _Resp(_OK)

    monkeypatch.setattr(dlp.requests, "get", slow_get)
    threads = [threading.Thread(target=dlp.buscar_historico, args=("A",)) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1


def test_get_uses_settings_base_url(monkeypatch, settings):
    settings.DLP_BASE_URL = "https://dlp.exemplo.test"
    seen = []
    monkeypatch.setattr(
        dlp.requests, "get",
        lambda url, **k: seen.append(url) or _Resp({"historico": [{"data": "2024-01-02"}]}),
    )
    dlp.buscar_historico("tok")
    assert seen == ["https://dlp.exemplo.test/historico"]


def test_default_base_url_is_dlpinvest():
    from django.conf import settings as dj

    assert dj.DLP_BASE_URL == "https://users.dlpinvest.com.br"


_ROWS = [
    {"data": "2024-01-02", "ativo": "PETR4", "classe": "ACAO", "corretora": "XP", "vlr_mercado": 100},
    {"data": "2024-01-02", "ativo": "HGLG11", "classe": "FII", "corretora": "XP", "vlr_mercado": 50},
    {"data": "2024-01-02", "ativo": "PETR4", "classe": "ACAO", "corretora": "BTG", "vlr_mercado": 10},
]


def test_filtered_calls_share_one_http_request(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dlp.requests, "get",
        lambda url, headers=None, params=None, timeout=None: calls.append(params) or _Resp({"historico": _ROWS}),
    )
    full = dlp.buscar_historico("tok")
    acoes = dlp.buscar_historico("tok", classe="ACAO")
    petr_xp = dlp.buscar_historico("tok", ativo="PETR4", corretora="XP")
    assert calls == [{}]
    assert len(full) == 3
    assert set(acoes["ativo"]) == {"PETR4"} and len(acoes) == 2
    assert len(petr_xp) == 1 and petr_xp["vlr_mercado"].iloc[0] == 100
    assert "date" in acoes.columns


def test_filter_without_match_returns_none(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({"historico": _ROWS}))
    assert dlp.buscar_historico("tok", classe="CRIPTO") is None


def test_filtered_first_call_still_caches_full_frame(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dlp.requests, "get",
        lambda url, params=None, **k: calls.append(params) or _Resp({"historico": _ROWS}),
    )
    dlp.buscar_historico("tok", ativo="PETR4")
    assert len(dlp.buscar_historico("tok")) == 3
    assert calls == [{}]


def test_invalidate_drops_single_key(monkeypatch):
    calls = []
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: calls.append(1) or _Resp({"historico": _ROWS}))
    dlp.buscar_historico("tok", classe="ACAO")
    dlp.invalidate_user_cache("tok")
    dlp.buscar_historico("tok", classe="FII")
    assert len(calls) == 2


def test_filtered_result_mutation_does_not_leak(monkeypatch):
    monkeypatch.setattr(dlp.requests, "get", lambda *a, **k: _Resp({"historico": _ROWS}))
    df = dlp.buscar_historico("tok", classe="ACAO")
    df["vlr_mercado"] = 0
    assert dlp.buscar_historico("tok", classe="ACAO")["vlr_mercado"].tolist() == [100, 10]


def test_filter_strips_and_blank_means_no_filter():
    df = pd.DataFrame(
        _ROWS + [{"data": "2024-01-03", "ativo": None, "classe": "ACAO", "corretora": "XP", "vlr_mercado": 1}]
    )
    assert len(dlp._filter_historico(df, classe=" ACAO ")) == 3
    assert len(dlp._filter_historico(df, ativo="", classe="   ")) == 4
    assert dlp._filter_historico(df, ativo="nan") is None  # NaN/None nunca casa
    assert dlp._filter_historico(df, ativo="None") is None


def test_filter_column_lookup_is_case_insensitive():
    df = pd.DataFrame([{"Ativo": "PETR4"}, {"Ativo": "VALE3"}])
    assert dlp._filter_historico(df, ativo="VALE3")["Ativo"].tolist() == ["VALE3"]


def test_filter_missing_column_is_ignored_and_logged(caplog):
    df = pd.DataFrame([{"ativo": "PETR4"}])
    out = dlp._filter_historico(df, corretora="XP")
    assert len(out) == 1
    assert "corretora" in caplog.text
