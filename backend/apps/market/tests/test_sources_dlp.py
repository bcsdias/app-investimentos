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
