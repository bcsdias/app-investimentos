import pandas as pd
import pytest

from apps.market.services.sources import dlp
from apps.market.services.sources.base import DLPError


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

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


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
