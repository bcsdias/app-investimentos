import datetime as dt

import pytest

from apps.market.services.sources import ptax
from apps.market.services.sources.base import FetchError, RawPoint


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")


def test_fetch_ptax_returns_daily_last_quote(monkeypatch):
    payload = {
        "value": [
            {"cotacaoVenda": 5.40, "dataHoraCotacao": "2024-01-02 10:00:00.0"},
            {"cotacaoVenda": 5.42, "dataHoraCotacao": "2024-01-02 13:00:00.0"},
            {"cotacaoVenda": 5.50, "dataHoraCotacao": "2024-01-03 13:00:00.0"},
        ]
    }
    monkeypatch.setattr(ptax.requests, "get", lambda *a, **k: _Resp(payload))
    pts = ptax.fetch_ptax(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert pts == [
        RawPoint(dt.date(2024, 1, 2), 5.42),
        RawPoint(dt.date(2024, 1, 3), 5.50),
    ]


def test_fetch_ptax_empty_returns_empty_list(monkeypatch):
    monkeypatch.setattr(ptax.requests, "get", lambda *a, **k: _Resp({"value": []}))
    assert ptax.fetch_ptax(dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == []


def test_fetch_ptax_raises_on_http_failure(monkeypatch):
    monkeypatch.setattr(ptax.requests, "get", lambda *a, **k: _Resp({}, status=500))
    with pytest.raises(FetchError):
        ptax.fetch_ptax(dt.date(2024, 1, 1), dt.date(2024, 1, 31))
