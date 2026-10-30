import datetime as dt

import pandas as pd
import pytest

from apps.market.services.sources import yahoo
from apps.market.services.sources.base import FetchError, RawPoint


def test_fetch_yahoo_simple_columns(monkeypatch):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"Open": [1, 2], "Close": [459.1, 460.7]}, index=idx)
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: df)
    pts = yahoo.fetch_yahoo("^GSPC", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert pts == [
        RawPoint(dt.date(2024, 1, 2), 459.1),
        RawPoint(dt.date(2024, 1, 3), 460.7),
    ]


def test_fetch_yahoo_multiindex_columns(monkeypatch):
    idx = pd.to_datetime(["2024-01-02"])
    df = pd.DataFrame([[1.0, 459.1]], index=idx)
    df.columns = pd.MultiIndex.from_tuples([("Open", "^GSPC"), ("Close", "^GSPC")])
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: df)
    pts = yahoo.fetch_yahoo("^GSPC", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert pts == [RawPoint(dt.date(2024, 1, 2), 459.1)]


def test_fetch_yahoo_empty_raises_fetcherror(monkeypatch):
    # yf.download engole erro de rede e devolve DataFrame vazio sem levantar;
    # tratar isso como 'sem dados' mascararia a falha (contrato base.py).
    monkeypatch.setattr(yahoo.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(FetchError):
        yahoo.fetch_yahoo("^GSPC", dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_fetch_yahoo_raises_on_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("yf down")

    monkeypatch.setattr(yahoo.yf, "download", boom)
    with pytest.raises(FetchError):
        yahoo.fetch_yahoo("^GSPC", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
