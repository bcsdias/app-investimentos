import datetime as dt

import pandas as pd
import pytest

from apps.market.services.sources import bcb
from apps.market.services.sources.base import FetchError, RawPoint


def test_fetch_sgs_returns_raw_points(monkeypatch):
    def fake_get(codes, start, end):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return pd.DataFrame({"12": [0.0437, 0.0438]}, index=idx)

    monkeypatch.setattr(bcb.sgs, "get", fake_get)
    pts = bcb.fetch_sgs(12, dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert pts == [
        RawPoint(dt.date(2024, 1, 2), 0.0437),
        RawPoint(dt.date(2024, 1, 3), 0.0438),
    ]


def test_fetch_sgs_chunks_five_year_windows(monkeypatch):
    calls = []

    def fake_get(codes, start, end):
        calls.append((pd.Timestamp(start).date(), pd.Timestamp(end).date()))
        return pd.DataFrame({"12": [1.0]}, index=pd.to_datetime([start]))

    monkeypatch.setattr(bcb.sgs, "get", fake_get)
    bcb.fetch_sgs(12, dt.date(2010, 1, 1), dt.date(2024, 1, 1))
    assert len(calls) >= 3  # 2010-15, 2015-20, 2020-24
    assert calls[0][0] == dt.date(2010, 1, 1)


def test_fetch_sgs_raises_fetcherror_on_api_failure(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("bcb down")

    monkeypatch.setattr(bcb.sgs, "get", boom)
    with pytest.raises(FetchError):
        bcb.fetch_sgs(12, dt.date(2024, 1, 1), dt.date(2024, 1, 31))


def test_fetch_sgs_empty_response_returns_empty_list(monkeypatch):
    monkeypatch.setattr(bcb.sgs, "get", lambda *a, **k: pd.DataFrame())
    assert bcb.fetch_sgs(12, dt.date(2024, 1, 1), dt.date(2024, 1, 31)) == []
