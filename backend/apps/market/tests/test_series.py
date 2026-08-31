import datetime as dt

import pandas as pd
import pytest

from apps.market import series as sr

pytestmark = pytest.mark.django_db

S, E = dt.date(2024, 1, 1), dt.date(2024, 12, 31)


@pytest.fixture(autouse=True)
def _locmem_cache(settings):
    settings.CACHES = {
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}
    }
    from django.core.cache import cache

    cache.clear()


def _bdays(n, start="2024-01-02"):
    return pd.bdate_range(start=start, periods=n)


def test_get_series_rate_applies_accumulate(monkeypatch):
    raw = pd.Series([1.0, 1.0, 1.0], index=_bdays(3), name="bcb:12")
    monkeypatch.setattr(sr.repository, "load_series", lambda key, s, e: raw)
    out = sr.get_series("CDI", S, E)
    # accumulate: 1.01, 1.0201, 1.030301
    assert round(out.iloc[-1], 6) == 1.030301


def test_get_series_level_passthrough(monkeypatch):
    raw = pd.Series([100.0, 110.0], index=_bdays(2), name="yf:^GSPC")
    monkeypatch.setattr(sr.repository, "load_series", lambda key, s, e: raw)
    out = sr.get_series("S&P 500", S, E)
    assert list(out.values) == [100.0, 110.0]


def test_get_series_caches_second_call(monkeypatch):
    calls = []
    raw = pd.Series([100.0, 110.0], index=_bdays(2), name="yf:^GSPC")

    def fake_load(key, s, e):
        calls.append(key)
        return raw

    monkeypatch.setattr(sr.repository, "load_series", fake_load)
    sr.get_series("S&P 500", S, E)
    sr.get_series("S&P 500", S, E)
    assert len(calls) == 1  # segundo veio do cache


def test_cache_key_includes_window(monkeypatch):
    raw = pd.Series([100.0, 110.0], index=_bdays(2), name="yf:^GSPC")
    monkeypatch.setattr(sr.repository, "load_series", lambda key, s, e: raw)
    from apps.market import cache as mc

    sr.get_series("S&P 500", S, E)
    assert mc.get_df(mc.make_key("series", "S&P 500", S, E)) is not None
    assert mc.get_df(mc.make_key("series", "S&P 500", dt.date(2023, 1, 1), E)) is None


def test_get_series_proxy_splice(monkeypatch):
    def fake_load(key, s, e):
        if key == "bcb:12466":
            return pd.Series([100.0, 110.0], index=_bdays(2, "2024-01-02"), name=key)
        if key == "yf:IMAB11.SA":
            return pd.Series([50.0, 55.0, 60.5], index=_bdays(3, "2024-01-03"), name=key)
        return pd.Series(dtype=float, name=key)

    monkeypatch.setattr(sr.repository, "load_series", fake_load)
    out = sr.get_series("IMA-B", S, E)
    assert out.loc["2024-01-03"] == 110.0
    assert round(out.loc["2024-01-04"], 4) == round(110.0 * 55.0 / 50.0, 4)


def test_get_series_empty_returns_none(monkeypatch):
    monkeypatch.setattr(sr.repository, "load_series", lambda key, s, e: pd.Series(dtype=float))
    assert sr.get_series("S&P 500", S, E) is None


def test_get_many_skips_missing(monkeypatch):
    def fake_load(key, s, e):
        if key == "bcb:12":
            return pd.Series([1.0, 1.0], index=_bdays(2), name=key)
        return pd.Series(dtype=float, name=key)

    monkeypatch.setattr(sr.repository, "load_series", fake_load)
    out = sr.get_many(["CDI", "S&P 500"], S, E)
    assert set(out) == {"CDI"}
