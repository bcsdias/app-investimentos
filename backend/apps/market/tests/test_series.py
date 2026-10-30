import datetime as dt

import pandas as pd
import pytest

from apps.market import series as sr
from apps.market.services.sources.base import RawPoint

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
    gen = mc.series_generation()
    assert mc.get_df(mc.make_key("series", f"g{gen}", "S&P 500", S, E)) is not None
    assert mc.get_df(mc.make_key("series", f"g{gen}", "S&P 500", dt.date(2023, 1, 1), E)) is None


def test_write_invalidates_cached_series(monkeypatch):
    # 1ª leitura cacheia [100, 110]; upsert de um ponto (repository real, db) e load passa a
    # devolver [100, 110, 120] -> 2ª leitura devolve 3 pontos
    sr.repository.upsert_points(
        "yf:^GSPC", "YF", "level",
        [RawPoint(dt.date(2024, 1, 2), 100.0), RawPoint(dt.date(2024, 1, 3), 110.0)],
    )
    s1 = sr.get_series("S&P 500", S, E)
    assert len(s1) == 2
    assert list(s1.values) == [100.0, 110.0]

    sr.repository.upsert_points(
        "yf:^GSPC", "YF", "level",
        [RawPoint(dt.date(2024, 1, 4), 120.0)],
    )
    s2 = sr.get_series("S&P 500", S, E)
    assert len(s2) == 3
    assert list(s2.values) == [100.0, 110.0, 120.0]


def test_write_to_base_invalidates_derived_series(monkeypatch):
    # "S&P 500 BRL" cacheada; upsert em "ptax:USDBRL" -> próxima leitura recalcula (load chamado de novo)
    sr.repository.upsert_points(
        "yf:^GSPC", "YF", "level",
        [RawPoint(dt.date(2024, 1, 2), 100.0)],
    )
    sr.repository.upsert_points(
        "ptax:USDBRL", "PTAX", "level",
        [RawPoint(dt.date(2024, 1, 2), 5.0)],
    )
    s1 = sr.get_series("S&P 500 BRL", S, E)
    assert round(s1.iloc[0], 2) == 500.0

    sr.repository.upsert_points(
        "ptax:USDBRL", "PTAX", "level",
        [RawPoint(dt.date(2024, 1, 2), 5.5)],
    )
    s2 = sr.get_series("S&P 500 BRL", S, E)
    assert round(s2.iloc[0], 2) == 550.0


def test_derived_series_survive_cache_hit(monkeypatch):
    # ToBRL produz séries sem nome; a 2ª leitura (do cache) não
    # pode virar NaN (achado da paridade da Fase 6).
    def fake_load(key, s, e):
        if key == "yf:^GSPC":
            return pd.Series([100.0, 110.0], index=_bdays(2), name=key)
        if key == "ptax:USDBRL":
            return pd.Series([5.0, 5.5], index=_bdays(2), name=key)
        return pd.Series(dtype=float, name=key)

    monkeypatch.setattr(sr.repository, "load_series", fake_load)
    for name in ("S&P 500 BRL",):
        fresh = sr.get_series(name, S, E)
        cached = sr.get_series(name, S, E)
        assert not cached.isna().any(), name
        assert cached.tolist() == pytest.approx(fresh.tolist(), rel=1e-9), name


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


# --- tickers sob demanda (Task 6) ---------------------------------------------
from apps.market import catalog, repository  # noqa: E402
from apps.market.services.sources.base import FetchError, RawPoint  # noqa: E402

TODAY = dt.date(2024, 6, 30)


@pytest.fixture
def ondemand(monkeypatch, locmem_cache, assets_csv):
    monkeypatch.setattr(sr, "_today", lambda: TODAY)
    calls = []

    class Box:
        pass

    box = Box()
    box.calls = calls
    box.result = lambda t, s, e: []

    def fake(t, s, e):
        calls.append((t, s, e))
        r = box.result(t, s, e)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(sr.yahoo, "fetch_yahoo", fake)
    return box


def _pts(start, n, base=100.0):
    return [RawPoint(d.date(), base + i) for i, d in enumerate(pd.bdate_range(start, periods=n))]


def _store(key, start, n, source="YF", base=100.0):
    repository.upsert_points(key, source, "level", _pts(start, n, base))


def test_first_use_fetches_full_history(ondemand):
    ondemand.result = lambda t, s, e: _pts("2024-01-02", 5)
    s = sr.get_series("AAPL", dt.date(2024, 1, 1), TODAY)
    assert ondemand.calls == [("AAPL", catalog.EPOCH, TODAY)]
    assert repository.last_date("yf:AAPL") is not None and not s.empty


def test_fresh_data_does_not_refetch(ondemand):
    repository.upsert_points("yf:AAPL", "YF", "level", [RawPoint(TODAY - dt.timedelta(days=1), 10.0)])
    s = sr.get_series("AAPL", dt.date(2024, 1, 1), TODAY)
    assert ondemand.calls == []
    assert not s.empty


def test_stale_data_tops_up_from_last_date(ondemand):
    last = TODAY - dt.timedelta(days=10)
    repository.upsert_points("yf:AAPL", "YF", "level", [RawPoint(last, 10.0)])
    ondemand.result = lambda t, s, e: [RawPoint(TODAY - dt.timedelta(days=1), 11.0)]
    sr.get_series("AAPL", dt.date(2024, 1, 1), TODAY)
    assert ondemand.calls == [("AAPL", last + dt.timedelta(days=1), TODAY)]
    assert repository.last_date("yf:AAPL") == TODAY - dt.timedelta(days=1)


def test_fetch_error_falls_back_to_stored(ondemand):
    last = TODAY - dt.timedelta(days=10)
    repository.upsert_points("yf:AAPL", "YF", "level", [RawPoint(last, 10.0)])
    ondemand.result = lambda t, s, e: FetchError("fora do ar")
    s = sr.get_series("AAPL", dt.date(2024, 1, 1), TODAY)
    assert list(s.values) == [10.0]


def test_fetch_error_without_data_returns_none(ondemand):
    ondemand.result = lambda t, s, e: FetchError("fora do ar")
    assert sr.get_series("AAPL", dt.date(2024, 1, 1), TODAY) is None


def test_brl_variant_uses_ptax(ondemand):
    _store("yf:AAPL", "2024-06-03", 5, base=100.0)
    _store("ptax:USDBRL", "2024-06-03", 5, source="PTAX", base=5.0)
    s = sr.get_series("AAPL BRL", dt.date(2024, 6, 1), TODAY)
    expected = (
        repository.load_series("yf:AAPL", dt.date(2024, 6, 1), TODAY)
        * repository.load_series("ptax:USDBRL", dt.date(2024, 6, 1), TODAY)
    )
    assert list(s.values) == pytest.approx(list(expected.values))


def test_catalog_name_wins_over_asset(ondemand):
    _store("yf:IVVB11.SA", "2024-06-03", 3)
    s = sr.get_series("IVVB11", dt.date(2024, 6, 1), TODAY)
    assert ondemand.calls == []
    assert not s.empty


def test_unknown_name_raises_keyerror(ondemand):
    with pytest.raises(KeyError):
        sr.get_series("NAOEXISTE", dt.date(2024, 1, 1), TODAY)
