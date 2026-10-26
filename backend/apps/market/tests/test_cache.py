import threading
import time

import numpy as np
import pandas as pd
import pytest
from django.core.cache import cache as django_cache

from apps.market import cache as mc


@pytest.fixture(autouse=True)
def _locmem_cache(settings):
    settings.CACHES = {
        "default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}
    }
    from django.core.cache import cache

    cache.clear()


def test_make_key_namespaced_and_windowed():
    assert mc.make_key("series", "IMA-B", "2020-01-01", "2024-12-31") == (
        "market:series:IMA-B:2020-01-01:2024-12-31"
    )


def test_series_roundtrip_returns_series_not_frame():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    s = pd.Series([1.5, 2.5, 3.5], index=idx, name="CDI")
    mc.set_df("market:t:s", s, ttl=60)
    out = mc.get_df("market:t:s")
    assert isinstance(out, pd.Series)
    pd.testing.assert_series_equal(out, s, check_names=False)


def test_dataframe_roundtrip():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=idx)
    mc.set_df("market:t:df", df, ttl=60)
    out = mc.get_df("market:t:df")
    assert isinstance(out, pd.DataFrame)
    pd.testing.assert_frame_equal(out, df)


def test_plain_value_roundtrip():
    mc.set_df("market:t:plain", {"x": 1, "y": [1, 2, 3]}, ttl=60)
    assert mc.get_df("market:t:plain") == {"x": 1, "y": [1, 2, 3]}


def test_miss_returns_none():
    assert mc.get_df("market:t:absent") is None


def test_set_failure_is_swallowed(monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("redis down")

    from django.core.cache import cache

    monkeypatch.setattr(cache, "set", boom)
    mc.set_df("market:t:s", pd.Series([1.0]), ttl=60)  # não levanta
    assert "redis down" in caplog.text or "cache" in caplog.text.lower()


def test_multiindex_columns_are_flattened():
    idx = pd.to_datetime(["2024-01-02"])
    df = pd.DataFrame([[1.0]], index=idx)
    df.columns = pd.MultiIndex.from_tuples([("Close", "PETR4")])
    mc.set_df("market:t:mi", df, ttl=60)
    out = mc.get_df("market:t:mi")
    assert list(out.columns) == ["Close_PETR4"]


def test_single_flight_blocks_second_holder():
    order = []

    def worker(tag):
        with mc.single_flight("k", timeout=5, poll=0.01):
            order.append(f"{tag}-in")
            time.sleep(0.1)
            order.append(f"{tag}-out")

    ts = [threading.Thread(target=worker, args=(t,)) for t in "ab"]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert order[1].endswith("-out")  # sem intercalação


def test_single_flight_times_out_and_proceeds():
    django_cache.add("k:lock", 1, 30)  # lock "preso"
    t0 = time.monotonic()
    with mc.single_flight("k", timeout=0.2, poll=0.01):
        pass
    assert time.monotonic() - t0 < 1
