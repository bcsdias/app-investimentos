import numpy as np
import pandas as pd
import pytest

from apps.market import transforms as tf
from apps.market.synthetics import SyntheticSpec


def _s(values, start="2024-01-02"):
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_accumulate_cumprod_from_rate():
    rate = _s([1.0, 1.0, 1.0])  # 1% ao dia
    out = tf.accumulate(rate)
    np.testing.assert_allclose(out.values, [1.01, 1.0201, 1.030301])


def test_accumulate_first_nan_becomes_one():
    rate = pd.Series([np.nan, 1.0], index=pd.bdate_range("2024-01-02", periods=2))
    out = tf.accumulate(rate)
    assert out.iloc[0] == 1.0


def test_splice_proxy_continuous_at_seam():
    bcb = _s([100.0, 110.0], start="2024-01-02")            # termina em 110 em 2024-01-03
    etf = _s([50.0, 55.0, 60.5], start="2024-01-03")         # base 50 no dia da emenda
    out = tf.splice_proxy(bcb, etf)
    # até 2024-01-03: valores do BCB. Depois: 110 * (etf / 50)
    assert out.loc["2024-01-03"] == 110.0
    np.testing.assert_allclose(out.loc["2024-01-04"], 110.0 * 55.0 / 50.0)
    np.testing.assert_allclose(out.loc["2024-01-05"], 110.0 * 60.5 / 50.0)
    assert not out.isna().any()


def test_to_brl_multiplies_by_ffilled_ptax():
    usd = _s([100.0, 100.0, 100.0])
    ptax = pd.Series([5.0], index=[usd.index[0]])
    out = tf.to_brl(usd, ptax)
    np.testing.assert_allclose(out.values, [500.0, 500.0, 500.0])


def test_synthetic_spread_matches_legacy_formula():
    base = _s([100.0, 101.0, 102.0])
    spec = SyntheticSpec("CDI", "+", 2.0)
    out = tf.synthetic_from_spec(spec, base)
    ret = base.pct_change(fill_method=None).fillna(0)
    expected = (1 + ((1 + ret) * (1 + 2.0 / 100) ** (1 / 252) - 1)).cumprod() * 100
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_synthetic_percent_of_index():
    base = _s([100.0, 110.0, 121.0])
    spec = SyntheticSpec("CDI", "%", 50.0)
    out = tf.synthetic_from_spec(spec, base)
    ret = base.pct_change(fill_method=None).fillna(0)
    expected = (1 + ret * 0.5).cumprod() * 100
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_composite_weighted_returns():
    a = _s([100.0, 110.0])   # +10%
    b = _s([100.0, 100.0])   # 0%
    out = tf.composite({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    np.testing.assert_allclose(out.values, [1.0, 1.05])


def test_rebase_100():
    s = _s([200.0, 220.0, 180.0])
    out = tf.rebase_100(s)
    np.testing.assert_allclose(out.values, [100.0, 110.0, 90.0])
