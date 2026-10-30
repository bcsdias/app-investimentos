import math

import numpy as np
import pandas as pd

from parity.compare import (
    DESVIO,
    FALHA,
    INVESTIGAR,
    OK,
    REL_TOL_L1,
    SeriesDiff,
    apply_adr0012_recheck,
    classify_origin,
    compare_frames,
)

DATES = pd.bdate_range("2024-01-01", periods=5)


def _frame(**cols):
    return pd.DataFrame(cols, index=DATES[: len(next(iter(cols.values())))])


def _by_col(diffs):
    return {d.column: d for d in diffs}


def test_identical_frames_are_ok():
    df = _frame(A=[1.0, 2.0, 3.0, 4.0, 5.0], B=[10.0, 11.0, 12.0, 13.0, 14.0])
    diffs = compare_frames("twr", df, df.copy(), REL_TOL_L1, FALHA)
    assert len(diffs) == 2
    for d in diffs:
        assert d.status == OK
        assert d.max_rel == 0.0
        assert d.first_date is None
        assert d.n_out == 0


def test_relative_diff_above_tol_fails():
    leg = _frame(A=[100.0] * 5)
    new = leg.copy()
    new.loc[DATES[2], "A"] = 100.0002
    (d,) = compare_frames("twr", leg, new, REL_TOL_L1, FALHA)
    assert d.status == FALHA
    assert d.n_out == 1
    assert d.first_date == "2024-01-03"
    assert math.isclose(d.max_rel, 2e-6, rel_tol=1e-3)


def test_relative_diff_below_tol_is_ok():
    leg = _frame(A=[100.0] * 5)
    new = leg.copy()
    new.loc[DATES[2], "A"] = 100.00005
    (d,) = compare_frames("twr", leg, new, REL_TOL_L1, FALHA)
    assert d.status == OK
    assert d.max_rel > 0


def test_abs_floor_for_values_near_zero():
    leg = _frame(A=[0.0])
    (ok,) = compare_frames("drawdown", leg, _frame(A=[5e-10]), REL_TOL_L1, FALHA)
    assert ok.status == OK
    (bad,) = compare_frames("drawdown", leg, _frame(A=[5e-9]), REL_TOL_L1, FALHA)
    assert bad.status == FALHA


def test_nan_matches_nan_only():
    (ok,) = compare_frames("twr", _frame(A=[np.nan, 1.0]), _frame(A=[np.nan, 1.0]), REL_TOL_L1, FALHA)
    assert ok.status == OK
    (bad,) = compare_frames("twr", _frame(A=[np.nan, 1.0]), _frame(A=[1.0, 1.0]), REL_TOL_L1, FALHA)
    assert bad.status == FALHA
    assert bad.n_out == 1
    assert bad.first_date == "2024-01-01"


def test_index_mismatch_counts_as_divergence():
    leg = _frame(A=[1.0, 2.0, 3.0, 4.0, 5.0])
    new = leg.drop(index=pd.Timestamp("2024-01-04"))
    (d,) = compare_frames("twr", leg, new, REL_TOL_L1, FALHA)
    assert d.only_legacy == 1
    assert d.only_new == 0
    assert d.n_out == 1
    assert d.first_date == "2024-01-04"
    assert d.status == FALHA


def test_column_only_on_one_side_gets_fail_status_and_note():
    leg = _frame(A=[1.0, 2.0, 3.0])
    new = _frame(A=[1.0, 2.0, 3.0], IFIX=[5.0, 6.0, 7.0])
    diffs = _by_col(compare_frames("twr", leg, new, 1e-4, INVESTIGAR))
    assert diffs["A"].status == OK
    ifix = diffs["IFIX"]
    assert ifix.status == INVESTIGAR
    assert ifix.note == "só no novo"
    assert math.isnan(ifix.max_rel)
    assert ifix.n_out == 3

    diffs = _by_col(compare_frames("twr", new, leg, 1e-4, INVESTIGAR))
    assert diffs["IFIX"].note == "só no legado"


def test_non_date_index_is_compared_by_label():
    leg = pd.DataFrame({"Sharpe": [0.5, 0.1]}, index=["Carteira", "CDI"])
    new = pd.DataFrame({"Sharpe": [0.1, 0.6]}, index=["CDI", "Carteira"])
    (d,) = compare_frames("risk_return", leg, new, REL_TOL_L1, FALHA)
    assert d.status == FALHA
    assert d.n_out == 1
    assert d.first_date == "Carteira"


def _diff(metric, column, status=FALHA, only_legacy=0, only_new=0, max_rel=0.1, n_out=1):
    return SeriesDiff(
        metric=metric,
        column=column,
        status=status,
        max_rel=max_rel,
        n_out=n_out,
        first_date="2024-01-01",
        only_legacy=only_legacy,
        only_new=only_new,
    )


def test_classify_origin_rules():
    assert classify_origin(_diff("twr", "S&P 500 BRL", only_new=1)) == "calendário/datas"
    assert classify_origin(_diff("twr", "S&P 500 BRL")) == "conversão de câmbio"
    assert classify_origin(_diff("kpis", "valor")) == "cálculo"
    assert classify_origin(_diff("twr", "Carteira")) == "cálculo"
    assert classify_origin(_diff("shadow", "Total Investido")) == "cálculo"
    assert classify_origin(_diff("twr", "CDI")) == "revisão de cotação na fonte"


def test_adr0012_recheck_marks_known_deviation():
    new = {"rolling_vol": _frame(A=[1.0, 2.0])}
    diffs = [
        _diff("rolling_vol", "A", max_rel=3e-3, n_out=2),
        _diff("twr", "A", status=OK, max_rel=0.0, n_out=0),
    ]
    out = apply_adr0012_recheck(diffs, {"rolling_vol": _frame(A=[1.0, 2.0])}, new)
    by = {(d.metric, d.column): d for d in out}
    vol = by[("rolling_vol", "A")]
    assert vol.status == DESVIO
    assert vol.max_rel == 3e-3
    assert vol.n_out == 2
    assert "fill_method=None" in vol.note
    assert by[("twr", "A")].status == OK


def test_adr0012_recheck_keeps_failure_when_other_metric_fails():
    new = {"rolling_vol": _frame(A=[1.0, 2.0])}
    diffs = [_diff("rolling_vol", "A"), _diff("twr", "A")]
    out = apply_adr0012_recheck(diffs, {"rolling_vol": _frame(A=[1.0, 2.0])}, new)
    assert out == diffs


def test_adr0012_recheck_keeps_failure_when_recheck_still_differs():
    new = {"rolling_vol": _frame(A=[1.0, 2.0])}
    diffs = [_diff("rolling_vol", "A")]
    out = apply_adr0012_recheck(diffs, {"rolling_vol": _frame(A=[1.0, 2.5])}, new)
    assert out == diffs
