import datetime as dt
import pandas as pd
from apps.market.gaps import Gap, find_gaps, infer_periodicity

D = dt.date


def test_daily_gap_beyond_tolerance():
    assert find_gaps([D(2024, 1, 2), D(2024, 1, 3), D(2024, 1, 20)], "daily") == [
        Gap(D(2024, 1, 3), D(2024, 1, 20))
    ]


def test_holiday_cluster_is_not_a_gap():
    assert find_gaps([D(2024, 12, 23), D(2025, 1, 1)], "daily") == []  # 9 dias


def test_monthly_tolerance():
    assert find_gaps([D(2024, 1, 1), D(2024, 2, 1), D(2024, 4, 1)], "monthly") == [
        Gap(D(2024, 2, 1), D(2024, 4, 1))
    ]


def test_empty_and_single_point_have_no_gaps():
    assert find_gaps([], "daily") == [] and find_gaps([D(2024, 1, 2)], "daily") == []


def test_unsorted_duplicated_timestamps():
    idx = pd.to_datetime(["2024-01-20", "2024-01-02", "2024-01-02"])
    assert find_gaps(idx, "daily") == [Gap(D(2024, 1, 2), D(2024, 1, 20))]


def test_infer_periodicity():
    assert (
        infer_periodicity(pd.date_range("2024-01-01", periods=6, freq="MS"))
        == "monthly"
    )
    assert infer_periodicity(pd.bdate_range("2024-01-01", periods=6)) == "daily"
