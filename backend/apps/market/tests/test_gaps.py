import datetime as dt
import pandas as pd
from apps.market.gaps import Gap, find_gaps, infer_periodicity, stale_tail

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


def test_stale_tail_beyond_tolerance():
    assert stale_tail([D(2026, 4, 1), D(2026, 4, 10)], "daily", D(2026, 10, 29)) == Gap(
        D(2026, 4, 10), D(2026, 10, 29)
    )


def test_stale_tail_within_tolerance_is_none():
    assert stale_tail([D(2026, 10, 20)], "daily", D(2026, 10, 29)) is None


def test_stale_tail_empty_is_none():
    assert stale_tail([], "monthly", D(2026, 10, 29)) is None



def test_stale_tail_monthly_tolerates_publication_lag():
    # IPCA de setembro é datado em 01/09 e só sai por volta de 10/10; o de outubro, por
    # volta de 10/11. Em 31/10 o último ponto tem 60 dias e isso é o atraso normal.
    assert stale_tail([D(2026, 8, 1), D(2026, 9, 1)], "monthly", D(2026, 10, 31)) is None
    assert stale_tail([D(2026, 8, 1), D(2026, 9, 1)], "monthly", D(2026, 11, 10)) is None


def test_stale_tail_monthly_beyond_publication_lag():
    # Dois meses sem publicação: aí é série parada.
    assert stale_tail([D(2026, 7, 1)], "monthly", D(2026, 10, 31)) == Gap(D(2026, 7, 1), D(2026, 10, 31))
