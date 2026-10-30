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


def _ref_to_date(val):
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    return pd.to_datetime(val).date()


def _ref_find_gaps(dates, periodicity):
    from apps.market.gaps import TOLERANCE_DAYS
    tolerance = TOLERANCE_DAYS[periodicity]
    clean_dates = sorted({_ref_to_date(d) for d in dates})
    if len(clean_dates) < 2:
        return []
    gaps = []
    for d1, d2 in zip(clean_dates[:-1], clean_dates[1:]):
        if (d2 - d1).days > tolerance:
            gaps.append(Gap(start=d1, end=d2))
    return gaps


def _ref_stale_tail(dates, periodicity, until):
    from apps.market.gaps import TAIL_TOLERANCE_DAYS
    clean_dates = sorted({_ref_to_date(d) for d in dates})
    if not clean_dates:
        return None
    last = clean_dates[-1]
    until_d = _ref_to_date(until)
    tolerance = TAIL_TOLERANCE_DAYS[periodicity]
    if (until_d - last).days > tolerance:
        return Gap(start=last, end=until_d)
    return None


def test_vectorized_gaps_match_reference():
    until = D(2024, 6, 30)

    # Entradas de teste:
    # 1. DatetimeIndex diário com horas diferentes de zero e duplicatas
    idx_hours = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 15:00", "2024-01-20 08:00", "2024-01-21 00:00"])
    # 2. lista de dt.date fora de ordem
    dates_unsorted = [D(2024, 2, 1), D(2024, 1, 1), D(2024, 3, 1), D(2024, 1, 15)]
    # 3. pd.Timestamp misturado com dt.date
    mixed = [pd.Timestamp("2024-01-01"), D(2024, 1, 20), pd.Timestamp("2024-02-15")]
    # 4. mensal com buraco de 60 dias
    monthly_gap = [D(2024, 1, 1), D(2024, 2, 1), D(2024, 5, 1)]
    # 5. vazio
    empty = []
    # 6. um elemento só
    single = [D(2024, 1, 1)]

    test_inputs = [idx_hours, dates_unsorted, mixed, monthly_gap, empty, single]

    for inp in test_inputs:
        for periodicity in ("daily", "monthly"):
            assert find_gaps(inp, periodicity) == _ref_find_gaps(inp, periodicity)
            assert stale_tail(inp, periodicity, until) == _ref_stale_tail(inp, periodicity, until)


def test_find_gaps_does_not_call_to_date_per_element(monkeypatch):
    import apps.market.gaps as gaps

    calls = []
    real_to_date = gaps._to_date

    def counting_to_date(val):
        calls.append(val)
        return real_to_date(val)

    monkeypatch.setattr(gaps, "_to_date", counting_to_date)

    dates = pd.date_range("2020-01-01", periods=1000, freq="D")

    # find_gaps não deve chamar _to_date por elemento (0 chamadas)
    calls.clear()
    gaps.find_gaps(dates, "daily")
    assert len(calls) == 0

    # stale_tail não deve chamar _to_date por elemento (no máximo 1 chamada para o until)
    calls.clear()
    gaps.stale_tail(dates, "daily", D(2024, 1, 1))
    assert len(calls) <= 1

