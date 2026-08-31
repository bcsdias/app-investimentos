import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from apps.core.models import MarketSeries
from apps.market import repository as repo

pytestmark = pytest.mark.django_db


@dataclass(frozen=True)
class P:
    date: dt.date
    value: float


def test_upsert_inserts_and_counts():
    n = repo.upsert_points(
        "bcb:12", "BCB", "rate",
        [P(dt.date(2024, 1, 2), 0.0437), P(dt.date(2024, 1, 3), 0.0438)],
    )
    assert n == 2
    assert MarketSeries.objects.filter(series_key="bcb:12").count() == 2
    row = MarketSeries.objects.get(series_key="bcb:12", reference_date=dt.date(2024, 1, 2))
    assert row.kind == "rate"
    assert str(row.value) == "0.04370000"


def test_upsert_is_idempotent_and_updates_value():
    repo.upsert_points("bcb:12", "BCB", "rate", [P(dt.date(2024, 1, 2), 0.0437)])
    repo.upsert_points("bcb:12", "BCB", "rate", [P(dt.date(2024, 1, 2), 0.0999)])
    qs = MarketSeries.objects.filter(series_key="bcb:12", reference_date=dt.date(2024, 1, 2))
    assert qs.count() == 1
    assert str(qs.get().value) == "0.09990000"


def test_upsert_dedupes_same_date_in_one_batch_keeping_last():
    # bulk_create(update_conflicts=True) levanta "cannot affect row a second time"
    # se o mesmo reference_date aparecer duas vezes no lote (ex.: CSV com dia
    # repetido, PTAX com duas cotações no dia). O último ponto vence.
    n = repo.upsert_points(
        "ptax:USDBRL", "PTAX", "level",
        [P(dt.date(2024, 1, 2), 5.40), P(dt.date(2024, 1, 2), 5.42)],
    )
    assert n == 1
    qs = MarketSeries.objects.filter(series_key="ptax:USDBRL", reference_date=dt.date(2024, 1, 2))
    assert qs.count() == 1
    assert str(qs.get().value) == "5.42000000"


def test_upsert_drops_nan_and_inf_with_log(caplog):
    n = repo.upsert_points(
        "yf:^GSPC", "YF", "level",
        [P(dt.date(2024, 1, 2), 100.0), P(dt.date(2024, 1, 3), float("nan")),
         P(dt.date(2024, 1, 4), np.inf)],
    )
    assert n == 1
    assert MarketSeries.objects.filter(series_key="yf:^GSPC").count() == 1
    assert "descart" in caplog.text.lower()


def test_load_series_returns_sorted_float_series():
    repo.upsert_points(
        "bcb:12", "BCB", "rate",
        [P(dt.date(2024, 1, 4), 3.0), P(dt.date(2024, 1, 2), 1.0), P(dt.date(2024, 1, 3), 2.0)],
    )
    s = repo.load_series("bcb:12", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert isinstance(s, pd.Series)
    assert s.dtype == float
    assert list(s.values) == [1.0, 2.0, 3.0]
    assert s.index.is_monotonic_increasing
    assert s.name == "bcb:12"


def test_load_series_respects_window():
    repo.upsert_points(
        "bcb:12", "BCB", "rate",
        [P(dt.date(2023, 12, 31), 9.0), P(dt.date(2024, 1, 2), 1.0), P(dt.date(2024, 2, 1), 5.0)],
    )
    s = repo.load_series("bcb:12", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert list(s.values) == [1.0]


def test_load_series_empty_is_series_not_none():
    s = repo.load_series("bcb:999", dt.date(2024, 1, 1), dt.date(2024, 1, 31))
    assert isinstance(s, pd.Series)
    assert s.empty


def test_last_date_and_available_keys():
    repo.upsert_points("bcb:12", "BCB", "rate", [P(dt.date(2024, 1, 2), 1.0), P(dt.date(2024, 1, 9), 2.0)])
    repo.upsert_points("yf:^GSPC", "YF", "level", [P(dt.date(2024, 1, 5), 100.0)])
    assert repo.last_date("bcb:12") == dt.date(2024, 1, 9)
    assert repo.last_date("bcb:absent") is None
    assert repo.available_keys() == ["bcb:12", "yf:^GSPC"]
