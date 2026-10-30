import datetime as dt
import importlib

import django.apps
import pytest

from apps.core.models import MarketSeries
from apps.market import cache


@pytest.mark.django_db
def test_forwards_removes_only_bcb_ima_keys(monkeypatch):
    MarketSeries.objects.create(
        series_key="bcb:12469",
        reference_date=dt.date(2023, 5, 22),
        value=7000.0,
        source="BCB",
        kind="level",
    )
    MarketSeries.objects.create(
        series_key="bcb:12461",
        reference_date=dt.date(2023, 5, 22),
        value=16000.0,
        source="BCB",
        kind="level",
    )
    MarketSeries.objects.create(
        series_key="bcb:12",
        reference_date=dt.date(2023, 5, 22),
        value=13.65,
        source="BCB",
        kind="rate",
    )
    MarketSeries.objects.create(
        series_key="anbima:IMAS",
        reference_date=dt.date(2023, 5, 22),
        value=5926.72,
        source="ANBIMA",
        kind="level",
    )

    bumps = 0

    def fake_bump():
        nonlocal bumps
        bumps += 1

    monkeypatch.setattr(cache, "bump_series_generation", fake_bump)

    mod = importlib.import_module("apps.core.migrations.0003_remove_bcb_ima_series")
    mod.forwards(django.apps.apps, None)

    assert set(MarketSeries.objects.values_list("series_key", flat=True)) == {"bcb:12", "anbima:IMAS"}
    assert bumps == 1
