import datetime as dt
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command

from apps.core.models import MarketSeries

pytestmark = pytest.mark.django_db

FIX = Path("apps/market/tests/fixtures")


def test_backfill_loads_fixture_csvs(tmp_path):
    out = StringIO()
    call_command(
        "backfill_market_data",
        "--raw-dir", str(FIX),
        "--static-dir", str(FIX),
        stdout=out,
    )
    # BCB_12 (rate), BCB_433 (rate), BCB_12466 (level), YF_GSPC, IFIX_all, TD_...
    assert MarketSeries.objects.filter(series_key="bcb:12", kind="rate").exists()
    assert MarketSeries.objects.filter(series_key="bcb:12466", kind="level").exists()
    assert MarketSeries.objects.filter(series_key="yf:^GSPC").exists()
    assert MarketSeries.objects.filter(series_key="b3:IFIX").exists()
    assert "pontos" in out.getvalue().lower()


def test_backfill_is_idempotent():
    call_command("backfill_market_data", "--raw-dir", str(FIX), "--static-dir", str(FIX))
    n1 = MarketSeries.objects.count()
    call_command("backfill_market_data", "--raw-dir", str(FIX), "--static-dir", str(FIX))
    assert MarketSeries.objects.count() == n1


def test_backfill_dry_run_writes_nothing():
    call_command("backfill_market_data", "--raw-dir", str(FIX), "--static-dir", str(FIX), "--dry-run")
    assert MarketSeries.objects.count() == 0


def test_backfill_only_filter():
    call_command(
        "backfill_market_data", "--raw-dir", str(FIX), "--static-dir", str(FIX),
        "--only", "bcb:12",
    )
    assert set(MarketSeries.objects.values_list("series_key", flat=True).distinct()) == {"bcb:12"}


def test_backfill_skips_unmapped_files(tmp_path):
    (tmp_path / "IBLV-2005.csv").write_text("Dia;Jan\n1;100,0\n")
    (tmp_path / "BCB_12.csv").write_text("Date,12\n2024-01-02,0.0437\n")
    out = StringIO()
    call_command("backfill_market_data", "--raw-dir", str(tmp_path), "--static-dir", str(tmp_path), stdout=out)
    assert MarketSeries.objects.filter(series_key="bcb:12").exists()
    assert "IBLV-2005.csv" in out.getvalue() or "pulad" in out.getvalue().lower()
