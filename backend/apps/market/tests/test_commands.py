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
    msg = out.getvalue()
    assert "IBLV-2005.csv" in msg
    assert "pulado (não mapeado)" in msg  # texto ASCII latino, sem homóglifo cirílico


# --- ingest_market_data ---

from apps.market.services.sources.base import FetchError, RawPoint


def test_ingest_incremental_from_last_date(monkeypatch, db):
    from apps.core.models import MarketSeries
    from apps.market import repository

    repository.upsert_points("bcb:12", "BCB", "rate", [RawPoint(dt.date(2024, 1, 2), 0.04)])

    seen = {}

    def fake_fetch(sdef, start, end):
        seen[sdef.series_key] = (start, end)
        return [RawPoint(dt.date(2024, 1, 3), 0.05)]

    from apps.market.management.commands import ingest_market_data as cmd

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    call_command("ingest_market_data", "--only", "bcb:12")
    assert seen["bcb:12"][0] == dt.date(2024, 1, 3)  # last_date + 1
    assert MarketSeries.objects.filter(series_key="bcb:12").count() == 2


def test_ingest_full_starts_from_1995(monkeypatch, db):
    seen = {}
    from apps.market.management.commands import ingest_market_data as cmd

    def fake_fetch(sdef, start, end):
        seen[sdef.series_key] = start
        return []

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    call_command("ingest_market_data", "--only", "bcb:12", "--full")
    assert seen["bcb:12"] == dt.date(1995, 1, 1)


def test_ingest_one_source_fails_others_continue(monkeypatch, db):
    from apps.core.models import MarketSeries
    from apps.market.management.commands import ingest_market_data as cmd

    def fake_fetch(sdef, start, end):
        if sdef.series_key == "bcb:12":
            raise FetchError("bcb down")
        return [RawPoint(dt.date(2024, 6, 3), 1.0)]

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    with pytest.raises(SystemExit) as ei:
        call_command("ingest_market_data", "--only", "bcb:12,bcb:11")
    assert ei.value.code == 1
    assert MarketSeries.objects.filter(series_key="bcb:11").exists()  # a outra passou


def test_warm_cache_does_not_crash_on_leap_day(monkeypatch):
    # today.replace(year=today.year - 5) levanta ValueError em 29/fev (ano-alvo
    # não bissexto), e a expressão fica fora do try do loop.
    from apps.market.management.commands import ingest_market_data as cmd

    seen_starts = []
    monkeypatch.setattr(cmd.series, "get_series", lambda name, s, e: seen_starts.append(s))
    cmd._warm_cache(dt.date(2028, 2, 29))  # não pode levantar
    assert seen_starts  # e chegou a chamar get_series


def test_ingest_db_error_on_one_series_does_not_abort_batch(monkeypatch, db):
    from apps.core.models import MarketSeries
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd

    monkeypatch.setattr(cmd, "_fetch", lambda s, a, b: [RawPoint(dt.date(2024, 6, 3), 1.0)])
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)

    real_upsert = repository.upsert_points

    def flaky_upsert(series_key, *a, **k):
        if series_key == "bcb:12":
            raise __import__("django").db.DatabaseError("deadlock")
        return real_upsert(series_key, *a, **k)

    monkeypatch.setattr(cmd.repository, "upsert_points", flaky_upsert)
    with pytest.raises(SystemExit) as ei:
        call_command("ingest_market_data", "--only", "bcb:12,bcb:11")
    assert ei.value.code == 1
    assert MarketSeries.objects.filter(series_key="bcb:11").exists()  # a outra gravou
