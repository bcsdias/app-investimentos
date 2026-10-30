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
    # BCB_12 (rate), BCB_433 (rate), YF_GSPC, IFIX_all, TD_...
    assert MarketSeries.objects.filter(series_key="bcb:12", kind="rate").exists()
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


def test_ingest_dispatches_anbima(monkeypatch, db):
    from apps.core.models import MarketSeries
    from apps.market import repository
    from apps.market.services.sources import anbima

    calls = []

    def fake_fetch_anbima(code, start, end):
        calls.append((code, start, end))
        return [RawPoint(dt.date(2024, 1, 2), 6000.0)]

    monkeypatch.setattr(anbima, "fetch_anbima", fake_fetch_anbima)
    call_command("ingest_market_data", "--only", "anbima:IMAS")

    assert calls and calls[0][0] == "IMAS"
    assert repository.last_date("anbima:IMAS") == dt.date(2024, 1, 2)
    s = MarketSeries.objects.filter(series_key="anbima:IMAS").first()
    assert s is not None and s.source == "ANBIMA"


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


# --- ingest_market_data: tickers sob demanda (D9) ---


def test_ingest_updates_on_demand_ticker(monkeypatch, db):
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd_mod

    repository.upsert_points("yf:AAPL", "YF", "level", [RawPoint(dt.date(2024, 1, 2), 1.0)])
    fetched = []
    monkeypatch.setattr(cmd_mod, "_fetch", lambda sdef, s, e: fetched.append(sdef.series_key) or [])
    monkeypatch.setattr(cmd_mod, "_warm_cache", lambda *a, **k: None)
    call_command("ingest_market_data", "--only", "yf:AAPL")
    assert fetched == ["yf:AAPL"]


def test_on_demand_failure_marks_failed(monkeypatch, db, capsys):
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd_mod

    repository.upsert_points("yf:AAPL", "YF", "level", [RawPoint(dt.date(2024, 1, 2), 1.0)])

    def boom(sdef, s, e):
        raise FetchError("yahoo down")

    monkeypatch.setattr(cmd_mod, "_fetch", boom)
    monkeypatch.setattr(cmd_mod, "_warm_cache", lambda *a, **k: None)
    with pytest.raises(SystemExit) as ei:
        call_command("ingest_market_data", "--only", "yf:AAPL")
    assert ei.value.code == 1
    assert "yf:AAPL" in capsys.readouterr().err


def test_catalog_yf_key_not_duplicated(db):
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd_mod

    repository.upsert_points("yf:IVVB11.SA", "YF", "level", [RawPoint(dt.date(2024, 1, 2), 1.0)])
    assert "yf:IVVB11.SA" not in {d.series_key for d in cmd_mod._on_demand_defs()}


def test_ingest_b3_selenium_failure_falls_back_and_marks_failed(monkeypatch, db, settings):
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd

    settings.B3_INGEST_USE_SELENIUM = True
    monkeypatch.setattr(cmd.b3, "fetch_b3_selenium",
                        lambda *a, **k: (_ for _ in ()).throw(FetchError("site mudou")))
    monkeypatch.setattr(cmd.b3, "read_b3_static",
                        lambda ident, s, e: [RawPoint(dt.date(2024, 6, 3), 3200.0)])
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    err = StringIO()
    with pytest.raises(SystemExit) as exc:
        call_command("ingest_market_data", "--only", "b3:IFIX", stderr=err)
    assert exc.value.code == 1
    assert "b3:IFIX" in err.getvalue()
    assert repository.last_date("b3:IFIX") == dt.date(2024, 6, 3)  # histórico do CSV gravado


def test_ingest_b3_selenium_success_does_not_read_static(monkeypatch, db, settings):
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd

    settings.B3_INGEST_USE_SELENIUM = True
    monkeypatch.setattr(cmd.b3, "fetch_b3_selenium",
                        lambda *a, **k: [RawPoint(dt.date(2026, 10, 26), 3500.0)])
    monkeypatch.setattr(cmd.b3, "read_b3_static",
                        lambda *a: pytest.fail("não deve ler CSV"))
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)
    call_command("ingest_market_data", "--only", "b3:IFIX")  # sem SystemExit
    assert repository.last_date("b3:IFIX") == dt.date(2026, 10, 26)


# --- market_gaps ---


def test_market_gaps_lists_gap(db, capsys):
    from apps.market import repository
    from apps.market.services.sources.base import RawPoint

    repository.upsert_points(
        "bcb:12",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(dt.date(2024, 1, 3), 0.04),
            RawPoint(dt.date(2024, 2, 1), 0.05),
        ],
    )
    call_command("market_gaps", "--only", "bcb:12")
    out = capsys.readouterr().out
    assert "bcb:12" in out
    assert "2024-01-03 → 2024-02-01" in out
    assert "29 dias" in out


def test_market_gaps_no_gap_prints_zero_gaps(db, capsys):
    from apps.market import repository
    from apps.market.services.sources.base import RawPoint

    repository.upsert_points(
        "bcb:12",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(dt.date(2024, 1, 3), 0.04),
        ],
    )
    call_command("market_gaps", "--only", "bcb:12")
    out = capsys.readouterr().out
    assert "bcb:12 (daily) última=2024-01-03 lacunas=0" in out


def test_market_gaps_includes_on_demand_yf(db, capsys):
    from apps.market import repository
    from apps.market.services.sources.base import RawPoint

    repository.upsert_points(
        "yf:CUSTOM.SA",
        "YF",
        "level",
        [
            RawPoint(dt.date(2024, 1, 2), 10.0),
            RawPoint(dt.date(2024, 1, 25), 11.0),
        ],
    )
    call_command("market_gaps", "--only", "yf:CUSTOM.SA")
    out = capsys.readouterr().out
    assert "yf:CUSTOM.SA (daily) última=2024-01-25 lacunas=1" in out
    assert "2024-01-02 → 2024-01-25 (23 dias)" in out


def test_market_gaps_prints_stale_tail(monkeypatch, db, capsys):
    from apps.market import repository
    from apps.market.management.commands import market_gaps as cmd
    from apps.market.services.sources.base import RawPoint

    monkeypatch.setattr(cmd, "_today", lambda: dt.date(2024, 3, 1))
    repository.upsert_points(
        "bcb:12",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(dt.date(2024, 1, 10), 0.04),
        ],
    )
    call_command("market_gaps", "--only", "bcb:12")
    out = capsys.readouterr().out
    assert "defasada desde 2024-01-10 (51 dias)" in out



# --- ingest_market_data: re-busca de lacunas (Task 3) ---


def test_ingest_refills_middle_gap(monkeypatch, db):
    from apps.core.models import MarketSeries
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd

    repository.upsert_points(
        "bcb:12",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(dt.date(2024, 2, 1), 0.05),
        ],
    )

    fetch_calls = []

    def fake_fetch(sdef, start, end):
        fetch_calls.append((sdef.series_key, start, end))
        if sdef.series_key == "bcb:12" and start == dt.date(2024, 1, 3):
            return [RawPoint(dt.date(2024, 1, 15), 0.045)]
        return []

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)

    out = StringIO()
    call_command("ingest_market_data", "--only", "bcb:12", stdout=out)

    assert ("bcb:12", dt.date(2024, 1, 3), dt.date(2024, 1, 31)) in fetch_calls
    assert MarketSeries.objects.filter(series_key="bcb:12").count() == 3
    assert "1 lacuna(s) re-buscada(s), 0 sem dado na fonte" in out.getvalue()


def test_gap_without_source_data_is_not_a_failure(monkeypatch, db, caplog):
    import logging
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd

    repository.upsert_points(
        "bcb:12",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(dt.date(2024, 2, 1), 0.05),
        ],
    )

    def fake_fetch(sdef, start, end):
        if start == dt.date(2024, 1, 3):
            raise FetchError("dados indisponíveis no SGS")
        return []

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)

    out = StringIO()
    with caplog.at_level(logging.WARNING):
        call_command("ingest_market_data", "--only", "bcb:12", stdout=out)

    assert "lacuna sem dado na fonte em bcb:12" in caplog.text
    assert "1 lacuna(s) re-buscada(s), 1 sem dado na fonte" in out.getvalue()


def test_only_filters_gap_refill(monkeypatch, db):
    from apps.market import repository
    from apps.market.management.commands import ingest_market_data as cmd

    repository.upsert_points(
        "bcb:12",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(dt.date(2024, 2, 1), 0.05),
        ],
    )
    repository.upsert_points(
        "bcb:11",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(dt.date(2024, 2, 1), 0.05),
        ],
    )

    fetched_keys = set()

    def fake_fetch(sdef, start, end):
        fetched_keys.add(sdef.series_key)
        return []

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)

    call_command("ingest_market_data", "--only", "bcb:12")

    assert "bcb:11" not in fetched_keys
    assert "bcb:12" in fetched_keys


def test_ingest_refills_gap_even_when_already_up_to_date(monkeypatch, db):
    from apps.core.models import MarketSeries
    from apps.market import catalog, repository
    from apps.market.management.commands import ingest_market_data as cmd

    # Data mais recente no futuro de "today" (simulando "já atualizado")
    future = dt.date.today() + dt.timedelta(days=2)
    repository.upsert_points(
        "bcb:12",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(future, 0.05),
        ],
    )

    fetch_calls = []

    def fake_fetch(sdef, start, end):
        fetch_calls.append((sdef.series_key, start, end))
        return [RawPoint(dt.date(2024, 1, 15), 0.045)]

    monkeypatch.setattr(cmd, "_fetch", fake_fetch)
    monkeypatch.setattr(cmd, "_warm_cache", lambda *a, **k: None)

    out = StringIO()
    call_command("ingest_market_data", "--only", "bcb:12", stdout=out)

    val = out.getvalue()
    assert "bcb:12: já atualizado" in val
    assert "bcb:12: 1 lacuna(s) re-buscada(s), 0 sem dado na fonte" in val
    assert MarketSeries.objects.filter(series_key="bcb:12").count() == 3


def test_refill_gaps_function_signature_and_types(monkeypatch, db):
    from apps.market import catalog, gaps, repository
    from apps.market.management.commands import ingest_market_data as cmd

    sdef = catalog.SeriesDef("CDI", "BCB", "bcb:12", "rate")
    repository.upsert_points(
        "bcb:12",
        "BCB",
        "rate",
        [
            RawPoint(dt.date(2024, 1, 2), 0.04),
            RawPoint(dt.date(2024, 2, 1), 0.05),
        ],
    )

    monkeypatch.setattr(cmd, "_fetch", lambda s, st, e: [RawPoint(dt.date(2024, 1, 15), 0.045)])
    pts_written, remaining = cmd._refill_gaps(sdef)

    assert isinstance(pts_written, int)
    assert pts_written == 1
    assert isinstance(remaining, list)
    assert len(remaining) == 0



