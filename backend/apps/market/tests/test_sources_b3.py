import datetime as dt

import pytest

from apps.market.services.sources import b3
from apps.market.services.sources.base import FetchError, RawPoint

FIXTURES = "apps/market/tests/fixtures"


def test_read_b3_static_reads_and_filters(tmp_path):
    csv = tmp_path / "IFIX_all.csv"
    csv.write_text(
        "Date,Close\n2023-12-29,3000.0\n2024-01-02,3010.5\n2024-06-03,3200.0\n"
    )
    pts = b3.read_b3_static("IFIX", dt.date(2024, 1, 1), dt.date(2024, 3, 31), static_dir=tmp_path)
    assert pts == [RawPoint(dt.date(2024, 1, 2), 3010.5)]


def test_read_b3_static_missing_file_returns_empty(tmp_path):
    assert b3.read_b3_static("NOPE", dt.date(2024, 1, 1), dt.date(2024, 12, 31), static_dir=tmp_path) == []


def test_read_b3_static_uses_repo_fixture():
    # o IFIX_all.csv da fixture existe e tem dados
    pts = b3.read_b3_static("IFIX", dt.date(2010, 1, 1), dt.date(2030, 1, 1),
                            static_dir=__import__("pathlib").Path(FIXTURES))
    assert len(pts) > 0
    assert isinstance(pts[0], RawPoint)


def test_fetch_b3_selenium_raises_when_webdriver_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no chrome")

    monkeypatch.setattr(b3.webdriver, "Chrome", boom)
    with pytest.raises(FetchError):
        b3.fetch_b3_selenium("IFIX", [2024])
