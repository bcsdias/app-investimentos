import datetime as dt
import io

import pytest

from apps.market.services.sources import tesouro
from apps.market.services.sources.base import FetchError, RawPoint

_CSV = (
    "Tipo Titulo;Data Vencimento;Data Base;PU Base Manha\n"
    "Tesouro IPCA+;15/05/2035;08/03/2010;405,61\n"
    "Tesouro IPCA+;15/05/2035;09/03/2010;403,91\n"
    "Tesouro IPCA+;15/05/2045;08/03/2010;250,00\n"
    "Tesouro Selic;01/03/2027;08/03/2010;9000,00\n"
)


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield _CSV.encode("latin-1")


def test_fetch_tesouro_filters_titulo_and_vencimento(monkeypatch, tmp_path):
    monkeypatch.setattr(tesouro, "_DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(tesouro.requests, "get", lambda *a, **k: _Resp())
    pts = tesouro.fetch_tesouro(
        "Tesouro IPCA+", dt.date(2035, 5, 15), dt.date(2010, 1, 1), dt.date(2010, 12, 31)
    )
    assert pts == [
        RawPoint(dt.date(2010, 3, 8), 405.61),
        RawPoint(dt.date(2010, 3, 9), 403.91),
    ]


def test_fetch_tesouro_no_match_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(tesouro, "_DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(tesouro.requests, "get", lambda *a, **k: _Resp())
    assert tesouro.fetch_tesouro(
        "Tesouro Prefixado", dt.date(2031, 1, 1), dt.date(2010, 1, 1), dt.date(2010, 12, 31)
    ) == []


def test_fetch_tesouro_raises_on_download_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(tesouro, "_DOWNLOAD_DIR", tmp_path)

    def boom(*a, **k):
        raise ConnectionError("tesouro down")

    monkeypatch.setattr(tesouro.requests, "get", boom)
    with pytest.raises(FetchError):
        tesouro.fetch_tesouro(
            "Tesouro IPCA+", dt.date(2035, 5, 15), dt.date(2010, 1, 1), dt.date(2010, 12, 31)
        )
