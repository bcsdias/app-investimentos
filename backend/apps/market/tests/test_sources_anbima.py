import datetime as dt
import io

import pandas as pd
import pytest

from apps.market.services.sources import anbima
from apps.market.services.sources.base import FetchError, RawPoint


class _DummyResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _build_excel_content(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


def test_fetch_history_filters_index_and_window(monkeypatch):
    rows = [
        {"Índice": "IMA-S", "Data de Referência": pd.Timestamp(2024, 1, 2), "Número Índice": 6393.10},
        {"Índice": "IMA-S", "Data de Referência": pd.Timestamp(2024, 1, 3), "Número Índice": 6396.27},
        {"Índice": "IMA-S", "Data de Referência": pd.Timestamp(2024, 1, 4), "Número Índice": 6399.20},
        {"Índice": "IMA - S", "Data de Referência": pd.Timestamp(2026, 1, 5), "Número Índice": 8130.00},
        {"Índice": "IMA-B", "Data de Referência": pd.Timestamp(2024, 1, 3), "Número Índice": 8000.00},
    ]
    content = _build_excel_content(rows)
    monkeypatch.setattr(anbima.requests, "get", lambda *a, **k: _DummyResponse(content))

    pts = anbima.fetch_history("IMAS", dt.date(2024, 1, 3), dt.date(2024, 1, 4))
    assert pts == [
        RawPoint(dt.date(2024, 1, 3), 6396.27),
        RawPoint(dt.date(2024, 1, 4), 6399.20),
    ]

    pts2026 = anbima.fetch_history("IMAS", dt.date(2026, 1, 5), dt.date(2026, 1, 5))
    assert pts2026 == [RawPoint(dt.date(2026, 1, 5), 8130.00)]


def test_fetch_history_empty_window_returns_empty(monkeypatch):
    rows = [
        {"Índice": "IMA-S", "Data de Referência": pd.Timestamp(2024, 1, 2), "Número Índice": 6393.10},
    ]
    content = _build_excel_content(rows)
    monkeypatch.setattr(anbima.requests, "get", lambda *a, **k: _DummyResponse(content))

    pts = anbima.fetch_history("IMAS", dt.date(2024, 2, 1), dt.date(2024, 2, 5))
    assert pts == []


def test_fetch_history_http_error_raises(monkeypatch):
    monkeypatch.setattr(anbima.requests, "get", lambda *a, **k: _DummyResponse(b"", status_code=403))
    with pytest.raises(FetchError):
        anbima.fetch_history("IMAS", dt.date(2024, 1, 1), dt.date(2024, 1, 10))


def test_fetch_history_renamed_column_raises(monkeypatch):
    rows = [
        {"Índice": "IMA-S", "Data de Referência": pd.Timestamp(2024, 1, 2), "Numero Indice": 6393.10},
    ]
    content = _build_excel_content(rows)
    monkeypatch.setattr(anbima.requests, "get", lambda *a, **k: _DummyResponse(content))

    with pytest.raises(FetchError):
        anbima.fetch_history("IMAS", dt.date(2024, 1, 1), dt.date(2024, 1, 10))


def test_fetch_history_without_index_rows_raises(monkeypatch):
    rows = [
        {"Índice": "IMA-B", "Data de Referência": pd.Timestamp(2024, 1, 2), "Número Índice": 8000.00},
    ]
    content = _build_excel_content(rows)
    monkeypatch.setattr(anbima.requests, "get", lambda *a, **k: _DummyResponse(content))

    with pytest.raises(FetchError):
        anbima.fetch_history("IMAS", dt.date(2024, 1, 1), dt.date(2024, 1, 10))


def test_fetch_history_unknown_code_raises():
    with pytest.raises(FetchError):
        anbima.fetch_history("XPTO", dt.date(2024, 1, 1), dt.date(2024, 1, 10))


def test_fetch_daily_parses_brazilian_number(monkeypatch):
    csv_text = (
        "TOTAIS - QUADRO RESUMO\r\n"
        "Índice;Data de Referência;Número Índice;Variação Diária (%)\r\n"
        "IMA-S;29/10/2026;9.086,067150;0,04\r\n"
        "IMA-B;29/10/2026;11.234,567890;0,10\r\n"
    )
    content = csv_text.encode("cp1252")
    monkeypatch.setattr(anbima.requests, "post", lambda *a, **k: _DummyResponse(content))

    pt = anbima.fetch_daily("IMAS", dt.date(2026, 10, 29))
    assert pt == RawPoint(dt.date(2026, 10, 29), 9086.06715)


def test_fetch_daily_without_row_returns_none(monkeypatch):
    csv_text = "Índice;Data de Referência;Número Índice;Variação Diária (%)\r\n"
    content = csv_text.encode("cp1252")
    monkeypatch.setattr(anbima.requests, "post", lambda *a, **k: _DummyResponse(content))

    pt = anbima.fetch_daily("IMAS", dt.date(2026, 10, 29))
    assert pt is None


def test_fetch_daily_no_data_message_returns_none(monkeypatch):
    # Resposta real da ANBIMA em feriado, fim de semana ou dia ainda não publicado.
    content = "Não há dados disponíveis para [25/12/2024] !\r\n".encode("cp1252")
    monkeypatch.setattr(anbima.requests, "post", lambda *a, **k: _DummyResponse(content))

    assert anbima.fetch_daily("IMAS", dt.date(2024, 12, 25)) is None


def test_fetch_daily_unexpected_body_raises(monkeypatch):
    content = "<html><body>Erro interno</body></html>".encode("cp1252")
    monkeypatch.setattr(anbima.requests, "post", lambda *a, **k: _DummyResponse(content))

    with pytest.raises(FetchError):
        anbima.fetch_daily("IMAS", dt.date(2024, 12, 24))


def test_fetch_daily_row_with_other_date_returns_none(monkeypatch):
    csv_text = (
        "Índice;Data de Referência;Número Índice\r\n"
        "IMA-S;28/10/2026;9.082,000000\r\n"
    )
    monkeypatch.setattr(anbima.requests, "post", lambda *a, **k: _DummyResponse(csv_text.encode("cp1252")))

    assert anbima.fetch_daily("IMAS", dt.date(2026, 10, 29)) is None


def test_fetch_anbima_fallback_skips_holiday(monkeypatch):
    # S3 fora do ar; janela 23..27/12/2024 com o Natal no meio.
    def _fail_history(*a, **k):
        raise FetchError("s3 down")

    def _post(url, data=None, **kwargs):
        if data["Dt_Ref"] == "25/12/2024":
            body = f"Não há dados disponíveis para [{data['Dt_Ref']}] !\r\n"
        else:
            body = (
                "TOTAIS - QUADRO RESUMO\r\n"
                "Índice;Data de Referência;Número Índice\r\n"
                f"IMA-S;{data['Dt_Ref']};7.000,000000\r\n"
            )
        return _DummyResponse(body.encode("cp1252"))

    monkeypatch.setattr(anbima, "fetch_history", _fail_history)
    monkeypatch.setattr(anbima.requests, "post", _post)

    pts = anbima.fetch_anbima("IMAS", dt.date(2024, 12, 23), dt.date(2024, 12, 27))
    assert [p.date.day for p in pts] == [23, 24, 26, 27]


def test_fetch_daily_posts_expected_form(monkeypatch):
    captured_data = {}

    def _post(url, data=None, **kwargs):
        captured_data.update(data or {})
        return _DummyResponse("Índice;Data de Referência;Número Índice\r\n".encode("cp1252"), status_code=200)

    monkeypatch.setattr(anbima.requests, "post", _post)
    anbima.fetch_daily("IMAS", dt.date(2024, 1, 2))

    assert captured_data == {
        "Pai": "ima",
        "escolha": "2",
        "Idioma": "PT",
        "saida": "csv",
        "Dt_Ref": "02/01/2024",
        "Dt_Ref_Ver": "20240102",
    }


def test_fetch_daily_http_error_raises(monkeypatch):
    monkeypatch.setattr(anbima.requests, "post", lambda *a, **k: _DummyResponse(b"", status_code=500))
    with pytest.raises(FetchError):
        anbima.fetch_daily("IMAS", dt.date(2024, 1, 2))


def test_fetch_anbima_falls_back_to_daily_for_short_window(monkeypatch):
    def _fail_history(*a, **k):
        raise FetchError("s3 down")

    monkeypatch.setattr(anbima, "fetch_history", _fail_history)

    calls = []

    def _mock_daily(code, day):
        calls.append(day)
        # Retorna ponto apenas para terça e quinta
        if day.weekday() in (1, 3):
            return RawPoint(day, 1000.0)
        return None

    monkeypatch.setattr(anbima, "fetch_daily", _mock_daily)

    # Janela de 7 dias: 2024-01-01 (seg) até 2024-01-07 (dom) -> 5 dias úteis (seg-sex: 1, 2, 3, 4, 5)
    pts = anbima.fetch_anbima("IMAS", dt.date(2024, 1, 1), dt.date(2024, 1, 7))
    assert len(calls) == 5
    assert len(pts) == 2
    assert pts == [RawPoint(dt.date(2024, 1, 2), 1000.0), RawPoint(dt.date(2024, 1, 4), 1000.0)]


def test_fetch_anbima_long_window_does_not_fall_back(monkeypatch):
    def _fail_history(*a, **k):
        raise FetchError("s3 down")

    monkeypatch.setattr(anbima, "fetch_history", _fail_history)

    daily_called = False

    def _mock_daily(*a, **k):
        nonlocal daily_called
        daily_called = True
        return None

    monkeypatch.setattr(anbima, "fetch_daily", _mock_daily)

    with pytest.raises(FetchError):
        anbima.fetch_anbima("IMAS", dt.date(2023, 1, 1), dt.date(2024, 2, 5))

    assert not daily_called
