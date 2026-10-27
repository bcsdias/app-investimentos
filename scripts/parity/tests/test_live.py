import math

import pandas as pd
import pytest
from requests.cookies import RequestsCookieJar

from parity import legacy_side
from parity.cases import build_cases
from parity.compare import INVESTIGAR, OK
from parity.engine_side import run_engine
from parity.kpis import KPI_NAMES
from parity.live import (
    ApiClient,
    ApiError,
    rows_to_frame,
    series_list_to_frame,
    summary_to_kpis,
    yearly_to_frame,
)
from parity.tests.synthetic import TODAY, make_hist, make_snapshot

BASE = "http://api.test"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, set_cookie=None):
        self.status_code = status_code
        self._payload = payload
        self.set_cookie = set_cookie or {}
        self.text = "" if payload is None else str(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("sem corpo")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.cookies = RequestsCookieJar()
        self.responses = list(responses)
        self.requests = []

    def _respond(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        resp = self.responses.pop(0)
        for k, v in resp.set_cookie.items():
            self.cookies.set(k, v)
        return resp

    def get(self, url, **kwargs):
        return self._respond("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._respond("POST", url, **kwargs)


def test_login_sends_csrf_header_from_cookie():
    s = FakeSession([
        FakeResponse(204, set_cookie={"csrftoken": "tok-1"}),
        FakeResponse(200, {"id": 1, "username": "u", "email": ""}, set_cookie={"csrftoken": "tok-2"}),
    ])
    ApiClient(BASE, session=s).login("u", "p")
    (m1, u1, _), (m2, u2, kw2) = s.requests
    assert (m1, u1) == ("GET", f"{BASE}/api/v1/auth/csrf")
    assert (m2, u2) == ("POST", f"{BASE}/api/v1/auth/login")
    assert kw2["headers"]["X-CSRFToken"] == "tok-1"
    assert kw2["json"] == {"username": "u", "password": "p"}


def test_post_sends_csrf_header_and_returns_json():
    s = FakeSession([FakeResponse(200, {"data": {"x": 1}})])
    s.cookies.set("csrftoken", "tok-9")
    client = ApiClient(BASE + "/", session=s)
    assert client.post("/api/v1/portfolio/summary", {"a": 1}) == {"data": {"x": 1}}
    method, url, kw = s.requests[0]
    assert url == f"{BASE}/api/v1/portfolio/summary"
    assert kw["headers"]["X-CSRFToken"] == "tok-9"
    assert kw["json"] == {"a": 1}


def test_api_client_error_has_no_body_or_token():
    body = {"code": "dlp_token_rejected", "message": "Token rejeitado", "details": {"x": "DLP-segredo"}}
    s = FakeSession([FakeResponse(422, body)])
    s.cookies.set("csrftoken", "csrf-segredo")
    with pytest.raises(ApiError) as info:
        ApiClient(BASE, session=s).post("/api/v1/portfolio/summary", {})
    exc = info.value
    assert exc.status == 422
    assert exc.code == "dlp_token_rejected"
    assert str(exc) == "HTTP 422 (dlp_token_rejected)"
    for secret in ("DLP-segredo", "csrf-segredo", "Token rejeitado"):
        assert secret not in str(exc)
        assert secret not in repr(exc)


def test_api_client_error_without_json_body():
    s = FakeSession([FakeResponse(502, None)])
    with pytest.raises(ApiError) as info:
        ApiClient(BASE, session=s).post("/x", {})
    assert info.value.code is None
    assert str(info.value) == "HTTP 502 (None)"


def test_series_list_to_frame_handles_nulls():
    df = series_list_to_frame([
        {"name": "Carteira", "data": [["2024-01-02", 100.0], ["2024-01-03", None]]},
        {"name": "CDI", "data": [["2024-01-02", 100.0], ["2024-01-03", 100.04]]},
    ])
    assert list(df.columns) == ["Carteira", "CDI"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert math.isnan(df.loc["2024-01-03", "Carteira"])
    assert df.loc["2024-01-03", "CDI"] == 100.04
    assert series_list_to_frame([]).empty


def test_rows_to_frame_restores_legacy_column_names():
    df = rows_to_frame([
        {"name": "Carteira", "volatilidade": 0.2, "retorno_cagr": 0.1, "sharpe": None},
    ])
    assert list(df.columns) == ["Volatilidade", "Retorno (CAGR)", "Sharpe"]
    assert list(df.index) == ["Carteira"]
    assert math.isnan(df.loc["Carteira", "Sharpe"])


def test_yearly_to_frame_columns():
    df = yearly_to_frame({
        "years": [2024, 2025],
        "rows": [{"name": "CDI", "values": {"2024": 0.1, "2025": None}, "total": 0.15}],
    })
    assert list(df.columns) == ["2024", "2025", "Total Acum."]
    assert df.loc["CDI", "Total Acum."] == 0.15
    assert math.isnan(df.loc["CDI", "2025"])


def test_summary_to_kpis_index_matches_derive_kpis():
    data = {k: 1.0 for k in KPI_NAMES} | {"xirr": None, "data_inicio": "2024-01-01", "data_fim": "2024-12-31"}
    df = summary_to_kpis(data)
    assert list(df.index) == list(KPI_NAMES)
    assert list(df.columns) == ["valor"]
    assert math.isnan(df.loc["xirr", "valor"])


def test_run_case_layer2_legacy_missing_column(monkeypatch):
    case = build_cases(make_hist(), TODAY)[0]
    new_out = run_engine(case, make_snapshot(), make_hist())
    legacy_out = {k: v.copy() for k, v in new_out.items()}
    legacy_out["twr"] = legacy_out["twr"].drop(columns=["IFIX"])
    api_out = {k: v for k, v in new_out.items() if k != "portfolio"}

    monkeypatch.setattr(legacy_side, "run_legacy", lambda token, c: (object(), legacy_out))
    monkeypatch.setattr(legacy_side, "api_outputs", lambda client, c: api_out)

    diffs, notes, leg, new = legacy_side.run_layer2(case, "tok", client=None)
    assert notes == []
    bad = [d for d in diffs if d.status != OK]
    assert len(bad) == 1
    (d,) = bad
    assert (d.metric, d.column, d.status, d.note) == ("twr", "IFIX", INVESTIGAR, "só no novo")
    assert d.origin
    assert all(d.metric != "portfolio" for d in diffs)


def test_run_case_layer2_api_error_becomes_note(monkeypatch):
    case = build_cases(make_hist(), TODAY)[0]

    def boom(client, c):
        raise ApiError(422, "dlp_token_rejected")

    monkeypatch.setattr(legacy_side, "run_legacy", lambda token, c: (object(), {}))
    monkeypatch.setattr(legacy_side, "api_outputs", boom)
    diffs, notes, _, _ = legacy_side.run_layer2(case, "tok", client=None)
    assert diffs == []
    assert notes == ["API: HTTP 422 (dlp_token_rejected)"]


def test_run_case_layer2_legacy_none_becomes_note(monkeypatch):
    case = build_cases(make_hist(), TODAY)[0]
    monkeypatch.setattr(legacy_side, "run_legacy", lambda token, c: None)
    monkeypatch.setattr(legacy_side, "api_outputs", lambda client, c: {})
    diffs, notes, _, _ = legacy_side.run_layer2(case, "tok", client=None)
    assert diffs == []
    assert notes == ["legado: sem carteira"]
