"""Camada 2, lado novo: a API real em execução (containers da Fase 5) via HTTP.

Credenciais só chegam por argumento (lidas de env pelo CLI); erros carregam o
status HTTP e o `code` da API, nunca o corpo nem headers.
"""
from __future__ import annotations

import math

import pandas as pd
import requests

from parity.cases import Case
from parity.engine_side import ROLLING_WINDOW
from parity.kpis import KPI_NAMES

ENDPOINTS = {
    "twr": "/api/v1/portfolio/twr-evolution",
    "drawdown": "/api/v1/portfolio/drawdown",
    "rolling": "/api/v1/portfolio/rolling-metrics",
    "risk_return": "/api/v1/portfolio/risk-return",
    "irr": "/api/v1/portfolio/irr-evolution",
    "shadow": "/api/v1/portfolio/shadow",
    "yearly": "/api/v1/portfolio/yearly-summary",
    "summary": "/api/v1/portfolio/summary",
}

# Inverso de _RISK_RETURN_COLUMNS de apps/api/routers/portfolio.py.
_RISK_RETURN_COLUMNS = {
    "volatilidade": "Volatilidade",
    "retorno_cagr": "Retorno (CAGR)",
    "sharpe": "Sharpe",
}

_TIMEOUT = 300  # s — análise de histórico completo pode demorar no LAB


class ApiError(RuntimeError):
    def __init__(self, status: int, code: str | None = None):
        super().__init__(f"HTTP {status} ({code})")
        self.status = status
        self.code = code


def _error_code(resp) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    code = body.get("code") if isinstance(body, dict) else None
    return str(code) if code is not None else None


class ApiClient:
    def __init__(self, base_url: str, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()

    def _headers(self) -> dict:
        return {"X-CSRFToken": self.session.cookies.get("csrftoken") or ""}

    def _check(self, resp) -> None:
        if resp.status_code not in (200, 204):
            raise ApiError(resp.status_code, _error_code(resp))

    def login(self, username: str, password: str) -> None:
        self._check(self.session.get(f"{self.base_url}/api/v1/auth/csrf", timeout=_TIMEOUT))
        resp = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password},
            headers=self._headers(),
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            raise ApiError(resp.status_code, _error_code(resp))

    def post(self, path: str, payload: dict) -> dict:
        resp = self.session.post(
            f"{self.base_url}{path}", json=payload, headers=self._headers(), timeout=_TIMEOUT
        )
        if resp.status_code != 200:
            raise ApiError(resp.status_code, _error_code(resp))
        return resp.json()


def _num(v) -> float:
    return math.nan if v is None else float(v)


def series_list_to_frame(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    cols = {}
    for item in items:
        data = item.get("data") or []
        idx = pd.to_datetime([d for d, _ in data])
        cols[item["name"]] = pd.Series([_num(v) for _, v in data], index=idx, dtype=float)
    return pd.DataFrame(cols)


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(_RISK_RETURN_COLUMNS.values()), dtype=float)
    return pd.DataFrame(
        {target: [_num(r.get(src)) for r in rows] for src, target in _RISK_RETURN_COLUMNS.items()},
        index=[r["name"] for r in rows],
    )


def yearly_to_frame(data: dict) -> pd.DataFrame:
    years = [str(y) for y in data.get("years") or []]
    rows = data.get("rows") or []
    if not rows:
        return pd.DataFrame()
    frame = {y: [_num(r["values"].get(y)) for r in rows] for y in years}
    frame["Total Acum."] = [_num(r.get("total")) for r in rows]
    return pd.DataFrame(frame, index=[r["name"] for r in rows])


def summary_to_kpis(data: dict) -> pd.DataFrame:
    return pd.DataFrame({"valor": [_num(data.get(k)) for k in KPI_NAMES]}, index=list(KPI_NAMES))


def api_outputs(client: ApiClient, case: Case) -> dict[str, pd.DataFrame]:
    payload = case.api_payload()

    def data(name: str, body: dict = payload) -> dict:
        return client.post(ENDPOINTS[name], body)["data"]

    rolling = data("rolling", payload | {"window": ROLLING_WINDOW})
    irr = series_list_to_frame(data("irr")["series"])
    return {
        "twr": series_list_to_frame(data("twr")["series"]),
        "drawdown": series_list_to_frame(data("drawdown")["series"]),
        "rolling_vol": series_list_to_frame(rolling["volatility"]),
        "rolling_sharpe": series_list_to_frame(rolling["sharpe"]),
        "risk_return": rows_to_frame(data("risk_return")["rows"]),
        "irr": irr if not irr.empty else pd.DataFrame(columns=["TIR"], dtype=float),
        "shadow": series_list_to_frame(data("shadow")["series"]),
        "yearly": yearly_to_frame(data("yearly")),
        "kpis": summary_to_kpis(data("summary")),
    }
