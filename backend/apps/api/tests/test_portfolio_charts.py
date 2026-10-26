import json

import pytest

from apps.api.tests.conftest import PORTFOLIO_DF, PORTFOLIO_TWR
from engine.compute import compute_twr_evolution
from engine.dataset import build_dataset

P = "/api/v1/portfolio"
BODY = {"start": "2024-01-02", "end": "2025-06-30", "benchmarks": ["CDI"]}


def _post(c, path, body=BODY):
    return c.post(f"{P}/{path}", body, content_type="application/json")


@pytest.mark.parametrize(
    "path",
    [
        "twr-evolution",
        "drawdown",
        "rolling-metrics",
        "risk-return",
        "shadow",
        "yearly-summary",
        "irr-evolution",
    ],
)
def test_chart_endpoints_return_envelope(portfolio_env, path):
    r = _post(portfolio_env, path)
    assert r.status_code == 200 and set(r.json()) == {"data", "meta"}


@pytest.mark.parametrize("path", ["twr-evolution", "drawdown"])
def test_chart_requires_session(client, db, path):
    assert _post(client, path).status_code == 401


def test_twr_values_match_engine(portfolio_env):
    r = _post(portfolio_env, "twr-evolution")
    assert r.status_code == 200
    payload = r.json()["data"]

    ds = build_dataset(
        user_series=PORTFOLIO_TWR.copy(),
        benchmarks=["CDI"],
        start_date=BODY["start"],
        end_date=BODY["end"],
        portfolio_df=PORTFOLIO_DF.copy(),
    )
    expected = compute_twr_evolution(ds.df_combined)

    carteira_series = next(s for s in payload["series"] if s["name"] == "Carteira")
    assert abs(carteira_series["data"][-1][1] - float(expected["series"]["Carteira"].iloc[-1])) < 1e-9
    for k, v in expected["final_returns"].items():
        assert abs(payload["final_returns"][k] - v) < 1e-9


def test_rolling_metrics_is_strict_json(portfolio_env):
    def no_nan(x):
        raise AssertionError(f"constante não-JSON {x}")

    r = _post(portfolio_env, "rolling-metrics", {**BODY, "benchmarks": ["CDI", "BOVA11"]})
    body = json.loads(r.content, parse_constant=no_nan)
    bova = next(s for s in body["data"]["volatility"] if s["name"] == "BOVA11")
    assert bova["data"][0][1] is None



def test_rolling_window_out_of_range(portfolio_env):
    r = _post(portfolio_env, "rolling-metrics", {**BODY, "window": 10})
    assert r.status_code == 422 and r.json()["code"] == "invalid_window"


def test_missing_benchmark_reported_in_meta(portfolio_env):
    r = _post(portfolio_env, "twr-evolution", {**BODY, "benchmarks": ["CDI", "IMA-B"]})
    assert r.status_code == 200 and r.json()["meta"]["missing"] == ["IMA-B"]


def test_meta_reflects_effective_period(portfolio_env):
    assert _post(portfolio_env, "twr-evolution").json()["meta"]["start"] == "2024-03-01"


def test_no_token_is_409(auth_client):
    r = _post(auth_client, "twr-evolution")
    assert r.status_code == 409 and r.json()["code"] == "dlp_token_missing"


def test_yearly_summary_shape(portfolio_env):
    d = _post(portfolio_env, "yearly-summary").json()["data"]
    assert d["years"] == [2024, 2025] and {r["name"] for r in d["rows"]} >= {"Carteira", "CDI"}
