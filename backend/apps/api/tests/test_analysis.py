import datetime as dt

import pandas as pd
import pytest

from apps.api import analysis
from apps.api.analysis import run_analysis, validate_request
from apps.api.errors import ApiError
from apps.api.schemas import AnalysisRequest
from apps.market import catalog
from apps.market.services.sources.base import DLPAuthError, DLPError, DLPUnavailableError
from apps.market.synthetics import SyntheticSpec
from engine.dataset import DatasetResult

TODAY = dt.date(2026, 10, 26)


def _req(**kw) -> AnalysisRequest:
    return AnalysisRequest(**kw)


def _code(fn):
    with pytest.raises(ApiError) as ei:
        fn()
    return ei.value.status, ei.value.code


def test_defaults():
    v = validate_request(_req(), today=TODAY)
    assert (v.start, v.end) == (dt.date(2025, 10, 26), TODAY)
    assert v.benchmarks == catalog.DEFAULT_BENCHMARKS


def test_empty_benchmarks_is_valid():
    assert validate_request(_req(benchmarks=[]), today=TODAY).benchmarks == []


def test_period_inverted():
    assert _code(
        lambda: validate_request(_req(start=TODAY, end=dt.date(2026, 1, 1)), today=TODAY)
    ) == (422, "invalid_period")


def test_unknown_benchmark_lists_invalid(assets_csv):
    with pytest.raises(ApiError) as ei:
        validate_request(_req(benchmarks=["CDI", "PETR4.SA BRL", "XPTO"]), today=TODAY)
    assert ei.value.status == 422
    assert ei.value.code == "unknown_benchmark"
    assert ei.value.details == {"invalid": ["PETR4.SA BRL", "XPTO"]}


def test_ticker_and_brl_variant_accepted(assets_csv):
    assert validate_request(
        _req(benchmarks=["PETR4.SA", "AAPL BRL"]), today=TODAY
    ).benchmarks == ["PETR4.SA", "AAPL BRL"]


@pytest.mark.parametrize(
    "syn,ok",
    [
        ({"base": "CDI", "operador": "+", "valor": 50}, True),
        ({"base": "CDI", "operador": "+", "valor": 50.01}, False),
        ({"base": "CDI", "operador": "+", "valor": 0}, False),
        ({"base": "CDI", "operador": "%", "valor": 500}, True),
        ({"base": "CDI", "operador": "%", "valor": 501}, False),
        ({"base": "CDI", "operador": "%", "valor": 0}, False),
        ({"base": "SELIC", "operador": "+", "valor": 2}, False),
        ({"base": "IPCA", "operador": "x", "valor": 2}, False),
    ],
)
def test_synthetic_bounds(syn, ok):
    req = _req(benchmarks=[], synthetics=[syn])
    if ok:
        v = validate_request(req, today=TODAY)
        assert len(v.specs) == 1
    else:
        assert _code(lambda: validate_request(req, today=TODAY)) == (422, "invalid_synthetic")


def test_too_many_items():
    req = _req(
        benchmarks=["CDI"] * 14,
        synthetics=[{"base": "CDI", "operador": "+", "valor": 1}] * 2,
    )
    assert _code(lambda: validate_request(req, today=TODAY)) == (422, "too_many_benchmarks")


def test_no_token_409(user):
    assert _code(lambda: run_analysis(user, _req(benchmarks=[]))) == (409, "dlp_token_missing")


def test_dlp_auth_error_422(user_with_token, monkeypatch):
    def boom(*a, **k):
        raise DLPAuthError("401")

    monkeypatch.setattr(analysis, "fetch_user_portfolio", boom)
    assert _code(lambda: run_analysis(user_with_token, _req(benchmarks=[]))) == (
        422,
        "dlp_token_rejected",
    )


@pytest.mark.parametrize("exc", [DLPUnavailableError("503"), DLPError("400")])
def test_dlp_unavailable_502(user_with_token, monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(analysis, "fetch_user_portfolio", boom)
    assert _code(lambda: run_analysis(user_with_token, _req(benchmarks=[]))) == (
        502,
        "dlp_unavailable",
    )


def test_empty_portfolio_404(user_with_token, monkeypatch):
    monkeypatch.setattr(analysis, "fetch_user_portfolio", lambda *a, **k: (None, None))
    assert _code(lambda: run_analysis(user_with_token, _req(benchmarks=[]))) == (
        404,
        "portfolio_empty",
    )


def test_filters_and_specs_forwarded(user_with_token, monkeypatch):
    captured_fetch = {}
    captured_build = {}
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    us = pd.Series([100.0, 101.0], index=idx)
    pdf = pd.DataFrame(
        {
            "date": idx,
            "vlr_mercado": [100.0, 101.0],
            "vlr_investido": [100.0, 100.0],
            "proventos": [0.0, 0.0],
            "fluxo": [100.0, 0.0],
        }
    )

    def fake_fetch(token, **kw):
        captured_fetch.update({"token": token, **kw})
        return us, pdf

    def fake_build(**kw):
        captured_build.update(kw)
        df = pd.DataFrame({"Carteira": [100.0, 101.0], "CDI + 2%": [100.0, 100.5]}, index=idx)
        return DatasetResult(df_combined=df, rf_series=None, portfolio_df=pdf)

    monkeypatch.setattr(analysis, "fetch_user_portfolio", fake_fetch)
    monkeypatch.setattr(analysis, "build_dataset", fake_build)

    req = _req(
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 1, 3),
        ativo="PETR4",
        classe="ACAO",
        benchmarks=[],
        synthetics=[{"base": "CDI", "operador": "+", "valor": 2.0}],
    )
    ctx = run_analysis(user_with_token, req)
    assert captured_fetch == {
        "token": "tok-teste",
        "ativo": "PETR4",
        "classe": "ACAO",
        "start_date": "2024-01-02",
        "end_date": "2024-01-03",
    }
    assert captured_build["extra_specs"] == [SyntheticSpec("CDI", "+", 2.0)]
    assert ctx.start == dt.date(2024, 1, 2)
    assert ctx.end == dt.date(2024, 1, 3)
    assert ctx.missing == []


def test_missing_lists_absent_names(user_with_token, monkeypatch):
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    us = pd.Series([100.0, 101.0], index=idx)
    pdf = pd.DataFrame(
        {
            "date": idx,
            "vlr_mercado": [100.0, 101.0],
            "vlr_investido": [100.0, 100.0],
            "proventos": [0.0, 0.0],
            "fluxo": [100.0, 0.0],
        }
    )

    monkeypatch.setattr(analysis, "fetch_user_portfolio", lambda *a, **k: (us, pdf))

    def fake_build(**kw):
        df = pd.DataFrame({"Carteira": [100.0, 101.0], "CDI": [100.0, 100.2]}, index=idx)
        return DatasetResult(df_combined=df, rf_series=None, portfolio_df=pdf)

    monkeypatch.setattr(analysis, "build_dataset", fake_build)

    req = _req(
        start=dt.date(2024, 1, 2),
        end=dt.date(2024, 1, 3),
        benchmarks=["CDI", "IMA-B"],
        synthetics=[{"base": "CDI", "operador": "+", "valor": 2.0}],
    )
    ctx = run_analysis(user_with_token, req)
    assert ctx.missing == ["IMA-B", "CDI + 2%"]


def test_run_analysis_propagates_gaps(user_with_token, monkeypatch):
    from apps.market.gaps import SeriesGap

    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    us = pd.Series([100.0, 101.0], index=idx)
    pdf = pd.DataFrame(
        {
            "date": idx,
            "vlr_mercado": [100.0, 101.0],
            "vlr_investido": [100.0, 100.0],
            "proventos": [0.0, 0.0],
            "fluxo": [100.0, 0.0],
        }
    )

    expected_gap = SeriesGap("CDI", dt.date(2024, 1, 2), dt.date(2024, 1, 3))
    monkeypatch.setattr(analysis, "fetch_user_portfolio", lambda *a, **k: (us, pdf))

    def fake_build(**kw):
        df = pd.DataFrame({"Carteira": [100.0, 101.0], "CDI": [100.0, 100.2]}, index=idx)
        return DatasetResult(df_combined=df, rf_series=None, portfolio_df=pdf, gaps=[expected_gap])

    monkeypatch.setattr(analysis, "build_dataset", fake_build)

    req = _req(start=dt.date(2024, 1, 2), end=dt.date(2024, 1, 3), benchmarks=["CDI"])
    ctx = run_analysis(user_with_token, req)
    assert ctx.gaps == [expected_gap]

