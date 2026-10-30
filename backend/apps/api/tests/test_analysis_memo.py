import datetime as dt
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from apps.api import analysis
from apps.api.analysis import run_analysis
from apps.api.errors import ApiError
from apps.api.schemas import AnalysisRequest, RollingRequest
from apps.core.models import UserToken
from apps.market import cache as mc
from apps.market.services.sources import dlp


@pytest.fixture
def counting_fetch(monkeypatch, portfolio_env):
    calls = []
    real_fetch = analysis.fetch_user_portfolio

    def fake_fetch(*args, **kwargs):
        token = args[0] if args else kwargs.get("token")
        calls.append((token, args, kwargs))
        twr, pdf = real_fetch(*args, **kwargs)
        if token == "tok-other":
            # Carteira diferente para other_user
            twr = twr * 1.5
            pdf = pdf.copy()
            pdf["vlr_mercado"] = pdf["vlr_mercado"] * 1.5
        return twr, pdf

    monkeypatch.setattr("apps.api.analysis.fetch_user_portfolio", fake_fetch)
    return calls


def _req(**kwargs):
    defaults = {
        "start": "2024-03-01",
        "end": "2025-06-30",
        "ativo": None,
        "classe": None,
        "benchmarks": ["CDI", "BOVA11"],
        "synthetics": [],
    }
    defaults.update(kwargs)
    return AnalysisRequest(**defaults)


def test_same_request_computes_once(user_with_token, counting_fetch):
    req = _req()
    ctx1 = run_analysis(user_with_token, req)
    assert len(counting_fetch) == 1

    ctx2 = run_analysis(user_with_token, req)
    assert len(counting_fetch) == 1

    assert_frame_equal(ctx1.df_combined, ctx2.df_combined, check_exact=True)
    assert_frame_equal(ctx1.portfolio_df, ctx2.portfolio_df, check_exact=True)
    if ctx1.rf is not None or ctx2.rf is not None:
        assert_series_equal(ctx1.rf, ctx2.rf, check_exact=True)
    assert ctx1.gaps == ctx2.gaps
    assert ctx1.missing == ctx2.missing
    assert ctx1.start == ctx2.start
    assert ctx1.end == ctx2.end


def test_rolling_window_shares_context(user_with_token, counting_fetch):
    req1 = _req()
    req2 = RollingRequest(
        start=req1.start,
        end=req1.end,
        ativo=req1.ativo,
        classe=req1.classe,
        benchmarks=req1.benchmarks,
        synthetics=req1.synthetics,
        window=63,
    )
    ctx1 = run_analysis(user_with_token, req1)
    assert len(counting_fetch) == 1

    ctx2 = run_analysis(user_with_token, req2)
    assert len(counting_fetch) == 1


def test_different_request_recomputes(user_with_token, counting_fetch):
    # Benchmarks diferentes
    run_analysis(user_with_token, _req(benchmarks=["CDI"]))
    assert len(counting_fetch) == 1
    run_analysis(user_with_token, _req(benchmarks=["CDI", "BOVA11"]))
    assert len(counting_fetch) == 2

    # Ordem de benchmarks diferente
    run_analysis(user_with_token, _req(benchmarks=["BOVA11", "CDI"]))
    assert len(counting_fetch) == 3

    # Classe diferente
    run_analysis(user_with_token, _req(benchmarks=["CDI"], classe="Ações"))
    assert len(counting_fetch) == 4


def test_users_never_share_context(user_with_token, other_user, counting_fetch):
    # Configura token para other_user
    ut = UserToken(user=other_user)
    ut.set_token("tok-other")
    ut.save()

    req = _req()
    ctx1 = run_analysis(user_with_token, req)
    assert len(counting_fetch) == 1

    ctx2 = run_analysis(other_user, req)
    assert len(counting_fetch) == 2

    # Carteiras são distintas entre usuários
    assert not ctx1.portfolio_df["vlr_mercado"].equals(ctx2.portfolio_df["vlr_mercado"])


def test_dlp_refresh_invalidates(user_with_token, counting_fetch):
    req = _req()
    run_analysis(user_with_token, req)
    assert len(counting_fetch) == 1

    token = analysis.get_token(user_with_token)
    dlp.invalidate_user_cache(token)

    run_analysis(user_with_token, req)
    assert len(counting_fetch) == 2


def test_series_generation_bump_invalidates(user_with_token, counting_fetch):
    req = _req()
    run_analysis(user_with_token, req)
    assert len(counting_fetch) == 1

    mc.bump_series_generation()

    run_analysis(user_with_token, req)
    assert len(counting_fetch) == 2


def test_day_rollover_invalidates(user_with_token, counting_fetch, monkeypatch):
    day1 = dt.date(2025, 6, 30)
    day2 = dt.date(2025, 7, 1)

    current_day = [day1]

    monkeypatch.setattr(analysis, "_today", lambda: current_day[0])

    req = _req()
    run_analysis(user_with_token, req)
    assert len(counting_fetch) == 1

    # Próximo dia
    current_day[0] = day2
    run_analysis(user_with_token, req)
    assert len(counting_fetch) == 2


def test_cache_failure_falls_back_to_compute(user_with_token, counting_fetch, monkeypatch):
    req = _req()

    def boom(*args, **kwargs):
        raise ConnectionError("Redis is down")

    monkeypatch.setattr(analysis.cache, "get", boom)
    monkeypatch.setattr(analysis.cache, "set", boom)

    ctx1 = run_analysis(user_with_token, req)
    assert len(counting_fetch) == 1
    assert ctx1 is not None

    ctx2 = run_analysis(user_with_token, req)
    assert len(counting_fetch) == 2
    assert ctx2 is not None


def test_api_error_not_memoized(user_with_token, monkeypatch, portfolio_env):
    attempts = [0]
    real_fetch = analysis.fetch_user_portfolio

    def faulty_fetch(*args, **kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            return None, None
        return real_fetch(*args, **kwargs)

    monkeypatch.setattr("apps.api.analysis.fetch_user_portfolio", faulty_fetch)

    req = _req()
    # 1ª tentativa levanta ApiError 404
    with pytest.raises(ApiError) as exc_info:
        run_analysis(user_with_token, req)
    assert exc_info.value.status == 404
    assert exc_info.value.code == "portfolio_empty"

    # 2ª tentativa devolve contexto (contador 2)
    ctx = run_analysis(user_with_token, req)
    assert ctx is not None
    assert attempts[0] == 2
