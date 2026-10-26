from __future__ import annotations

from django.http import HttpRequest
from ninja import Router

from apps.api.analysis import AnalysisContext, run_analysis
from apps.api.errors import ApiError
from apps.api.schemas import (
    AnalysisRequest,
    RiskReturnOut,
    RollingMetricsOut,
    RollingRequest,
    SeriesListOut,
    TwrEvolutionOut,
    YearlySummaryOut,
)
from apps.api.serializers import (
    clean,
    frame_to_rows,
    frame_to_series_list,
    series_to_json,
    yearly_to_json,
)
from engine.compute import (
    compute_drawdown,
    compute_irr_evolution,
    compute_risk_return,
    compute_rolling_metrics,
    compute_shadow,
    compute_twr_evolution,
    compute_yearly_summary,
)

router = Router(tags=["portfolio"])

_RISK_RETURN_COLUMNS = {
    "Volatilidade": "volatilidade",
    "Retorno (CAGR)": "retorno_cagr",
    "Sharpe": "sharpe",
}


def _envelope(ctx: AnalysisContext, data: dict) -> dict:
    return {
        "data": data,
        "meta": {
            "start": ctx.start.isoformat(),
            "end": ctx.end.isoformat(),
            "missing": ctx.missing,
        },
    }


@router.post("/twr-evolution", response=TwrEvolutionOut)
def twr_evolution(request: HttpRequest, payload: AnalysisRequest):
    ctx = run_analysis(request.user, payload)
    out = compute_twr_evolution(ctx.df_combined)
    data = {
        "series": frame_to_series_list(out["series"]),
        "final_returns": {k: clean(v) for k, v in out["final_returns"].items()},
    }
    return _envelope(ctx, data)


@router.post("/drawdown", response=SeriesListOut)
def drawdown(request: HttpRequest, payload: AnalysisRequest):
    ctx = run_analysis(request.user, payload)
    df_dd = compute_drawdown(ctx.df_combined)
    return _envelope(ctx, {"series": frame_to_series_list(df_dd)})


@router.post("/rolling-metrics", response=RollingMetricsOut)
def rolling_metrics(request: HttpRequest, payload: RollingRequest):
    if not (21 <= payload.window <= 756):
        raise ApiError(
            422,
            "invalid_window",
            "A janela móvel deve estar entre 21 e 756 dias úteis.",
        )
    ctx = run_analysis(request.user, payload)
    out = compute_rolling_metrics(ctx.df_combined, ctx.selic, window=payload.window)
    data = {
        "volatility": frame_to_series_list(out["volatility"]),
        "sharpe": frame_to_series_list(out["sharpe"]),
    }
    return _envelope(ctx, data)


@router.post("/risk-return", response=RiskReturnOut)
def risk_return(request: HttpRequest, payload: AnalysisRequest):
    ctx = run_analysis(request.user, payload)
    df_rr = compute_risk_return(ctx.df_combined)
    return _envelope(ctx, {"rows": frame_to_rows(df_rr, _RISK_RETURN_COLUMNS)})


@router.post("/shadow", response=SeriesListOut)
def shadow(request: HttpRequest, payload: AnalysisRequest):
    ctx = run_analysis(request.user, payload)
    df_sw = compute_shadow(ctx.portfolio_df, ctx.df_combined)
    return _envelope(ctx, {"series": frame_to_series_list(df_sw)})


@router.post("/yearly-summary", response=YearlySummaryOut)
def yearly_summary(request: HttpRequest, payload: AnalysisRequest):
    ctx = run_analysis(request.user, payload)
    df_yr = compute_yearly_summary(ctx.df_combined)
    return _envelope(ctx, yearly_to_json(df_yr))


@router.post("/irr-evolution", response=SeriesListOut)
def irr_evolution(request: HttpRequest, payload: AnalysisRequest):
    ctx = run_analysis(request.user, payload)
    irr = compute_irr_evolution(ctx.portfolio_df)
    return _envelope(ctx, {"series": [series_to_json(irr, name="TIR")]})
