from __future__ import annotations

import logging

import pandas as pd
from django.http import HttpRequest
from ninja import Router, Status

from apps.api.analysis import AnalysisContext, dlp_errors, get_token, run_analysis
from apps.api.errors import ApiError
from apps.api.schemas import (
    AnalysisRequest,
    PortfolioFiltersOut,
    RiskReturnOut,
    RollingMetricsOut,
    RollingRequest,
    SeriesListOut,
    SummaryOut,
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
from apps.market.services.sources import dlp
from engine.compute import (
    compute_drawdown,
    compute_irr_evolution,
    compute_risk_return,
    compute_rolling_metrics,
    compute_shadow,
    compute_summary_kpis,
    compute_twr_evolution,
    compute_yearly_summary,
)

router = Router(tags=["portfolio"])
logger = logging.getLogger(__name__)

ASSET_COLUMNS = ("ativo", "ticker", "codigo")
CLASS_COLUMNS = ("classe", "tipo")

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
            "gaps": [
                {"series": g.series, "start": str(g.start), "end": str(g.end)}
                for g in getattr(ctx, "gaps", [])
            ],
        },
    }


def _extract_distinct(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> list[str]:
    lower_map = {str(col).lower(): col for col in df.columns}
    for cand in candidates:
        col = lower_map.get(cand.lower())
        if col is not None:
            raw_unique = pd.unique(df[col].dropna())
            vals = (str(v).strip() for v in raw_unique)
            return sorted({v for v in vals if v})
    logger.warning(
        "Nenhuma coluna de %s (%s) encontrada no histórico DLP: %s",
        label,
        candidates,
        list(df.columns),
    )
    return []


@router.get("/filters", response=PortfolioFiltersOut)
def filters(request: HttpRequest):
    token = get_token(request.user)
    with dlp_errors():
        df = dlp.buscar_historico(token)
    if df is None or df.empty:
        return {"ativos": [], "classes": []}
    return {
        "ativos": _extract_distinct(df, ASSET_COLUMNS, "ativo"),
        "classes": _extract_distinct(df, CLASS_COLUMNS, "classe"),
    }


@router.post("/refresh", response={204: None})
def refresh(request: HttpRequest):
    token = get_token(request.user)
    dlp.invalidate_user_cache(token)
    return Status(204, None)


@router.post("/summary", response=SummaryOut)
def summary(request: HttpRequest, payload: AnalysisRequest):
    ctx = run_analysis(request.user, payload)
    pdf = ctx.portfolio_df

    summary_kpis = compute_summary_kpis(pdf)

    twr_out = compute_twr_evolution(ctx.df_combined)
    twr_total = clean(twr_out["final_returns"].get("Carteira"))

    rr_df = compute_risk_return(ctx.df_combined, ctx.rf)
    if "Carteira" in rr_df.index:
        cagr = clean(rr_df.loc["Carteira", "Retorno (CAGR)"])
        vol = clean(rr_df.loc["Carteira", "Volatilidade"])
        sharpe = clean(rr_df.loc["Carteira", "Sharpe"])
    else:
        cagr = vol = sharpe = None

    dd_df = compute_drawdown(ctx.df_combined)
    max_dd = clean(dd_df["Carteira"].min()) if "Carteira" in dd_df.columns else None

    irr = compute_irr_evolution(pdf)
    xirr = clean(irr.iloc[-1]) if not irr.empty else None

    data = {
        "patrimonio_inicial": clean(summary_kpis["patrimonio_inicial"]),
        "aportes_periodo": clean(summary_kpis["aportes_periodo"]),
        "patrimonio": clean(summary_kpis["patrimonio"]),
        "ganho_periodo": clean(summary_kpis["ganho_periodo"]),
        "twr_total": twr_total,
        "cagr": cagr,
        "volatilidade": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "xirr": xirr,
        "data_inicio": ctx.start.isoformat(),
        "data_fim": ctx.end.isoformat(),
    }
    return _envelope(ctx, data)


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
    out = compute_rolling_metrics(ctx.df_combined, ctx.rf, window=payload.window)
    data = {
        "volatility": frame_to_series_list(out["volatility"]),
        "sharpe": frame_to_series_list(out["sharpe"]),
    }
    return _envelope(ctx, data)


@router.post("/risk-return", response=RiskReturnOut)
def risk_return(request: HttpRequest, payload: AnalysisRequest):
    ctx = run_analysis(request.user, payload)
    df_rr = compute_risk_return(ctx.df_combined, ctx.rf)
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
