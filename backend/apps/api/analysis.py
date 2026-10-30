"""Validação do AnalysisRequest, leitura do token DLP e orquestração da análise."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
from dataclasses import dataclass, field

import pandas as pd
from cryptography.fernet import InvalidToken
from django.core.cache import cache

from apps.api.errors import ApiError
from apps.api.schemas import AnalysisRequest
from apps.core.models import UserToken
from apps.market import assets, catalog, synthetics
from apps.market import cache as mc
from apps.market.gaps import SeriesGap
from apps.market.services.sources import dlp
from apps.market.services.sources.base import DLPAuthError, DLPError, DLPUnavailableError
from apps.market.synthetics import SyntheticSpec
from engine.dataset import build_dataset, fetch_user_portfolio

logger = logging.getLogger(__name__)

MAX_ITEMS = 15
_SYNTHETIC_BASES = {"IPCA", "CDI"}
ANALYSIS_TTL = 300  # s (menor que TTL_DLP = 600s)


def _today() -> dt.date:
    return dt.date.today()


@dataclass(frozen=True)
class ValidRequest:
    start: dt.date
    end: dt.date
    benchmarks: list[str]
    specs: list[SyntheticSpec]


@dataclass
class AnalysisContext:
    df_combined: pd.DataFrame
    portfolio_df: pd.DataFrame
    rf: pd.Series | None
    start: dt.date
    end: dt.date
    missing: list[str]
    gaps: list[SeriesGap] = field(default_factory=list)



def _one_year_ago(d: dt.date) -> dt.date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:
        return d.replace(year=d.year - 1, day=28)


def _is_known_benchmark(name: str) -> bool:
    try:
        catalog.resolve(name)
        return True
    except KeyError:
        return assets.resolve(name) is not None


def _validate_synthetic(s) -> SyntheticSpec:
    if s.base not in _SYNTHETIC_BASES or s.operador not in {"+", "%"}:
        raise ApiError(
            422,
            "invalid_synthetic",
            "Sintético inválido: use base IPCA ou CDI e operador '+' ou '%'.",
        )
    if s.operador == "+" and not (0 < s.valor <= 50):
        raise ApiError(
            422,
            "invalid_synthetic",
            "Para operador '+', o valor deve estar entre 0 (exclusivo) e 50.",
        )
    if s.operador == "%" and not (0 < s.valor <= 500):
        raise ApiError(
            422,
            "invalid_synthetic",
            "Para operador '%', o valor deve estar entre 0 (exclusivo) e 500.",
        )
    return SyntheticSpec(s.base, s.operador, float(s.valor))


def validate_request(req: AnalysisRequest, today: dt.date | None = None) -> ValidRequest:
    ref_today = today or _today()
    end = req.end or ref_today
    start = req.start or _one_year_ago(end)
    if start > end:
        raise ApiError(
            422,
            "invalid_period",
            "A data inicial não pode ser posterior à data final.",
        )

    benchmarks = (
        list(catalog.DEFAULT_BENCHMARKS) if req.benchmarks is None else list(req.benchmarks)
    )
    specs = [_validate_synthetic(s) for s in req.synthetics]

    if len(benchmarks) + len(specs) > MAX_ITEMS:
        raise ApiError(
            422,
            "too_many_benchmarks",
            f"O limite é de {MAX_ITEMS} benchmarks + sintéticos por análise.",
        )

    invalid = [name for name in benchmarks if not _is_known_benchmark(name)]
    if invalid:
        raise ApiError(
            422,
            "unknown_benchmark",
            "Um ou mais benchmarks informados são desconhecidos.",
            details={"invalid": invalid},
        )

    return ValidRequest(start=start, end=end, benchmarks=benchmarks, specs=specs)


def get_token(user) -> str:
    ut = UserToken.objects.filter(user=user).first()
    if ut is None:
        raise ApiError(409, "dlp_token_missing", "Cadastre seu token DLP.")
    try:
        token = ut.get_token()
    except InvalidToken as exc:
        raise ApiError(409, "dlp_token_missing", "Cadastre seu token DLP.") from exc
    if not token:
        raise ApiError(409, "dlp_token_missing", "Cadastre seu token DLP.")
    return token


@contextlib.contextmanager
def dlp_errors():
    try:
        yield
    except DLPAuthError as exc:
        raise ApiError(
            422,
            "dlp_token_rejected",
            "Token DLP rejeitado pela API. Verifique e cadastre novamente.",
        ) from exc
    except (DLPUnavailableError, DLPError) as exc:
        raise ApiError(
            502,
            "dlp_unavailable",
            "Serviço DLP indisponível no momento. Tente novamente em instantes.",
        ) from exc


def _request_digest(valid: ValidRequest, req: AnalysisRequest) -> str:
    """Digest SHA-256 (16 hex) das partes do pedido que determinam o contexto."""
    payload = {
        "start": valid.start.isoformat(),
        "end": valid.end.isoformat(),
        "ativo": req.ativo,
        "classe": req.classe,
        "benchmarks": valid.benchmarks,
        "synthetics": [[s.base, s.operador, s.valor] for s in valid.specs],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _analysis_key(user, token: str, valid: ValidRequest, req: AnalysisRequest) -> str:
    return mc.make_key(
        "analysis",
        user.pk,
        dlp.user_cache_version(token),
        mc.series_generation(),
        _today().isoformat(),
        _request_digest(valid, req),
    )


def _compute_analysis(valid: ValidRequest, req: AnalysisRequest, token: str) -> AnalysisContext:
    with dlp_errors():
        user_series, portfolio_df = fetch_user_portfolio(
            token,
            ativo=req.ativo,
            classe=req.classe,
            start_date=valid.start.isoformat(),
            end_date=valid.end.isoformat(),
        )

    if user_series is None or portfolio_df is None or user_series.empty or portfolio_df.empty:
        raise ApiError(
            404,
            "portfolio_empty",
            "Nenhum histórico encontrado para a carteira no período ou filtro informado.",
        )

    res = build_dataset(
        user_series=user_series,
        benchmarks=valid.benchmarks,
        extra_specs=valid.specs,
        start_date=valid.start.isoformat(),
        end_date=valid.end.isoformat(),
        portfolio_df=portfolio_df,
    )
    df = res.df_combined
    if df.empty:
        raise ApiError(
            404,
            "portfolio_empty",
            "Nenhum histórico encontrado para a carteira no período ou filtro informado.",
        )

    effective_start = pd.Timestamp(df.index[0]).date()
    effective_end = pd.Timestamp(df.index[-1]).date()

    present = set(df.columns)
    expected = [*valid.benchmarks, *(synthetics.display_name(s) for s in valid.specs)]
    missing = [name for name in expected if name not in present]

    return AnalysisContext(
        df_combined=df,
        portfolio_df=res.portfolio_df,
        rf=res.rf_series,
        start=effective_start,
        end=effective_end,
        missing=missing,
        gaps=res.gaps,
    )


def run_analysis(user, req: AnalysisRequest) -> AnalysisContext:
    valid = validate_request(req)
    token = get_token(user)
    key = _analysis_key(user, token, valid, req)

    try:
        cached = cache.get(key)
        if isinstance(cached, AnalysisContext):
            return cached
    except Exception as exc:  # noqa: BLE001
        logger.warning("analysis cache get falhou: %s", exc)

    with mc.single_flight(key):
        try:
            cached = cache.get(key)
            if isinstance(cached, AnalysisContext):
                return cached
        except Exception as exc:  # noqa: BLE001
            logger.warning("analysis cache get single-flight falhou: %s", exc)

        ctx = _compute_analysis(valid, req, token)

        try:
            cache.set(key, ctx, timeout=ANALYSIS_TTL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("analysis cache set falhou: %s", exc)

        return ctx

