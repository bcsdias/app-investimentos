from __future__ import annotations

import datetime as dt

from ninja import Schema
from pydantic import Field


class TokenIn(Schema):
    token: str


class TokenStatusOut(Schema):
    configured: bool
    updated_at: dt.datetime | None = None


class SyntheticIn(Schema):
    base: str
    operador: str
    valor: float


class AnalysisRequest(Schema):
    start: dt.date | None = None
    end: dt.date | None = None
    ativo: str | None = None
    classe: str | None = None
    benchmarks: list[str] | None = None
    synthetics: list[SyntheticIn] = Field(default_factory=list)


class RollingRequest(AnalysisRequest):
    window: int = 252


class AnalysisMetaOut(Schema):
    start: str
    end: str
    missing: list[str]


class SerieOut(Schema):
    name: str
    data: list[list[str | float | None]]


class TwrEvolutionDataOut(Schema):
    series: list[SerieOut]
    final_returns: dict[str, float | None]


class TwrEvolutionOut(Schema):
    data: TwrEvolutionDataOut
    meta: AnalysisMetaOut


class SeriesListDataOut(Schema):
    series: list[SerieOut]


class SeriesListOut(Schema):
    data: SeriesListDataOut
    meta: AnalysisMetaOut


class RollingMetricsDataOut(Schema):
    volatility: list[SerieOut]
    sharpe: list[SerieOut]


class RollingMetricsOut(Schema):
    data: RollingMetricsDataOut
    meta: AnalysisMetaOut


class RiskReturnRowOut(Schema):
    name: str
    volatilidade: float | None = None
    retorno_cagr: float | None = None
    sharpe: float | None = None


class RiskReturnDataOut(Schema):
    rows: list[RiskReturnRowOut]


class RiskReturnOut(Schema):
    data: RiskReturnDataOut
    meta: AnalysisMetaOut


class YearlyRowOut(Schema):
    name: str
    values: dict[str, float | None]
    total: float | None = None


class YearlySummaryDataOut(Schema):
    years: list[int]
    rows: list[YearlyRowOut]


class YearlySummaryOut(Schema):
    data: YearlySummaryDataOut
    meta: AnalysisMetaOut


class SummaryDataOut(Schema):
    patrimonio_inicial: float | None = None
    aportes_periodo: float | None = None
    patrimonio: float | None = None
    ganho_periodo: float | None = None
    twr_total: float | None = None
    cagr: float | None = None
    volatilidade: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    xirr: float | None = None
    data_inicio: str
    data_fim: str


class SummaryOut(Schema):
    data: SummaryDataOut
    meta: AnalysisMetaOut


class PortfolioFiltersOut(Schema):
    ativos: list[str]
    classes: list[str]

