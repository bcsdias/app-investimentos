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
