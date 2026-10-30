"""Contrato comum das fontes de dados de mercado.

Regra de ouro: uma fonte SEMPRE levanta FetchError/DLPError em falha de
rede/parse. Devolver [] / None significa APENAS 'a fonte não tem esse dado' —
nunca 'deu erro'.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class RawPoint:
    date: dt.date
    value: float


class FetchError(Exception):
    """Falha ao buscar dados de uma fonte de mercado (rede, parse, formato)."""


class DLPError(Exception):
    """Falha na API DLP (por-usuário)."""


class DLPAuthError(DLPError):
    """Token DLP rejeitado (HTTP 401/403)."""


class DLPUnavailableError(DLPError):
    """DLP indisponível: timeout, rede, 5xx ou resposta inválida."""
