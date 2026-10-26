"""Cliente da API DLP (carteira pessoal). LIVE, por-usuário.

Porta de src/data/sources/market_data.py::buscar_historico / buscar_resumo_carteira.
NÃO grava em MarketSeries (é dado por-usuário). Cacheia no Redis com TTL 600s.
Levanta DLPAuthError (401/403), DLPUnavailableError (rede/5xx/JSON inválido) ou DLPError; devolve None só para 'sem dados'.
"""
from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd
import requests
from django.core.cache import cache

from apps.market import cache as mc

from .base import DLPAuthError, DLPError, DLPUnavailableError

logger = logging.getLogger(__name__)

_BASE = "https://users.dlombelloplanilhas.com"


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _norm_date_col(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: "date" for c in df.columns if c.lower() in ("date", "data")}
    return df.rename(columns=rename) if rename else df


def _gen_key(token: str) -> str:
    return mc.make_key("dlp", "gen", _token_hash(token))


def _generation(token: str) -> int:
    try:
        return int(cache.get(_gen_key(token), 0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("dlp gen get falhou: %s", exc)
        return 0


def invalidate_user_cache(token: str) -> None:
    """Invalida hist/resumo do usuário incrementando a geração (vale em redis e locmem)."""
    key = _gen_key(token)
    try:
        cache.add(key, 0, None)
        cache.incr(key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("dlp invalidate falhou para %s: %s", _token_hash(token), exc)


def _get(path: str, token: str, params: dict) -> dict:
    try:
        resp = requests.get(
            f"{_BASE}{path}",
            headers={"Content-Type": "application/json", "Authorization": token},
            params=params,
            timeout=60,
        )
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise DLPUnavailableError(f"DLP {path}: {type(exc).__name__}") from None
    status = resp.status_code
    if status in (401, 403):
        raise DLPAuthError(f"DLP {path}: HTTP {status}")
    if status >= 500:
        raise DLPUnavailableError(f"DLP {path}: HTTP {status}")
    if status >= 400:
        raise DLPError(f"DLP {path}: HTTP {status}")
    try:
        return resp.json()
    except ValueError:
        raise DLPUnavailableError(f"DLP {path}: resposta JSON inválida") from None


def buscar_historico(token, ativo=None, classe=None, corretora=None):
    params = {k: v for k, v in {"ativo": ativo, "classe": classe, "corretora": corretora}.items() if v}
    pstr = json.dumps(params, sort_keys=True)
    key = mc.make_key(
        "dlp", "hist", _token_hash(token), _generation(token),
        hashlib.sha256(pstr.encode()).hexdigest()[:12],
    )
    cached = mc.get_df(key)
    if cached is not None:
        return _norm_date_col(cached) if isinstance(cached, pd.DataFrame) else cached

    with mc.single_flight(key):
        cached = mc.get_df(key)
        if cached is not None:
            return _norm_date_col(cached) if isinstance(cached, pd.DataFrame) else cached
        payload = _get("/historico", token, params)
        df = pd.DataFrame(payload.get("historico", []))
        if df.empty:
            return None
        df = _norm_date_col(df)
        mc.set_df(key, df, ttl=mc.TTL_DLP)
        return df


def buscar_resumo_carteira(token):
    key = mc.make_key("dlp", "resumo", _token_hash(token), _generation(token))
    cached = mc.get_df(key)
    if cached is not None:
        return cached

    with mc.single_flight(key):
        cached = mc.get_df(key)
        if cached is not None:
            return cached
        payload = _get(
            "/resumo", token,
            {"summary": "true", "wallet": "true", "multiwallet": "false", "notifications": "false"},
        )
        if not payload:
            return None
        mc.set_df(key, payload, ttl=mc.TTL_DLP)
        return payload
