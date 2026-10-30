"""Cliente da API DLP (carteira pessoal). LIVE, por-usuário.

Porta de src/data/sources/market_data.py::buscar_historico / buscar_resumo_carteira.
NÃO grava em MarketSeries (é dado por-usuário). Cacheia no Redis com TTL 600s.
Levanta DLPAuthError (401/403), DLPUnavailableError (rede/5xx/JSON inválido) ou DLPError; devolve None só para 'sem dados'.
"""
from __future__ import annotations

import hashlib
import logging
import time

import pandas as pd
import requests
from django.conf import settings
from django.core.cache import cache

from apps.market import cache as mc

from .base import DLPAuthError, DLPError, DLPUnavailableError

logger = logging.getLogger(__name__)

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _norm_date_col(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: "date" for c in df.columns if c.lower() in ("date", "data")}
    return df.rename(columns=rename) if rename else df


def _gen_key(token: str) -> str:
    return mc.make_key("dlp", "gen", _token_hash(token))


def _generation(token: str) -> int:
    key = _gen_key(token)
    try:
        val = cache.get(key)
        if val is None:
            cache.add(key, time.time_ns(), timeout=None)
            val = cache.get(key)
        return int(val) if val is not None else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("dlp gen get falhou: %s", exc)
        return 0


def invalidate_user_cache(token: str) -> None:
    """Invalida hist/resumo do usuário incrementando a geração (vale em redis e locmem)."""
    key = _gen_key(token)
    try:
        cache.incr(key)
    except ValueError:
        try:
            cache.add(key, time.time_ns(), timeout=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dlp invalidate add falhou para %s: %s", _token_hash(token), exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("dlp invalidate falhou para %s: %s", _token_hash(token), exc)


def _get(path: str, token: str, params: dict) -> dict:
    try:
        resp = requests.get(
            f"{settings.DLP_BASE_URL}{path}",
            headers={"Content-Type": "application/json", "Authorization": token},
            params=params,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise DLPUnavailableError(f"DLP {path}: {type(exc).__name__}") from None
    status = resp.status_code
    if status in (401, 403):
        raise DLPAuthError(f"DLP {path}: HTTP {status}")
    if status >= 500:
        raise DLPUnavailableError(f"DLP {path}: HTTP {status}")
    if status >= 400:
        raise DLPError(f"DLP {path}: HTTP {status}")
    try:
        payload = resp.json()
    except ValueError:
        raise DLPUnavailableError(f"DLP {path}: resposta JSON inválida") from None
    if not isinstance(payload, dict):
        raise DLPUnavailableError(f"DLP {path}: resposta não é um objeto JSON")
    return payload


ERR_TTL = 30  # s — falha recente da DLP; o refresh (nova geração) libera nova tentativa


def _failed_recently(key: str) -> str | None:
    try:
        if cache.get(key + ":auth"):
            return "auth"
        if cache.get(key + ":err"):
            return "unavailable"
    except Exception as exc:  # noqa: BLE001
        logger.warning("dlp err get falhou: %s", exc)
    return None


def _mark_failed(key: str, kind: str = "unavailable") -> None:
    suffix = ":auth" if kind == "auth" else ":err"
    try:
        cache.set(key + suffix, 1, ERR_TTL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("dlp err set falhou: %s", exc)


def _raise_if_failed(failed: str | None) -> None:
    if failed == "auth":
        raise DLPAuthError("DLP autenticação rejeitada (falha recente)")
    if failed == "unavailable":
        raise DLPUnavailableError("DLP indisponível (falha recente)")


def _get_or_mark(key: str, path: str, token: str, params: dict) -> dict:
    """`_get` que lembra DLPUnavailableError e DLPAuthError por ERR_TTL (não serializa timeouts no single-flight)."""
    _raise_if_failed(_failed_recently(key))
    try:
        return _get(path, token, params)
    except DLPUnavailableError:
        _mark_failed(key, kind="unavailable")
        raise
    except DLPAuthError:
        _mark_failed(key, kind="auth")
        raise


_FILTER_COLUMNS = ("ativo", "classe", "corretora")


def _filter_historico(df: pd.DataFrame, **filters) -> pd.DataFrame | None:
    """Recorta o histórico completo em memória; None se nenhuma linha casar."""
    lower_map = {str(c).lower(): c for c in df.columns}
    mask = pd.Series(True, index=df.index)
    for name in _FILTER_COLUMNS:
        val = filters.get(name)
        if val is None or str(val).strip() == "":
            continue
        col = lower_map.get(name)
        if col is None:
            logger.warning("coluna %s ausente no /historico; nenhuma linha casa", name)
            return None
        mask &= df[col].notna() & (df[col].astype(str).str.strip() == str(val).strip())
    out = df[mask]
    if out.empty:
        return None
    return out.reset_index(drop=True)


def buscar_historico(token, ativo=None, classe=None, corretora=None):
    """Um único DataFrame por (usuário, geração) no cache; filtros aplicados em memória."""
    filters = {"ativo": ativo, "classe": classe, "corretora": corretora}
    key = mc.make_key("dlp", "hist", _token_hash(token), _generation(token))
    cached = mc.get_df(key)
    if cached is not None:
        return _filter_historico(_norm_date_col(cached), **filters)
    _raise_if_failed(_failed_recently(key))

    with mc.single_flight(key):
        cached = mc.get_df(key)
        if cached is not None:
            return _filter_historico(_norm_date_col(cached), **filters)
        _raise_if_failed(_failed_recently(key))
        payload = _get_or_mark(key, "/historico", token, {})
        df = pd.DataFrame(payload.get("historico", []))
        if df.empty:
            return None
        df = _norm_date_col(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        mc.set_df(key, df, ttl=mc.TTL_DLP)
        return _filter_historico(df, **filters)


def buscar_resumo_carteira(token):
    key = mc.make_key("dlp", "resumo", _token_hash(token), _generation(token))
    cached = mc.get_df(key)
    if cached is not None:
        return cached
    _raise_if_failed(_failed_recently(key))

    with mc.single_flight(key):
        cached = mc.get_df(key)
        if cached is not None:
            return cached
        _raise_if_failed(_failed_recently(key))
        payload = _get_or_mark(
            key, "/resumo", token,
            {"summary": "true", "wallet": "true", "multiwallet": "false", "notifications": "false"},
        )
        if not payload:
            return None
        mc.set_df(key, payload, ttl=mc.TTL_DLP)
        return payload
