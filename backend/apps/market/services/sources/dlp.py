"""Cliente da API DLP (carteira pessoal). LIVE, por-usuário.

Porta de src/data/sources/market_data.py::buscar_historico / buscar_resumo_carteira.
NÃO grava em MarketSeries (é dado por-usuário). Cacheia no Redis com TTL 600s.
Levanta DLPError em erro HTTP; devolve None só para 'sem dados'.
"""
from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd
import requests

from apps.market import cache as mc

from .base import DLPError

logger = logging.getLogger(__name__)

_BASE = "https://users.dlombelloplanilhas.com"


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def _norm_date_col(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: "date" for c in df.columns if c.lower() in ("date", "data")}
    return df.rename(columns=rename) if rename else df


def buscar_historico(token, ativo=None, classe=None, corretora=None):
    params = {k: v for k, v in {"ativo": ativo, "classe": classe, "corretora": corretora}.items() if v}
    pstr = json.dumps(params, sort_keys=True)
    key = mc.make_key("dlp", "hist", _token_hash(token), hashlib.sha256(pstr.encode()).hexdigest()[:12])
    cached = mc.get_df(key)
    if cached is not None:
        return _norm_date_col(cached) if isinstance(cached, pd.DataFrame) else cached

    try:
        resp = requests.get(
            f"{_BASE}/historico",
            headers={"Content-Type": "application/json", "Authorization": token},
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise DLPError(f"DLP /historico: {exc}") from exc

    df = pd.DataFrame(payload.get("historico", []))
    if df.empty:
        return None
    df = _norm_date_col(df)
    mc.set_df(key, df, ttl=mc.TTL_DLP)
    return df


def buscar_resumo_carteira(token):
    key = mc.make_key("dlp", "resumo", _token_hash(token))
    cached = mc.get_df(key)
    if cached is not None:
        return cached

    try:
        resp = requests.get(
            f"{_BASE}/resumo",
            headers={"Content-Type": "application/json", "Authorization": token},
            params={"summary": "true", "wallet": "true", "multiwallet": "false", "notifications": "false"},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise DLPError(f"DLP /resumo: {exc}") from exc

    if not payload:
        return None
    mc.set_df(key, payload, ttl=mc.TTL_DLP)
    return payload
