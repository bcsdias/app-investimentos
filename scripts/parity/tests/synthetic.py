"""Dados sintéticos determinísticos para os testes do harness (sem DLP, sem Postgres)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from parity.frozen import MarketSnapshot

TODAY = dt.date(2025, 6, 30)

_MARKET_DAYS = pd.bdate_range("2023-01-02", "2025-06-30")
_HIST_DAYS = pd.bdate_range("2023-03-01", "2025-06-27")


def _walk(rng: np.random.Generator, start: float, n: int, vol: float) -> np.ndarray:
    return start * np.cumprod(1 + rng.normal(0.0003, vol, n))


def _drop_some(s: pd.Series, rng: np.random.Generator, frac: float) -> pd.Series:
    keep = rng.random(len(s)) >= frac
    keep[0] = True
    return s[keep]


def make_snapshot() -> MarketSnapshot:
    rng = np.random.default_rng(42)
    n = len(_MARKET_DAYS)
    monthly = pd.date_range("2023-01-01", "2025-06-01", freq="MS")
    series = {
        "bcb:11": pd.Series(0.04 + rng.normal(0, 0.002, n), index=_MARKET_DAYS),
        "bcb:12": pd.Series(0.0395 + rng.normal(0, 0.002, n), index=_MARKET_DAYS),
        "bcb:433": pd.Series(0.4 + rng.normal(0, 0.15, len(monthly)), index=monthly),
        # Calendários próprios (feriados) para exercitar o ffill do alinhamento.
        "yf:^GSPC": _drop_some(pd.Series(_walk(rng, 4000.0, n, 0.01), index=_MARKET_DAYS), rng, 0.04),
        "ptax:USDBRL": _drop_some(pd.Series(_walk(rng, 5.0, n, 0.006), index=_MARKET_DAYS), rng, 0.02),
        "b3:IFIX": _drop_some(pd.Series(_walk(rng, 3000.0, n, 0.005), index=_MARKET_DAYS), rng, 0.03),
    }
    for key, s in series.items():
        s.name = key
    return MarketSnapshot(series=series)


def make_hist() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frames = []
    specs = [
        ("PETR4", "XP", "Ações", 10_000.0, 0.012),
        ("CDB X", "BTG", "Renda Fixa", 20_000.0, 0.0005),
    ]
    aportes = {pd.Timestamp("2023-06-01"): 2_000.0, pd.Timestamp("2024-02-01"): 3_000.0,
               pd.Timestamp("2025-01-02"): 1_500.0}
    proventos = {pd.Timestamp("2023-09-15"): 120.0, pd.Timestamp("2024-09-16"): 150.0}
    for ativo, corretora, classe, inicial, vol in specs:
        ret = 1 + rng.normal(0.0004, vol, len(_HIST_DAYS))
        investido = np.full(len(_HIST_DAYS), inicial)
        mercado = np.empty(len(_HIST_DAYS))
        prov = np.zeros(len(_HIST_DAYS))
        level = inicial
        extra = 0.0
        for i, d in enumerate(_HIST_DAYS):
            level *= ret[i]
            if d in aportes:
                extra += aportes[d]
                level += aportes[d]
            if d in proventos and classe == "Ações":
                prov[i] = proventos[d]
            investido[i] = inicial + extra
            mercado[i] = level
        frames.append(
            pd.DataFrame(
                {
                    "date": _HIST_DAYS.strftime("%Y-%m-%d"),
                    "ativo": ativo,
                    "corretora": corretora,
                    "classe": classe,
                    "vlr_mercado": mercado,
                    "vlr_investido": investido,
                    "proventos": prov,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
