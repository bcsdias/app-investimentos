"""Derivações puras de séries de mercado. pd.Series/DataFrame entra, sai.
Zero I/O, zero Django. É a superfície que a paridade da Fase 6 valida contra o
processar_benchmarks legado.
"""
from __future__ import annotations

import pandas as pd

from apps.market.synthetics import SyntheticSpec


def accumulate(rate: pd.Series, periodicity: str = "daily") -> pd.Series:
    factor = (rate / 100.0) + 1.0
    return factor.cumprod().fillna(1.0)


def splice_proxy(bcb_level: pd.Series, etf_price: pd.Series) -> pd.Series:
    bcb_level = bcb_level.dropna().sort_index()
    etf_price = etf_price.dropna().sort_index()
    if bcb_level.empty:
        return etf_price
    if etf_price.empty:
        return bcb_level
    last_date = bcb_level.index[-1]
    tail = etf_price[etf_price.index >= last_date]
    if tail.empty:
        return bcb_level
    base = tail.iloc[0]
    projected = bcb_level.iloc[-1] * (tail.iloc[1:] / base)
    return pd.concat([bcb_level, projected]).sort_index()


def to_brl(usd_series: pd.Series, ptax: pd.Series) -> pd.Series:
    aligned = ptax.reindex(usd_series.index, method="ffill")
    return usd_series * aligned


def _steps_per_year(index: pd.Index) -> float:
    """Passos por ano da série base, para anualizar o spread do sintético '+'.
    Casa com o legado: IPCA (mensal) -> 12; benchmarks diários -> 252. Sem isto,
    um spread de 6% a.a. aplicado a cada ponto de uma série mensal renderia
    ~0.28% a.a. (1.06 ** (1/252) doze vezes) em vez de 6%.
    """
    if len(index) < 3:
        return 252.0
    gap_days = pd.Series(index).diff().dropna().dt.days.median()
    if gap_days >= 20:
        return 12.0
    if gap_days >= 5:
        return 52.0
    return 252.0


def synthetic_from_spec(spec: SyntheticSpec, base: pd.Series) -> pd.Series:
    ret = base.pct_change(fill_method=None).fillna(0)
    if spec.operador == "+":
        n = _steps_per_year(base.index)
        step = (1 + ret) * (1 + spec.valor / 100.0) ** (1 / n) - 1
        return (1 + step).cumprod() * 100
    if spec.operador == "%":
        return (1 + ret * (spec.valor / 100.0)).cumprod() * 100
    raise ValueError(f"operador sintético desconhecido: {spec.operador!r}")


def composite(members: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    parts = []
    for name, weight in weights.items():
        s = members[name]
        parts.append(s.pct_change(fill_method=None).fillna(0) * weight)
    combined = pd.concat(parts, axis=1).fillna(0).sum(axis=1)
    return (1 + combined).cumprod()


def rebase_100(series: pd.Series, at: pd.Timestamp | None = None) -> pd.Series:
    s = series if at is None else series[series.index >= at]
    first = s.first_valid_index()
    if first is None:
        return series
    base_val = s.loc[first]
    if base_val == 0:
        return series
    return (series / base_val) * 100
