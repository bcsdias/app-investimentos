"""Montagem do dataset de análise — porta funcional de FinancialReport.

fetch_user_portfolio: DLP -> TWR da carteira.
build_dataset: Carteira + benchmarks -> df_combined (base 100), consumindo
apps.market.series (NÃO mais processar_benchmarks). Corrige o bug self.logger.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from apps.market import series, synthetics, transforms
from apps.market.services.sources import dlp
from apps.market.synthetics import SyntheticSpec
from engine.twr import calculate_twr

logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    df_combined: pd.DataFrame
    selic_series: pd.Series | None = None
    portfolio_df: pd.DataFrame | None = None


def fetch_user_portfolio(
    token: str, ativo=None, classe=None, start_date=None, end_date=None
) -> tuple[pd.Series | None, pd.DataFrame | None]:
    df = dlp.buscar_historico(token, ativo=ativo, classe=classe)
    if df is None or df.empty:
        logger.warning("DLP não retornou histórico para a carteira.")
        return None, None

    if "date" not in df.columns:
        for col in df.columns:
            if col.lower() in ("date", "data"):
                df = df.rename(columns={col: "date"})
                break
    if "date" not in df.columns:
        logger.error("Histórico DLP sem coluna 'date'.")
        return None, None

    df["date"] = pd.to_datetime(df["date"])
    grp = (
        df.groupby("date")[["vlr_mercado", "vlr_investido", "proventos"]]
        .sum()
        .reset_index()
    )
    grp["fluxo"] = (
        grp["vlr_investido"].diff().fillna(grp["vlr_investido"].iloc[0])
        - grp["proventos"]
    )

    if start_date:
        filt = grp[grp["date"] >= pd.to_datetime(start_date)].copy()
        if not filt.empty:
            gap = filt["vlr_mercado"].iloc[0] - filt["vlr_investido"].iloc[0]
            filt["vlr_investido"] = filt["vlr_investido"] + gap
        grp = filt

    twr = calculate_twr(grp, start_date, end_date)
    if twr.empty:
        return None, None
    return twr, grp


def build_dataset(
    user_series: pd.Series | None = None,
    benchmarks: list[str] | None = None,
    extra_specs: list[SyntheticSpec] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    portfolio_df: pd.DataFrame | None = None,
) -> DatasetResult:
    if user_series is not None and not user_series.empty:
        start = user_series.index.min().date()
        end = user_series.index.max().date()
    else:
        start = pd.to_datetime(start_date).date() if start_date else (pd.Timestamp.today() - pd.DateOffset(years=1)).date()
        end = pd.to_datetime(end_date).date() if end_date else pd.Timestamp.today().date()

    names = list(benchmarks or [])
    bench = series.get_many(names, start, end)
    selic = series.get_series("SELIC", start, end)

    frames: list[pd.Series] = []
    if user_series is not None and not user_series.empty:
        us = user_series.copy()
        us.name = "Carteira"
        us.index = pd.to_datetime(us.index)
        frames.append(us)

    for name in names:
        s = bench.get(name)
        if s is None or s.empty:
            continue
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        s.name = name
        frames.append(pd.to_numeric(s, errors="coerce"))

    for spec in extra_specs or []:
        base = series.get_series(spec.base, start, end)
        if base is None or base.empty:
            continue
        syn = transforms.synthetic_from_spec(spec, base)
        syn.name = synthetics.display_name(spec)
        syn.index = pd.to_datetime(syn.index)
        frames.append(syn)

    if not frames:
        logger.warning("Nenhum dado disponível para análise.")
        return DatasetResult(df_combined=pd.DataFrame(), selic_series=selic, portfolio_df=portfolio_df)

    raw = pd.concat(frames, axis=1).sort_index()
    if start_date and end_date:
        raw = raw.loc[str(start_date):str(end_date)]
    raw = raw.ffill()
    raw = raw.dropna(subset=["Carteira"]) if "Carteira" in raw.columns else raw.dropna(how="all")

    for col in raw.columns:
        raw[col] = transforms.rebase_100(raw[col])

    return DatasetResult(df_combined=raw, selic_series=selic, portfolio_df=portfolio_df)
