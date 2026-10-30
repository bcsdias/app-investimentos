"""Montagem do dataset de análise — porta funcional de FinancialReport.

fetch_user_portfolio: DLP -> TWR da carteira.
build_dataset: Carteira + benchmarks -> df_combined (base 100), consumindo
apps.market.series (NÃO mais processar_benchmarks). Corrige o bug self.logger.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import pandas as pd

from apps.market import catalog, gaps, series, synthetics, transforms
from apps.market.gaps import SeriesGap
from apps.market.services.sources import dlp
from apps.market.synthetics import SyntheticSpec
from engine import flows
from engine.twr import calculate_twr

logger = logging.getLogger(__name__)

BENCH_LOOKBACK_DAYS = 40


def _today() -> dt.date:
    return dt.date.today()


def _as_date(d: dt.date | dt.datetime | pd.Timestamp | str) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return pd.to_datetime(d).date()


@dataclass
class DatasetResult:
    df_combined: pd.DataFrame
    rf_series: pd.Series | None = None
    portfolio_df: pd.DataFrame | None = None
    gaps: list[SeriesGap] = field(default_factory=list)



def fetch_user_portfolio(
    token: str, ativo=None, classe=None, start_date=None, end_date=None
) -> tuple[pd.Series | None, pd.DataFrame | None]:
    df = dlp.buscar_historico(token, ativo=ativo, classe=classe)
    if df is None or df.empty:
        logger.warning("DLP não retornou histórico para a carteira.")
        return None, None

    grp = flows.group_portfolio(df)
    if grp.empty or "date" not in grp.columns:
        logger.error("Histórico DLP sem coluna 'date' ou sem dados válidos.")
        return None, None

    grp["fluxo"] = flows.external_flows(grp)

    in_window = pd.Series(True, index=grp.index)
    if start_date:
        in_window &= grp["date"] >= pd.to_datetime(start_date)
    if end_date:
        in_window &= grp["date"] <= pd.to_datetime(end_date)
    if not in_window.any():
        return None, None

    # Cada linha é um fechamento mensal: o período parte do último fechamento <= start_date
    # (patrimônio de abertura), para o retorno do 1º mês do recorte entrar no TWR e no XIRR.
    has_gap = False
    if start_date:
        start_ts = pd.to_datetime(start_date)
        at_or_before = grp.index[grp["date"] <= start_ts]
        if len(at_or_before):
            in_window.loc[at_or_before[-1]] = True
        has_gap = (grp["date"] < grp.loc[in_window, "date"].iloc[0]).any()

    grp = grp[in_window]

    grp = grp.reset_index(drop=True)

    if has_gap:
        gap = grp["vlr_mercado"].iloc[0] - grp["vlr_investido"].iloc[0]
        grp["vlr_investido"] = grp["vlr_investido"] + gap
        grp.loc[0, "fluxo"] = float(grp["vlr_mercado"].iloc[0])

    twr = calculate_twr(grp)
    if twr is None or twr.empty:
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

    crop_start = pd.to_datetime(start_date) if start_date else pd.to_datetime(start)
    crop_end = pd.to_datetime(end_date) if end_date else pd.to_datetime(end)
    if user_series is not None and not user_series.empty:
        crop_start = min(crop_start, pd.to_datetime(user_series.index.min()))
        crop_end = min(crop_end, pd.to_datetime(user_series.index.max()))

    crop_start_d = _as_date(crop_start)
    crop_end_d = _as_date(crop_end)

    today = _today()
    bench_end = max(end, today)
    bench_start = start - dt.timedelta(days=BENCH_LOOKBACK_DAYS)
    crop_start_ts = pd.to_datetime(crop_start)

    names = list(benchmarks or [])
    bench = series.get_many(names, bench_start, bench_end)
    rf = series.get_series("CDI", start, end)

    frames: list[pd.Series] = []
    if user_series is not None and not user_series.empty:
        us = user_series.copy()
        us.name = "Carteira"
        us.index = pd.to_datetime(us.index)
        frames.append(us)

    for name in names:
        s = bench.get(name)
        if s is None or s.empty or not (pd.to_datetime(s.index) <= crop_start_ts).any():
            s_epoch = series.get_series(name, catalog.EPOCH, bench_end)
            if s_epoch is not None and not s_epoch.empty:
                s = s_epoch
        if s is None or s.empty:
            continue
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        s.name = name
        frames.append(pd.to_numeric(s, errors="coerce"))

    for spec in extra_specs or []:
        base = series.get_series(spec.base, bench_start, bench_end)
        if base is None or base.empty or not (pd.to_datetime(base.index) <= crop_start_ts).any():
            base_epoch = series.get_series(spec.base, catalog.EPOCH, bench_end)
            if base_epoch is not None and not base_epoch.empty:
                base = base_epoch
        if base is None or base.empty:
            continue
        syn = transforms.synthetic_from_spec(spec, base)
        syn.name = synthetics.display_name(spec)
        syn.index = pd.to_datetime(syn.index)
        frames.append(syn)

    if not frames:
        logger.warning("Nenhum dado disponível para análise.")
        return DatasetResult(df_combined=pd.DataFrame(), rf_series=rf, portfolio_df=portfolio_df)

    gaps_list: list[SeriesGap] = []
    tail_until = min(crop_end_d, today)
    for f in frames:
        if f.name == "Carteira":
            continue
        periodicity = gaps.infer_periodicity(f.index)
        for g in gaps.find_gaps(f.index, periodicity):
            if g.end > crop_start_d and g.start < crop_end_d:
                gaps_list.append(SeriesGap(series=str(f.name), start=g.start, end=g.end))
        tail = gaps.stale_tail(f.index, periodicity, until=tail_until)
        if tail is not None and tail.end > crop_start_d and tail.start < crop_end_d:
            gaps_list.append(SeriesGap(series=str(f.name), start=tail.start, end=tail.end))

    raw = pd.concat(frames, axis=1).sort_index()
    raw = raw.ffill()

    raw = raw.loc[crop_start:crop_end]

    raw = raw.dropna(subset=["Carteira"]) if "Carteira" in raw.columns else raw.dropna(how="all")

    for col in raw.columns:
        raw[col] = transforms.rebase_100(raw[col])

    return DatasetResult(df_combined=raw, rf_series=rf, portfolio_df=portfolio_df, gaps=gaps_list)
