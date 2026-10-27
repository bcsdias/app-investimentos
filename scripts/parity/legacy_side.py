"""Lado legado: FinancialReport do src/ sem UI, com mercado congelado (camada 1)
ou ao vivo (camada 2).

Importa src: sai junto com o legado na Task 9 da Fase 6.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
from unittest import mock

import matplotlib.pyplot as plt
import pandas as pd

from apps.market.catalog import EPOCH
from parity.cases import Case
from parity.compare import (
    ADR0012_METRICS,
    FALHA,
    INVESTIGAR,
    OK,
    REL_TOL_L1,
    REL_TOL_L2,
    SeriesDiff,
    apply_adr0012_recheck,
    classify_origin,
    compare_frames,
)
from parity.engine_side import METRICS, irr_frame, normalize_yearly, portfolio_frame, run_engine
from parity.frozen import MarketSnapshot
from parity.kpis import derive_kpis
from parity.live import ApiClient, ApiError, api_outputs

# Cópia de src/data/sources/market_data.py:397 (codigos_indices): séries do BCB que
# já vêm como número-índice e não como taxa.
LEGACY_INDEX_CODES = (12466, 12467, 12468, 12461, 12463, 12464, 12469, 12462)

_FAR_FUTURE = dt.date(2200, 1, 1)
_MD = "src.data.sources.market_data"


def _none_if_empty(s: pd.Series) -> pd.Series | None:
    return None if s is None or s.empty else s


def _between(s: pd.Series, start, end) -> pd.Series:
    mask = (s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))
    return s.loc[mask]


def _fake_bcb(snapshot: MarketSnapshot):
    # Caminho de cache hit do legado: série cheia desde 1995, depois fatiada.
    def buscar_dados_bcb(codigo_bcb, start_date, end_date, nome=None):
        s = snapshot.load(f"bcb:{codigo_bcb}", EPOCH, _FAR_FUTURE)
        if s.empty:
            return None
        if codigo_bcb in LEGACY_INDEX_CODES:
            s = s.ffill()
        else:
            s = ((s / 100) + 1).cumprod().fillna(1)
        s.index.name = "Date"
        return _none_if_empty(s.loc[start_date:end_date])

    return buscar_dados_bcb


def _fake_level(snapshot: MarketSnapshot, prefix: str):
    def fetch(ident, start_date, end_date, nome=None):
        s = snapshot.load(f"{prefix}:{ident}", EPOCH, _FAR_FUTURE)
        return _none_if_empty(_between(s, start_date, end_date))

    return fetch


def _fake_dolar(snapshot: MarketSnapshot):
    def buscar_dolar_bcb(start_date, end_date):
        return _none_if_empty(snapshot.load("ptax:USDBRL", pd.Timestamp(start_date), pd.Timestamp(end_date)))

    return buscar_dolar_bcb


@contextlib.contextmanager
def legacy_patches(snapshot: MarketSnapshot, hist: pd.DataFrame):
    # Importar antes de qualquer patch: se o mock importasse financial_report já com
    # market_data.buscar_historico trocado, o `from ... import` capturaria o fake e a
    # restauração devolveria o fake (a camada 2 usaria o histórico congelado).
    import src.engine.financial_report  # noqa: F401

    hist_fake = lambda *a, **k: hist.copy(deep=True)  # noqa: E731
    with contextlib.ExitStack() as stack:
        for target, fake in (
            (f"{_MD}.buscar_historico", hist_fake),
            ("src.engine.financial_report.buscar_historico", hist_fake),
            (f"{_MD}.buscar_dados_bcb", _fake_bcb(snapshot)),
            (f"{_MD}.buscar_dados_benchmark", _fake_level(snapshot, "yf")),
            (f"{_MD}.buscar_dados_b3", _fake_level(snapshot, "b3")),
            (f"{_MD}.buscar_dolar_bcb", _fake_dolar(snapshot)),
            (f"{_MD}.buscar_dados_tesouro", lambda *a, **k: None),
        ):
            stack.enter_context(mock.patch(target, fake))
        yield


def legacy_yearly(df_combined: pd.DataFrame) -> pd.DataFrame:
    # Cópia literal de src/engine/financial_report.py:406-413 (generate_summary_table),
    # parando antes do .map de formatação — o legado não devolve esses números.
    df = df_combined
    total_ret = (df.iloc[-1] / df.iloc[0]) - 1
    yearly = df.resample('YE').last()
    yearly_ret = yearly.pct_change()
    yearly_ret.iloc[0] = (yearly.iloc[0] / df.iloc[0]) - 1
    summary = yearly_ret.T
    summary.columns = [c.year for c in summary.columns]
    summary['Total Acum.'] = total_ret
    return summary


def _oracle(result):
    """Separa (fig, dados) de um plot_*(return_fig=True) e fecha a figura."""
    fig, data = result if result is not None else (None, None)
    if fig is not None:
        plt.close(fig)
    return data


def _frame(data) -> pd.DataFrame:
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame()


def _adr0012_outputs(report) -> dict[str, pd.DataFrame]:
    return {
        "rolling_vol": _frame(_oracle(report.plot_rolling_volatility(252, return_fig=True))),
        "rolling_sharpe": _frame(_oracle(report.plot_rolling_sharpe(252, return_fig=True))),
        "risk_return": _frame(_oracle(report.plot_risk_return_scatter(return_fig=True))),
        "yearly": normalize_yearly(legacy_yearly(report.df_combined)),
    }


def legacy_outputs(report) -> dict[str, pd.DataFrame]:
    twr = _frame(_oracle(report.plot_twr_evolution(return_fig=True)))
    drawdown = _frame(_oracle(report.plot_drawdown(return_fig=True)))
    irr = _oracle(report.plot_irr_evolution(return_fig=True))
    shadow = _frame(_oracle(report.simulate_shadow_portfolios(return_fig=True)))
    adr = _adr0012_outputs(report)
    irr_series = irr if isinstance(irr, pd.Series) else pd.Series(dtype=float)
    out = {
        "twr": twr,
        "drawdown": drawdown,
        **adr,
        "irr": irr_frame(irr_series),
        "shadow": shadow,
        "kpis": derive_kpis(report.portfolio_df, twr, adr["risk_return"], drawdown, irr_series),
        "portfolio": portfolio_frame(report.portfolio_df),
    }
    return {m: out[m] for m in METRICS}


@contextlib.contextmanager
def pct_change_fill_none():
    orig_df, orig_s = pd.DataFrame.pct_change, pd.Series.pct_change

    def df_wrapper(self, *args, **kwargs):
        kwargs["fill_method"] = None
        return orig_df(self, *args, **kwargs)

    def s_wrapper(self, *args, **kwargs):
        kwargs["fill_method"] = None
        return orig_s(self, *args, **kwargs)

    with mock.patch.object(pd.DataFrame, "pct_change", df_wrapper), \
            mock.patch.object(pd.Series, "pct_change", s_wrapper):
        yield


def legacy_metrics_fill_none(report) -> dict[str, pd.DataFrame]:
    with pct_change_fill_none():
        out = _adr0012_outputs(report)
    return {m: out[m] for m in ADR0012_METRICS}


def run_legacy(token: str, case: Case):
    """Mesma sequência de src/ui/pages/4_configuracoes.py:123-134."""
    from src.engine.financial_report import FinancialReport

    start, end = case.window.start.isoformat(), case.window.end.isoformat()
    report = FinancialReport()
    user_series = report.fetch_user_portfolio(token, classe=case.classe, start_date=start, end_date=end)
    if user_series is None:
        return None
    report.build_dataset(
        user_series=user_series,
        active_benchmarks=list(case.benchmarks),
        start_date=start,
        end_date=end,
    )
    return report, legacy_outputs(report)


def freeze_history(token: str, classe: str | None = None) -> pd.DataFrame:
    from src.data.sources import market_data

    df = market_data.buscar_historico(token, classe=classe)
    if df is None or df.empty:
        raise RuntimeError("DLP sem histórico")
    return df.copy(deep=True)


def _empty_case_diff(note: str) -> SeriesDiff:
    return SeriesDiff(
        metric="caso", column="-", status=FALHA, max_rel=float("nan"), n_out=0,
        first_date=None, only_legacy=0, only_new=0, note=note,
    )


def run_layer1(case: Case, snapshot: MarketSnapshot, hist: pd.DataFrame):
    with legacy_patches(snapshot, hist):
        legacy = run_legacy("parity", case)
    new_out = run_engine(case, snapshot, hist)
    if legacy is None or new_out is None:
        if legacy is None and new_out is None:
            note = "sem carteira nos dois lados"
        else:
            note = "sem carteira só no legado" if legacy is None else "sem carteira só no novo"
        return [_empty_case_diff(note)], {}, new_out or {}

    report, legacy_out = legacy
    diffs: list[SeriesDiff] = []
    for metric in METRICS:
        diffs.extend(compare_frames(metric, legacy_out[metric], new_out[metric], REL_TOL_L1, FALHA))
    diffs = apply_adr0012_recheck(diffs, legacy_metrics_fill_none(report), new_out)
    return diffs, legacy_out, new_out


def run_layer2(case: Case, token: str, client: ApiClient):
    """Legado ao vivo (sem patches) × API em execução; divergências viram INVESTIGAR."""
    notes: list[str] = []
    legacy_out: dict = {}
    new_out: dict = {}

    legacy = run_legacy(token, case)
    if legacy is None:
        notes.append("legado: sem carteira")
    else:
        legacy_out = legacy[1]

    try:
        new_out = api_outputs(client, case)
    except ApiError as exc:
        notes.append(f"API: {exc}")

    if notes:
        return [], notes, legacy_out, new_out

    diffs: list[SeriesDiff] = []
    for metric in METRICS:
        if metric not in new_out:  # portfolio: só na camada 1
            continue
        for d in compare_frames(metric, legacy_out[metric], new_out[metric], REL_TOL_L2, INVESTIGAR):
            diffs.append(d if d.status == OK else dataclasses.replace(d, origin=classify_origin(d)))
    return diffs, notes, legacy_out, new_out
