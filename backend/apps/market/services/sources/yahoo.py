"""Fonte Yahoo Finance. Valor CRU: Close ajustado (auto_adjust=True).

Porta de src/data/sources/market_data.py::buscar_dados_benchmark — sem cache,
sem CSV local. Só a janela pedida.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import yfinance as yf

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)


def fetch_yahoo(ticker: str, start: dt.date, end: dt.date) -> list[RawPoint]:
    try:
        # end exclusivo no yfinance — soma 1 dia para incluir 'end'
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"Yahoo {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        # yf.download não levanta em falha de rede/rate-limit — devolve vazio.
        # Um ticker válido numa janela normal sempre traz dados; vazio aqui é
        # falha a ser exposta (o ingest marca failed_keys e sai 1), não silêncio.
        raise FetchError(f"Yahoo {ticker}: download vazio para {start}..{end}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # colunas MultiIndex -> uma coluna por ticker
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return [RawPoint(ts.date(), float(v)) for ts, v in close.items()]
