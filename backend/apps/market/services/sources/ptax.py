"""Fonte PTAX (USD/BRL venda) via API Olinda do BCB.

Porta de src/data/sources/market_data.py::buscar_dolar_bcb. Valor CRU:
cotacaoVenda diária (último registro do dia).
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import requests

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_URL = (
    "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
    "CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
    "?@dataInicial='{ini}'&@dataFinalCotacao='{fim}'"
    "&$top=100000&$format=json&$select=cotacaoVenda,dataHoraCotacao"
)


def fetch_ptax(start: dt.date, end: dt.date) -> list[RawPoint]:
    url = _URL.format(
        ini=start.strftime("%m-%d-%Y"), fim=end.strftime("%m-%d-%Y")
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"PTAX Olinda: {exc}") from exc

    values = data.get("value") or []
    if not values:
        return []
    df = pd.DataFrame(values)
    df["dt"] = pd.to_datetime(df["dataHoraCotacao"])
    df["day"] = df["dt"].dt.normalize()
    df = df.sort_values("dt").drop_duplicates(subset=["day"], keep="last")
    return [
        RawPoint(day.date(), float(v))
        for day, v in zip(df["day"], df["cotacaoVenda"])
    ]
