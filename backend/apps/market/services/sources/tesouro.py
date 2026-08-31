"""Fonte Tesouro Direto (Tesouro Transparente). Valor CRU: PU Base Manhã.

Porta de src/data/sources/market_data.py::buscar_dados_tesouro — sem cache. Baixa
o CSV nacional para _DOWNLOAD_DIR e filtra por título + vencimento.
"""
from __future__ import annotations

import datetime as dt
import logging
import tempfile
from pathlib import Path

import pandas as pd
import requests

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_DOWNLOAD_DIR = Path(tempfile.gettempdir())
_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/"
    "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
    "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
)


def fetch_tesouro(
    titulo: str, vencimento: dt.date, start: dt.date, end: dt.date
) -> list[RawPoint]:
    path = _DOWNLOAD_DIR / "PrecoTaxaTesouroDireto.csv"
    try:
        resp = requests.get(_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            encoding="latin-1",
            usecols=["Tipo Titulo", "Data Vencimento", "Data Base", "PU Base Manha"],
        )
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"Tesouro {titulo} {vencimento}: {exc}") from exc

    df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"], dayfirst=True)
    df["Data Base"] = pd.to_datetime(df["Data Base"], dayfirst=True)
    venc = pd.Timestamp(vencimento)
    mask = (df["Tipo Titulo"] == titulo) & (df["Data Vencimento"] == venc)
    sel = (
        df[mask]
        .set_index("Data Base")
        .sort_index()
        .loc[str(start) : str(end), "PU Base Manha"]
    )
    sel = sel[~sel.index.duplicated(keep="last")]
    return [RawPoint(ts.date(), float(v)) for ts, v in sel.items()]
