"""Fonte de dados ANBIMA para os índices da família IMA.

Camada 1: Arquivo de histórico completo (XLS no S3).
Camada 2: Consulta diária no endpoint legado ima-sh-down.asp para janelas curtas (<= 30 dias).
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import logging

import pandas as pd
import requests

from apps.market.services.sources.base import FetchError, RawPoint

logger = logging.getLogger(__name__)

HIST_URL = "https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico/{code}-HISTORICO.xls"
DAILY_URL = "https://www.anbima.com.br/informacoes/ima/ima-sh-down.asp"

INDEX_LABELS: dict[str, tuple[str, ...]] = {
    "IMAS": ("IMA-S", "IMA - S"),
    "IMAB": ("IMA-B", "IMA - B"),
    "IMAB5": ("IMA-B 5", "IMA - B 5"),
    "IMAB5MAIS": ("IMA-B 5+", "IMA - B 5+"),
    "IRFM": ("IRF-M", "IRF - M"),
    "IMAGERAL": ("IMA-GERAL", "IMA - Geral", "IMA - GERAL"),
}

FALLBACK_MAX_DAYS = 30
NO_DATA_PREFIX = "Não há dados disponíveis"


def _parse_brazilian_float(val: object) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(".", "").replace(",", ".")
    return float(s)


def fetch_history(code: str, start: dt.date, end: dt.date) -> list[RawPoint]:
    """Busca o histórico completo do índice no S3 da ANBIMA e filtra pela janela."""
    if code not in INDEX_LABELS:
        raise FetchError(f"Código ANBIMA desconhecido: {code}")

    labels = INDEX_LABELS[code]
    url = HIST_URL.format(code=code)

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        raise FetchError(f"Erro ao baixar histórico ANBIMA ({url}): {exc}") from exc

    try:
        df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
    except Exception as exc:
        raise FetchError(f"Erro ao ler planilha ANBIMA ({url}): {exc}") from exc

    required_cols = {"Índice", "Data de Referência", "Número Índice"}
    if not required_cols.issubset(df.columns):
        raise FetchError(f"Colunas obrigatórias {required_cols} ausentes no histórico ANBIMA: {df.columns.tolist()}")

    df_filtered = df[df["Índice"].astype(str).str.strip().isin(labels)]
    if df_filtered.empty:
        raise FetchError(f"Arquivo da ANBIMA sem linhas para o índice {labels}")

    points: list[RawPoint] = []
    for _, row in df_filtered.iterrows():
        raw_date = row["Data de Referência"]
        if pd.isna(raw_date):
            continue
        if isinstance(raw_date, pd.Timestamp):
            row_date = raw_date.date()
        elif isinstance(raw_date, dt.datetime):
            row_date = raw_date.date()
        elif isinstance(raw_date, dt.date):
            row_date = raw_date
        else:
            try:
                row_date = pd.to_datetime(raw_date).date()
            except Exception:
                continue

        if start <= row_date <= end:
            val = _parse_brazilian_float(row["Número Índice"])
            points.append(RawPoint(date=row_date, value=val))

    points.sort(key=lambda p: p.date)
    return points


def fetch_daily(code: str, day: dt.date) -> RawPoint | None:
    """Consulta o valor do índice para uma data específica via formulário ASP da ANBIMA."""
    if code not in INDEX_LABELS:
        raise FetchError(f"Código ANBIMA desconhecido: {code}")

    labels = INDEX_LABELS[code]
    data = {
        "Pai": "ima",
        "escolha": "2",
        "Idioma": "PT",
        "saida": "csv",
        "Dt_Ref": day.strftime("%d/%m/%Y"),
        "Dt_Ref_Ver": day.strftime("%Y%m%d"),
    }

    try:
        resp = requests.post(DAILY_URL, data=data, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        raise FetchError(f"Erro na requisição diária ANBIMA ({DAILY_URL}): {exc}") from exc

    try:
        text = resp.content.decode("cp1252")
    except UnicodeDecodeError:
        text = resp.content.decode("utf-8", errors="replace")

    # Feriado, fim de semana ou dia ainda não publicado: a ANBIMA devolve só esta frase.
    if text.strip().startswith(NO_DATA_PREFIX):
        return None

    reader = csv.reader(io.StringIO(text), delimiter=";")
    rows = list(reader)
    if not rows:
        return None

    header_idx = None
    header = []
    for i, r in enumerate(rows):
        cleaned = [c.strip() for c in r]
        if "Índice" in cleaned and "Número Índice" in cleaned:
            header_idx = i
            header = cleaned
            break

    if header_idx is None:
        raise FetchError(f"Cabeçalho retornado pela ANBIMA sem 'Número Índice': {rows[:3]}")

    idx_col = header.index("Índice")
    val_col = header.index("Número Índice")
    date_col = header.index("Data de Referência") if "Data de Referência" in header else None
    day_str = day.strftime("%d/%m/%Y")

    for row in rows[header_idx + 1 :]:
        if not row or len(row) <= max(idx_col, val_col):
            continue
        if row[idx_col].strip() in labels:
            if date_col is not None and row[date_col].strip() != day_str:
                return None
            val_str = row[val_col].strip()
            if not val_str or val_str == "--":
                return None
            val = _parse_brazilian_float(val_str)
            return RawPoint(date=day, value=val)

    return None


def fetch_anbima(code: str, start: dt.date, end: dt.date) -> list[RawPoint]:
    """Busca dados do índice da ANBIMA, com fallback para o diário em janelas curtas."""
    try:
        return fetch_history(code, start, end)
    except FetchError as exc:
        if (end - start).days <= FALLBACK_MAX_DAYS:
            logger.warning(
                "Falha ao baixar histórico ANBIMA para %s (%s). Executando fallback diário...",
                code,
                exc,
            )
            points: list[RawPoint] = []
            curr = start
            while curr <= end:
                if curr.weekday() < 5:  # segunda a sexta
                    pt = fetch_daily(code, curr)
                    if pt is not None:
                        points.append(pt)
                curr += dt.timedelta(days=1)
            return points
        raise
