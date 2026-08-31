"""Fonte B3 — índices de evolução diária.

read_b3_static: lê os CSVs estáticos data/static/<indice>_all.csv (porta de
src/data/sources/b3_source.py). É o caminho usado no host (sem Chrome).

fetch_b3_selenium: porta do scraper de src/data/sources/market_data.py. NÃO roda
no host; só é exercitado com mock. A Fase 5 o liga no container com Chrome.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
from selenium import webdriver  # noqa: F401 — usado por fetch_b3_selenium / mockado nos testes

from .base import FetchError, RawPoint

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[5]  # backend/apps/market/services/sources/b3.py
_DEFAULT_STATIC_DIR = _REPO_ROOT / "data" / "static"


def read_b3_static(
    indice: str, start: dt.date, end: dt.date, static_dir: Path | None = None
) -> list[RawPoint]:
    base = static_dir or _DEFAULT_STATIC_DIR
    path = Path(base) / f"{indice}_all.csv"
    if not path.exists():
        logger.info("B3 estático ausente para %s: %s", indice, path)
        return []
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"B3 estático {indice}: {exc}") from exc
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    series = pd.to_numeric(df.loc[mask, "Close"], errors="coerce").dropna()
    return [RawPoint(ts.date(), float(v)) for ts, v in series.items()]


def fetch_b3_selenium(
    indice: str, anos: list[int], download_dir: Path | None = None
) -> list[RawPoint]:
    """Scraper Selenium (headless Chrome). Porta de run_b3_downloader +
    parsing melt/unpivot de buscar_dados_b3. Só roda em ambiente com Chrome."""
    try:
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select
        from selenium.webdriver.support.wait import WebDriverWait
        from webdriver_manager.chrome import ChromeDriverManager

        opts = webdriver.ChromeOptions()
        opts.add_argument("--headless=new")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        try:
            frames: list[pd.DataFrame] = []
            meses = {
                "Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
                "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12,
            }
            for ano in anos:
                url = (
                    "https://sistemaswebb3-listados.b3.com.br/indexStatisticsPage/"
                    f"daily-evolution/{indice}?language=pt-br"
                )
                driver.get(url)
                sel = Select(WebDriverWait(driver, 30).until(
                    lambda d: d.find_element(By.ID, "selectYear")
                ))
                sel.select_by_value(str(ano))
                table_html = driver.find_element(By.TAG_NAME, "table").get_attribute("outerHTML")
                raw = pd.read_html(table_html)[0]
                raw = raw[pd.to_numeric(raw["Dia"], errors="coerce").notna()]
                melted = raw.melt(id_vars=["Dia"], var_name="Mes", value_name="Close").dropna()
                melted["Close"] = (
                    melted["Close"].astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )
                melted["month"] = melted["Mes"].map(meses)
                melted["year"] = ano
                melted = melted.dropna(subset=["month"])
                melted["Date"] = pd.to_datetime(
                    melted[["year", "month", "Dia"]].rename(columns={"Dia": "day"})
                )
                frames.append(melted[["Date", "Close"]])
        finally:
            driver.quit()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"B3 Selenium {indice}: {exc}") from exc

    if not frames:
        return []
    full = pd.concat(frames).sort_values("Date").drop_duplicates("Date", keep="last")
    return [RawPoint(ts.date(), float(v)) for ts, v in zip(full["Date"], full["Close"])]
