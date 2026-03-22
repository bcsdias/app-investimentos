"""
scripts/update_b3_cache.py

Baixa o histórico diário dos índices da B3 (IBOV, IFIX, SMLL, IDIV, etc.)
e consolida em CSVs prontos para uso pelo engine de análise.

Suporta duas fontes da B3:
1. Evolução Diária (Página de Estatísticas - busca por ano)
2. On Demand (Proxy de Download - planilha completa .xlsx)

Uso:
    python scripts/update_b3_cache.py              # todos os índices do CSV
    python scripts/update_b3_cache.py --index IBOV # índice específico
"""

import os
import time
import shutil
import argparse
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import sys

# Ajuste do Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.logger import setup_logger

DATA_STATIC_DIR = os.path.join(BASE_DIR, "data", "static")
DOWNLOAD_DIR    = os.path.join(BASE_DIR, "data", "downloads_tmp")
TICKERS_CSV     = os.path.join(BASE_DIR, "scripts", "B3_Ticker_Produtos.csv")

START_YEAR  = 2000
B3_DOWNLOAD_FILENAME = 'Evolucao_Diaria.csv'

MAPA_MESES = {
    'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4,
    'Mai': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8,
    'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12,
}

logger = setup_logger(log_file="update_b3_cache.log")

# ─── Helpers ───────────────────────────────────────────────────────────────

def load_tickers_from_b3_csv(file_path):
    """Retorna lista de dicionários com metadados do índice."""
    if not os.path.exists(file_path):
        logger.warning(f"CSV de tickers não encontrado: {file_path}. Usando padrões.")
        return [{'ticker': 'IBOV', 'fonte': 'evolution'}]
    
    try:
        df = pd.read_csv(file_path, sep=';', encoding='latin-1')
        results = []
        for _, row in df.iterrows():
            ticker = str(row.get('Ticker B3', '')).strip()
            if not ticker or ticker == 'nan': continue
            
            fonte_raw = str(row.get('Fonte', '')).strip()
            info = {'ticker': ticker, 'fonte': 'evolution', 'remote_id': ticker}
            
            if 'on_demand:' in fonte_raw:
                info['fonte'] = 'on_demand'
                info['remote_id'] = fonte_raw.split('on_demand:')[1].strip()
            
            # Evita duplicar o mesmo ticker (ex: IBOV aparece várias vezes por causa dos ETFs)
            if not any(r['ticker'] == ticker for r in results):
                results.append(info)
        return results
    except Exception as e:
        logger.error(f"Erro ao ler CSV de tickers: {e}")
        return [{'ticker': 'IBOV', 'fonte': 'evolution'}]

def safe_rename(src, dst, retries=5, delay=1):
    for i in range(retries):
        try:
            if os.path.exists(dst): os.remove(dst)
            os.rename(src, dst)
            return True
        except PermissionError:
            time.sleep(delay)
    return False

# ─── Setup ─────────────────────────────────────────────────────────────────

def setup_directories():
    os.makedirs(DATA_STATIC_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for f in os.listdir(DOWNLOAD_DIR):
        try: os.remove(os.path.join(DOWNLOAD_DIR, f))
        except: pass

def build_driver() -> webdriver.Chrome:
    options = Options()
    options.add_experimental_option("prefs", {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    })
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def _wait_for_file(pattern: str, timeout: int = 45) -> str:
    """Espera por um arquivo que combine com o padrão no DOWNLOAD_DIR."""
    start = time.time()
    while time.time() - start < timeout:
        for f in os.listdir(DOWNLOAD_DIR):
            if pattern in f and not f.endswith('.crdownload'):
                path = os.path.join(DOWNLOAD_DIR, f)
                if os.path.getsize(path) > 0:
                    return path
        time.sleep(1)
    return None

# ─── Download: Fonte EVOLUTION (Página) ────────────────────────────────────

def download_evolution_year(driver: webdriver.Chrome, index: str, year: int) -> str:
    url = f"https://sistemaswebb3-listados.b3.com.br/indexStatisticsPage/daily-evolution/{index}?language=pt-br"
    driver.get(url)
    try:
        select_el = WebDriverWait(driver, 15).until(lambda d: d.find_element(By.ID, "selectYear"))
        year_select = Select(select_el)
        if str(year) not in [opt.get_attribute("value") for opt in year_select.options]:
            return 'FAKE_TIMEOUT'
        year_select.select_by_value(str(year))
        time.sleep(1)
        btn = WebDriverWait(driver, 10).until(lambda d: d.find_element(By.XPATH, '//a[text()="Download (ano selecionado)"]'))
        btn.click()
    except TimeoutException: return 'FAKE_TIMEOUT'
    except Exception as e:
        logger.error(f"  [{index}] Erro: {type(e).__name__}")
        return 'ERROR'

    raw_path = os.path.join(DOWNLOAD_DIR, B3_DOWNLOAD_FILENAME)
    dest_path = os.path.join(DOWNLOAD_DIR, f"{index}_{year}.csv")
    for attempt in range(1, 4):
        path = _wait_for_file(B3_DOWNLOAD_FILENAME)
        if path:
            if safe_rename(path, dest_path): return 'OK'
            return 'ERROR'
        if attempt < 3:
            logger.warning(f"  [{index}] Retry Real Timeout {year}...")
            try: driver.find_element(By.XPATH, '//a[text()="Download (ano selecionado)"]').click()
            except: pass
    return 'REAL_TIMEOUT'

# ─── Download: Fonte ON DEMAND (Proxy) ─────────────────────────────────────

def download_on_demand(driver: webdriver.Chrome, ticker: str, remote_id: str) -> bool:
    url = f"https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/DownloadIndexOnDemand/{remote_id}.xlsx"
    logger.info(f"  [{ticker}] Buscando planilha On-Demand...")
    driver.get(url)
    
    # Espera por qualquer arquivo .xlsx
    path = _wait_for_file(".xlsx")
    if not path:
        logger.error(f"  [{ticker}] Falha ao baixar planilha on-demand (Timeout).")
        return False
    
    dest_path = os.path.join(DOWNLOAD_DIR, f"{ticker}_on_demand.xlsx")
    return safe_rename(path, dest_path)

# ─── Processamento ─────────────────────────────────────────────────────────

def process_evolution(index: str, current_year: int, start_year: int):
    dados = []
    for ano in range(start_year, current_year + 1):
        path = os.path.join(DOWNLOAD_DIR, f"{index}_{ano}.csv")
        if not os.path.exists(path): continue
        try:
            df = pd.read_csv(path, sep=";", decimal=",", skiprows=1, encoding="latin-1")
            df = df[pd.to_numeric(df["Dia"], errors="coerce").notna()].copy()
            if df.empty: continue
            df["Dia"] = df["Dia"].astype(int)
            df_long = df.melt(id_vars=["Dia"], var_name="Mes", value_name="Close")
            df_long.dropna(subset=["Close"], inplace=True)
            if df_long["Close"].dtype == object:
                df_long["Close"] = df_long["Close"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)
            df_long = df_long[df_long["Mes"].isin(MAPA_MESES)].copy()
            df_long["Mes_Num"] = df_long["Mes"].map(MAPA_MESES)
            df_long["Ano"] = ano
            df_long["Date"] = pd.to_datetime(df_long[["Ano", "Mes_Num", "Dia"]].rename(columns={"Ano":"year","Mes_Num":"month","Dia":"day"}), errors="coerce")
            dados.append(df_long.dropna(subset=["Date"])[["Date", "Close"]])
        except Exception as e: logger.error(f"  [{index}] Erro em {ano}: {e}")
    
    if dados:
        save_incremental(index, pd.concat(dados))

def process_on_demand(ticker: str):
    path = os.path.join(DOWNLOAD_DIR, f"{ticker}_on_demand.xlsx")
    if not os.path.exists(path): return
    try:
        # Requer openpyxl
        df = pd.read_excel(path, sheet_name="Indice Resumido")
        # Espera colunas: Data_Referencia e Indice
        df = df.rename(columns={"Data_Referencia": "Date", "Indice": "Close"})
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
        save_incremental(ticker, df)
    except Exception as e:
        logger.error(f"  [{ticker}] Erro ao processar Excel: {e}")

def save_incremental(ticker: str, df_new: pd.DataFrame):
    df_new = df_new.drop_duplicates(subset=["Date"]).set_index("Date")
    out_path = os.path.join(DATA_STATIC_DIR, f"{ticker}_all.csv")
    if os.path.exists(out_path):
        try:
            df_old = pd.read_csv(out_path, parse_dates=["Date"]).set_index("Date")
            df_final = pd.concat([df_old, df_new]).sort_index()
            df_final = df_final[~df_final.index.duplicated(keep="last")]
        except: df_final = df_new
    else: df_final = df_new
    df_final.to_csv(out_path)
    logger.info(f"  [{ticker}] Salvo -> {out_path} ({len(df_final)} registros)")

def cleanup_downloads():
    if os.path.exists(DOWNLOAD_DIR):
        for f in os.listdir(DOWNLOAD_DIR):
            try: os.remove(os.path.join(DOWNLOAD_DIR, f))
            except: pass
        logger.info("Arquivos temporários removidos.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    args = parser.parse_args()

    setup_directories()
    all_info = load_tickers_from_b3_csv(TICKERS_CSV)
    
    if args.index:
        active_indices = [i for i in all_info if i['ticker'] == args.index]
        if not active_indices: active_indices = [{'ticker': args.index, 'fonte': 'evolution', 'remote_id': args.index}]
    else:
        active_indices = all_info
    
    # Prioritiza on_demand (planilhas rápidas) antes de evolution (vários anos)
    active_indices.sort(key=lambda x: 0 if x.get('fonte') == 'on_demand' else 1)

    logger.info("=" * 60)
    logger.info(f"  Iniciando Atualização Cache B3 ({len(active_indices)} índices)")
    logger.info("=" * 60)

    driver = build_driver()
    stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'failures': []}
    
    try:
        current_year = datetime.now().year
        for info in active_indices:
            ticker = info['ticker']
            stats['total'] += 1
            logger.info(f"\n[ {ticker} ]")
            
            if info['fonte'] == 'on_demand':
                if download_on_demand(driver, ticker, info['remote_id']):
                    process_on_demand(ticker)
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['failures'].append(f"{ticker}: Planilha on-demand não encontrada ou erro no download.")
            else:
                ok_count = 0
                index_ok = False
                was_skipped = False
                for year in range(current_year, args.start_year - 1, -1):
                    res = download_evolution_year(driver, ticker, year)
                    if res == 'OK':
                        ok_count += 1
                        index_ok = True
                        logger.info(f"  {year} OK")
                    elif res == 'FAKE_TIMEOUT':
                        if year == current_year:
                            logger.warning(f"  Índice não responde para {current_year}. Pulando.")
                            stats['skipped'] += 1
                            stats['failures'].append(f"{ticker}: Indisponível/Página não responde (Ano {current_year})")
                            was_skipped = True
                            break
                        else: break
                    elif res == 'REAL_TIMEOUT':
                        stats['failures'].append(f"{ticker} ({year}): Timeout Real no download do arquivo.")
                        # Não interromper o índice, tentar anos anteriores
                
                if not was_skipped:
                    if index_ok:
                        stats['success'] += 1
                        process_evolution(ticker, current_year, args.start_year)
                    else:
                        stats['failed'] += 1
                        if not any(f.startswith(f"{ticker}:") for f in stats['failures']):
                            stats['failures'].append(f"{ticker}: Nenhum dado processado.")
    finally:
        driver.quit()
        cleanup_downloads()
    
    logger.info("\n" + "=" * 60)
    logger.info("  RESUMO DA EXECUÇÃO")
    logger.info("=" * 60)
    logger.info(f"  Total de índices tentados: {stats['total']}")
    logger.info(f"  Sucessos: {stats['success']}")
    logger.info(f"  Falhas/Timeouts: {stats['failed']}")
    logger.info(f"  Pulas (Sem dados no ano atual): {stats['skipped']}")
    
    if stats['failures']:
        logger.info("\n  DETALHAMENTO DE FALHAS:")
        for fail in stats['failures']:
            logger.info(f"  - {fail}")
    
    logger.info("\n" + "=" * 60)
    logger.info("  Processo concluído.")

if __name__ == "__main__":
    main()