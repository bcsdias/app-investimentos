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
import concurrent.futures
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
        return [{'ticker': 'IBOV', 'fonte': 'evolution', 'remote_id': 'IBOV'}]
    
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
        return [{'ticker': 'IBOV', 'fonte': 'evolution', 'remote_id': 'IBOV'}]

def get_start_year_for_ticker(ticker: str, default_start: int) -> int:
    """Verifica o último ano presente no arquivo local para não baixar histórico desnecessário."""
    path = os.path.join(DATA_STATIC_DIR, f"{ticker}_all.csv")
    if os.path.exists(path):
        try:
            # Lê apenas a coluna de datas para ser leve e rápido
            df = pd.read_csv(path, usecols=['Date'])
            if not df.empty:
                last_year = pd.to_datetime(df['Date']).max().year
                return last_year
        except Exception:
            pass
    return default_start

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
        try:
            p = os.path.join(DOWNLOAD_DIR, f)
            if os.path.isdir(p): shutil.rmtree(p)
            else: os.remove(p)
        except: pass

def build_driver(dl_dir: str) -> webdriver.Chrome:
    options = Options()
    options.add_experimental_option("prefs", {
        "download.default_directory": dl_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    })
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def _wait_for_file(dl_dir: str, pattern: str, timeout: int = 45) -> str:
    """Espera por um arquivo que combine com o padrão no diretório específico do worker."""
    start = time.time()
    while time.time() - start < timeout:
        for f in os.listdir(dl_dir):
            if pattern in f and not f.endswith('.crdownload'):
                path = os.path.join(dl_dir, f)
                if os.path.getsize(path) > 0:
                    return path
        time.sleep(1)
    return None

# ─── Download: Fonte EVOLUTION (Página) ────────────────────────────────────

def download_evolution_year(driver: webdriver.Chrome, dl_dir: str, index: str, year: int) -> str:
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

    raw_path = os.path.join(dl_dir, B3_DOWNLOAD_FILENAME)
    dest_path = os.path.join(dl_dir, f"{index}_{year}.csv")
    for attempt in range(1, 4):
        path = _wait_for_file(dl_dir, B3_DOWNLOAD_FILENAME)
        if path:
            if safe_rename(path, dest_path): return 'OK'
            return 'ERROR'
        if attempt < 3:
            logger.warning(f"  [{index}] Retry Real Timeout {year}...")
            try: driver.find_element(By.XPATH, '//a[text()="Download (ano selecionado)"]').click()
            except: pass
    return 'REAL_TIMEOUT'

# ─── Download: Fonte ON DEMAND (Proxy) ─────────────────────────────────────

def download_on_demand(driver: webdriver.Chrome, dl_dir: str, ticker: str, remote_id: str) -> bool:
    url = f"https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/DownloadIndexOnDemand/{remote_id}.xlsx"
    logger.info(f"  [{ticker}] Buscando planilha On-Demand...")
    driver.get(url)
    
    path = _wait_for_file(dl_dir, ".xlsx")
    if not path:
        logger.error(f"  [{ticker}] Falha ao baixar planilha on-demand (Timeout).")
        return False
    
    dest_path = os.path.join(dl_dir, f"{ticker}_on_demand.xlsx")
    return safe_rename(path, dest_path)

# ─── Processamento ─────────────────────────────────────────────────────────

def process_evolution(dl_dir: str, index: str, current_year: int, start_year: int):
    dados = []
    for ano in range(start_year, current_year + 1):
        path = os.path.join(dl_dir, f"{index}_{ano}.csv")
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

def process_on_demand(dl_dir: str, ticker: str):
    path = os.path.join(dl_dir, f"{ticker}_on_demand.xlsx")
    if not os.path.exists(path): return
    try:
        df = pd.read_excel(path, sheet_name="Indice Resumido")
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

def cleanup_downloads(dl_dir: str = DOWNLOAD_DIR):
    if os.path.exists(dl_dir):
        for f in os.listdir(dl_dir):
            try:
                p = os.path.join(dl_dir, f)
                if os.path.isdir(p): shutil.rmtree(p)
                else: os.remove(p)
            except: pass
        if dl_dir == DOWNLOAD_DIR:
            logger.info("Arquivos temporários globais removidos.")

# ─── Worker de Processamento ───────────────────────────────────────────────

def process_indices_worker(worker_id: str, indices: list, start_year: int):
    stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'failures': []}
    if not indices: return stats

    current_year = datetime.now().year
    dl_dir = os.path.join(DOWNLOAD_DIR, f"worker_{worker_id}")
    os.makedirs(dl_dir, exist_ok=True)
    
    # Limpa antes
    for f in os.listdir(dl_dir):
        try: os.remove(os.path.join(dl_dir, f))
        except: pass

    driver = build_driver(dl_dir)
    try:
        for info in indices:
            ticker = info['ticker']
            stats['total'] += 1
            logger.info(f"\n[ W-{worker_id} | {ticker} ]")
            
            if info['fonte'] == 'on_demand':
                if download_on_demand(driver, dl_dir, ticker, info['remote_id']):
                    process_on_demand(dl_dir, ticker)
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['failures'].append(f"{ticker}: Planilha on-demand não encontrada ou erro.")
            else:
                # Inteligência de atualização: só volta no tempo até o último ano que já temos localmente
                target_start = get_start_year_for_ticker(ticker, start_year)
                
                ok_count = 0
                index_ok = False
                was_skipped = False
                for year in range(current_year, target_start - 1, -1):
                    res = download_evolution_year(driver, dl_dir, ticker, year)
                    if res == 'OK':
                        ok_count += 1
                        index_ok = True
                        logger.info(f"  [W-{worker_id} | {ticker}] {year} OK")
                    elif res == 'FAKE_TIMEOUT':
                        if year == current_year:
                            logger.warning(f"  [W-{worker_id} | {ticker}] Índice não responde para {current_year}. Pulando.")
                            stats['skipped'] += 1
                            stats['failures'].append(f"{ticker}: Indisponível/Página não responde (Ano {current_year})")
                            was_skipped = True
                            break
                        else: break
                    elif res == 'REAL_TIMEOUT':
                        stats['failures'].append(f"{ticker} ({year}): Timeout Real no download.")
                
                if not was_skipped:
                    if index_ok:
                        stats['success'] += 1
                        process_evolution(dl_dir, ticker, current_year, target_start)
                    else:
                        stats['failed'] += 1
                        if not any(f.startswith(f"{ticker}:") for f in stats['failures']):
                            stats['failures'].append(f"{ticker}: Nenhum dado processado.")
    finally:
        driver.quit()
        # Limpeza do worker
        for f in os.listdir(dl_dir):
            try: os.remove(os.path.join(dl_dir, f))
            except: pass

    return stats

# ─── Main ──────────────────────────────────────────────────────────────────

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

    logger.info("=" * 60)
    logger.info(f"  Iniciando Atualização Cache B3 ({len(active_indices)} índices)")
    logger.info("=" * 60)

    final_stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'failures': []}

    # Separa os tipos de fonte
    on_demand_indices = [i for i in active_indices if i['fonte'] == 'on_demand']
    evolution_indices = [i for i in active_indices if i['fonte'] == 'evolution']

    # 1. Processa On-Demand em 1 única thread rápida
    if on_demand_indices:
        logger.info(f"\n>> Processando {len(on_demand_indices)} índices On-Demand (Sequencial)...")
        res = process_indices_worker("ON_DEMAND", on_demand_indices, args.start_year)
        for k in final_stats:
            if isinstance(final_stats[k], list): final_stats[k].extend(res[k])
            else: final_stats[k] += res[k]

    # 2. Processa Evolution em Paralelo (Máx 3 Workers para evitar bloqueio)
    if evolution_indices:
        n_workers = 3
        if len(evolution_indices) < n_workers:
            n_workers = len(evolution_indices)
        
        # Divide a lista em chunks quase iguais
        chunk_size = max(1, len(evolution_indices) // n_workers)
        chunks = [evolution_indices[i:i + chunk_size] for i in range(0, len(evolution_indices), chunk_size)]
        
        # Se houve sobra de truncagem, pode criar um 4º chunk pequeno, vamos ajustar para exatos n_workers
        if len(chunks) > n_workers:
            chunks[n_workers - 1].extend([item for sublist in chunks[n_workers:] for item in sublist])
            chunks = chunks[:n_workers]

        logger.info(f"\n>> Processando {len(evolution_indices)} índices Evolution em {len(chunks)} threads simultâneas...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(process_indices_worker, str(i+1), chunk, args.start_year) for i, chunk in enumerate(chunks)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    for k in final_stats:
                        if isinstance(final_stats[k], list): final_stats[k].extend(res[k])
                        else: final_stats[k] += res[k]
                except Exception as e:
                    logger.error(f"Erro no Worker: {e}")

    cleanup_downloads()
    
    logger.info("\n" + "=" * 60)
    logger.info("  RESUMO DA EXECUÇÃO")
    logger.info("=" * 60)
    logger.info(f"  Total de índices tentados: {final_stats['total']}")
    logger.info(f"  Sucessos: {final_stats['success']}")
    logger.info(f"  Falhas/Timeouts: {final_stats['failed']}")
    logger.info(f"  Pulas (Sem dados no ano atual): {final_stats['skipped']}")
    
    if final_stats['failures']:
        logger.info("\n  DETALHAMENTO DE FALHAS:")
        for fail in final_stats['failures']:
            logger.info(f"  - {fail}")
    
    logger.info("\n" + "=" * 60)
    logger.info("  Processo concluído.")

if __name__ == "__main__":
    main()