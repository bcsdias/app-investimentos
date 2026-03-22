"""
scripts/update_b3_cache.py

Baixa o histórico diário dos índices da B3 (IBOV, IFIX, SMLL, IDIV)
e consolida em CSVs prontos para uso pelo engine de análise.

Uso:
    python scripts/update_b3_cache.py              # todos os índices
    python scripts/update_b3_cache.py --index IBOV # índice específico

Executar:
    - Manualmente quando quiser atualizar
    - Via GitHub Actions (cron mensal) — ver .github/workflows/update_b3.yml
    - Requer Google Chrome instalado (o webdriver-manager cuida do chromedriver)
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

# Ajuste do Path para encontrar utils e app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.logger import setup_logger

# ─── Configurações ─────────────────────────────────────────────────────────
DATA_STATIC_DIR = os.path.join(BASE_DIR, "data", "static")
DOWNLOAD_DIR    = os.path.join(BASE_DIR, "data", "downloads_tmp")

ALL_INDICES = ['IBOV', 'IFIX', 'SMLL', 'IDIV']
START_YEAR  = 2000

# Nome exato do arquivo que a B3 gera no download
B3_DOWNLOAD_FILENAME = 'Evolucao_Diaria.csv'

# Mapa de meses PT-BR → número
MAPA_MESES = {
    'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4,
    'Mai': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8,
    'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12,
}

# Inicializa o logger
logger = setup_logger(log_file="update_b3_cache.log")

# ─── Helpers com Retry ─────────────────────────────────────────────────────

def safe_rename(src, dst, retries=5, delay=1):
    for i in range(retries):
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
            return True
        except PermissionError as e:
            if i < retries - 1:
                logger.warning(f"Erro de permissão ao renomear {os.path.basename(src)}. Tentando novamente em {delay}s... (Tentativa {i+1}/{retries})")
                time.sleep(delay)
            else:
                logger.error(f"Falha definitiva ao renomear {src} para {dst}: {e}")
                return False
    return False

def safe_rmtree(path, retries=5, delay=2):
    for i in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except PermissionError as e:
            if i < retries - 1:
                logger.warning(f"Erro de permissão ao remover pasta {os.path.basename(path)}. Tentando novamente em {delay}s... (Tentativa {i+1}/{retries})")
                time.sleep(delay)
            else:
                logger.error(f"Falha definitiva ao remover pasta {path}: {e}")
                return False
    return False

# ─── Setup ─────────────────────────────────────────────────────────────────

def setup_directories():
    os.makedirs(DATA_STATIC_DIR, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for f in os.listdir(DOWNLOAD_DIR):
        try: os.remove(os.path.join(DOWNLOAD_DIR, f))
        except Exception: pass

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

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

# ─── Download ──────────────────────────────────────────────────────────────

def _wait_for_download(dest_path: str, timeout: int = 30) -> bool:
    start = time.time()
    tmp_path = dest_path + ".crdownload"
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            return False
        if os.path.exists(dest_path) and not os.path.exists(tmp_path):
            if os.path.getsize(dest_path) > 0:
                return True
        time.sleep(0.5)

def download_index_year(driver: webdriver.Chrome, index: str, year: int) -> bool:
    url = f"https://sistemaswebb3-listados.b3.com.br/indexStatisticsPage/daily-evolution/{index}?language=pt-br"
    driver.get(url)

    try:
        select_el = WebDriverWait(driver, 15).until(lambda d: d.find_element(By.ID, "selectYear"))
        year_select = Select(select_el)
        available = [opt.get_attribute("value") for opt in year_select.options]
        
        if str(year) not in available:
            logger.info(f"  [{index}] Ano {year} indisponível na B3 (índice não existia ou histórico limitado).")
            return False

        year_select.select_by_value(str(year))
        time.sleep(1) # Extra time to guarantee AJAX updates the button invisibly
        btn = WebDriverWait(driver, 10).until(lambda d: d.find_element(By.XPATH, '//a[text()="Download (ano selecionado)"]'))
        btn.click()

    except TimeoutException:
        # Se ocorrer na formação/leitura do DOM, tratamos como "Timeout Fake" originado por lentidão ou índice vancante
        logger.info(f"  [{index}] Ano {year} (Timeout Fake/Página).")
        return False
    except Exception as e:
        logger.error(f"  [{index}] Erro inesperado no ano {year}: {type(e).__name__}")
        return False

    raw_path  = os.path.join(DOWNLOAD_DIR, B3_DOWNLOAD_FILENAME)
    dest_path = os.path.join(DOWNLOAD_DIR, f"{index}_{year}.csv")

    retries = 3
    for attempt in range(1, retries + 1):
        # Aumentamos gradativamente o tempo pra timeouts reais de arquivo na máquina
        if _wait_for_download(raw_path, timeout=30 + (5 * attempt)):
            if not safe_rename(raw_path, dest_path):
                return False
                
            if attempt > 1:
                logger.info(f"  [{index}] {year} OK (Sucesso após {attempt} tentativas)")
            else:
                logger.info(f"  [{index}] {year} OK")
            return True

        # Se não baixou, trata-se de um TIMEOUT REAL, vamos ao Retry!
        if attempt < retries:
            logger.warning(f"  [{index}] Timeout REAL no download do ano {year}. Tentando {attempt+1}/{retries}...")
            if os.path.exists(raw_path):
                try: os.remove(raw_path) 
                except: pass
            
            try:
                # Dispara novo clique sem recarregar e varrer a página toda de novo
                btn = driver.find_element(By.XPATH, '//a[text()="Download (ano selecionado)"]')
                btn.click()
            except Exception as click_err:
                logger.error(f"  [{index}] Erro no Retry ao reencontrar botão {year}: {click_err}")
                return False
        else:
            logger.error(f"  [{index}] Timeout REAL definitivo no arquivo de {year}.")
            return False
            
    return False

# ─── Processamento ─────────────────────────────────────────────────────────

def process_and_concat(index: str, current_year: int) -> bool:
    dados_completos = []
    for ano in range(START_YEAR, current_year + 1):
        file_path = os.path.join(DOWNLOAD_DIR, f"{index}_{ano}.csv")
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            continue
        try:
            df_ano = pd.read_csv(file_path, sep=";", decimal=",", skiprows=1, encoding="latin-1")
            df_ano = df_ano[pd.to_numeric(df_ano["Dia"], errors="coerce").notna()].copy()
            df_ano["Dia"] = df_ano["Dia"].astype(int)
            df_long = df_ano.melt(id_vars=["Dia"], var_name="Mes", value_name="Close")
            df_long.dropna(subset=["Close"], inplace=True)
            if df_long["Close"].dtype == object:
                df_long["Close"] = (df_long["Close"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float))
            df_long = df_long[df_long["Mes"].isin(MAPA_MESES)].copy()
            df_long["Mes_Num"] = df_long["Mes"].map(MAPA_MESES)
            df_long["Ano"] = ano
            df_long["Date"] = pd.to_datetime(df_long[["Ano", "Mes_Num", "Dia"]].rename(columns={"Ano": "year", "Mes_Num": "month", "Dia": "day"}), errors="coerce")
            df_long.dropna(subset=["Date"], inplace=True)
            dados_completos.append(df_long[["Date", "Close"]])
        except Exception as e:
            logger.error(f"  [{index}] Erro ao processar arquivo de {ano}: {e}")

    if not dados_completos:
        logger.warning(f"  [{index}] Nenhum dado válido encontrado para consolidar.")
        return False

    df_final = pd.concat(dados_completos).sort_values("Date").drop_duplicates(subset=["Date"]).set_index("Date")
    out_path = os.path.join(DATA_STATIC_DIR, f"{index}_all.csv")
    df_final.to_csv(out_path)
    logger.info(f"  [{index}] Salvo com sucesso -> {out_path} ({len(df_final)} registros)")
    return True

# ─── Cleanup ───────────────────────────────────────────────────────────────

def cleanup_downloads():
    if os.path.exists(DOWNLOAD_DIR):
        for f in os.listdir(DOWNLOAD_DIR):
            try: os.remove(os.path.join(DOWNLOAD_DIR, f))
            except Exception: pass
        logger.info("Arquivos temporários removidos.")

# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Atualiza cache estático dos índices da B3.")
    parser.add_argument("--index", choices=ALL_INDICES, help="Atualizar apenas um índice específico")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help=f"Ano inicial (padrão: {START_YEAR})")
    args = parser.parse_args()

    indices_to_run = [args.index] if args.index else ALL_INDICES
    current_year   = datetime.now().year

    logger.info("=" * 60)
    logger.info(f"  Iniciando Atualização de Cache B3: {', '.join(indices_to_run)}")
    logger.info(f"  Período planejado: {args.start_year} -> {current_year}")
    logger.info("=" * 60)

    setup_directories()
    driver = build_driver()

    try:
        for index in indices_to_run:
            logger.info(f"\n[ Processando {index} ]")
            ok_count = 0
            for year in range(args.start_year, current_year + 1):
                if download_index_year(driver, index, year):
                    ok_count += 1
            logger.info(f"  Resumo {index}: {ok_count} anos processados.")
            process_and_concat(index, current_year)
    except Exception as e:
        logger.exception(f"Erro fatal no processo principal: {e}")
    finally:
        driver.quit()
        cleanup_downloads()

    logger.info("\nProcesso concluído.")

if __name__ == "__main__":
    main()