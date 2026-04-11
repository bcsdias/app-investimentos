import os
import sys
import time
import argparse
import pandas as pd
import logging
from datetime import datetime

# Ajuste do Path para encontrar a src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data.cache import cache_set
from src.utils.logger import logger
from src.data.benchmarks_config import CATALOGO_BCB, CATALOGO_YF
from src.data.sources.market_data import buscar_dados_bcb, buscar_dados_benchmark

# Configuração de caminhos
DATA_STATIC_DIR = os.path.join(BASE_DIR, "data", "static")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
ATIVOS_CSV = os.path.join(BASE_DIR, "data", "ativos.csv")

# Estatísticas globais
stats = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "failed_items": []
}

def retry_cache_set(key, value, ttl=86400, max_retries=3):
    """Executa cache_set com lógica de retry e backoff exponencial."""
    stats["total"] += 1
    for attempt in range(1, max_retries + 1):
        # cache_set agora retorna um booleano de sucesso
        if cache_set(key, value, ttl):
            stats["success"] += 1
            return True
        else:
            wait_time = 2 ** attempt
            logger.warning(f"  [Attempt {attempt}] Falha na serialização/envio de {key}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    stats["failed"] += 1
    stats["failed_items"].append(key)
    logger.error(f"  [FAILED] Não foi possível subir {key} após {max_retries} tentativas.")
    return False

def should_sync(key, target_items):
    """Verifica se o item deve ser sincronizado com base nos filtros."""
    if not target_items:
        return True
    return any(target.lower() in key.lower() for target in target_items)

def sync_b3_indices(target_items=None):
    logger.info("\n--- Sintronizando Índices B3 (data/static) ---")
    if not os.path.exists(DATA_STATIC_DIR):
        logger.warning(f"Diretório {DATA_STATIC_DIR} não encontrado.")
        return

    files = [f for f in os.listdir(DATA_STATIC_DIR) if f.endswith("_all.csv")]
    for f in files:
        ticker = f.replace("_all.csv", "")
        key = f"b3:{ticker}"
        
        if not should_sync(key, target_items):
            continue

        path = os.path.join(DATA_STATIC_DIR, f)
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty:
                logger.info(f"Subindo {key} ({len(df)} registros)...")
                retry_cache_set(key, df.iloc[:, 0])
        except Exception as e:
            logger.error(f"Erro ao processar {f}: {e}")
            stats["failed"] += 1
            stats["failed_items"].append(key)

def sync_raw_data(target_items=None):
    logger.info("\n--- Sincronizando Dados Raw (YF, BCB, TD) ---")
    if not os.path.exists(DATA_RAW_DIR):
        logger.warning(f"Diretório {DATA_RAW_DIR} não encontrado.")
        return

    for f in os.listdir(DATA_RAW_DIR):
        if not f.endswith(".csv"): continue
        
        key = None
        path = os.path.join(DATA_RAW_DIR, f)
        
        # YFinance (YF_TICKER.csv)
        if f.startswith("YF_"):
            ticker = f.replace("YF_", "").replace(".csv", "").replace("_", ".")
            if ticker == "BVSP": ticker = "^BVSP"
            key = f"yf:{ticker}"
            
        # BCB (BCB_CODIGO.csv)
        elif f.startswith("BCB_"):
            codigo = f.replace("BCB_", "").replace(".csv", "")
            key = f"bcb:{codigo}"
            
        # Tesouro Direto (TD_TITULO_VENC.csv)
        elif f.startswith("TD_"):
            parts = f.replace("TD_", "").replace(".csv", "").split("_")
            if len(parts) >= 2:
                titulo = parts[0].replace("mais", "+")
                vencimento = parts[1].replace("-", "/")
                key = f"td:{titulo}:{vencimento}"

        if key:
            if not should_sync(key, target_items):
                continue
                
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty:
                    logger.info(f"Subindo {key} ({len(df)} registros)...")
                    retry_cache_set(key, df.iloc[:, 0])
            except Exception as e:
                logger.error(f"Erro ao processar {f}: {e}")
                stats["failed"] += 1
                stats["failed_items"].append(key)

def sync_assets_metadata(target_items=None):
    key = "meta:ativos"
    if not should_sync(key, target_items):
        return

    logger.info("\n--- Sincronizando Metadados de Ativos ---")
    if not os.path.exists(ATIVOS_CSV):
        logger.warning(f"Arquivo {ATIVOS_CSV} não encontrado.")
        return

    try:
        df = pd.read_csv(ATIVOS_CSV, sep=";", encoding="latin-1")
        logger.info(f"Subindo {key} ({len(df)} registros)...")
        retry_cache_set(key, df, ttl=None)
    except Exception as e:
        logger.error(f"Erro ao processar {ATIVOS_CSV}: {e}")
        stats["failed"] += 1
        stats["failed_items"].append(key)

import subprocess

def get_items_to_process(args, target_items):
    """Faz um scan prévio para listar o que será processado."""
    to_process = []
    
    # Índices B3
    if not (args.raw or args.meta) or args.indices or target_items:
        if os.path.exists(DATA_STATIC_DIR):
            files = [f for f in os.listdir(DATA_STATIC_DIR) if f.endswith("_all.csv")]
            for f in files:
                ticker = f.replace("_all.csv", "")
                key = f"b3:{ticker}"
                if should_sync(key, target_items):
                    to_process.append(key)
    
    # Raw Data
    if not (args.indices or args.meta) or args.raw or target_items:
        if os.path.exists(DATA_RAW_DIR):
            for f in os.listdir(DATA_RAW_DIR):
                if not f.endswith(".csv"): continue
                key = None
                if f.startswith("YF_"):
                    ticker = f.replace("YF_", "").replace(".csv", "").replace("_", ".")
                    if ticker == "BVSP": ticker = "^BVSP"
                    key = f"yf:{ticker}"
                elif f.startswith("BCB_"):
                    key = f"bcb:{f.replace('BCB_', '').replace('.csv', '')}"
                elif f.startswith("TD_"):
                    parts = f.replace("TD_", "").replace(".csv", "").split("_")
                    if len(parts) >= 2:
                        key = f"td:{parts[0].replace('mais', '+')}:{parts[1].replace('-', '/')}"
                
                if key and should_sync(key, target_items):
                    to_process.append(key)
                    
    # Meta
    if not (args.indices or args.raw) or args.meta or target_items:
        key = "meta:ativos"
        if should_sync(key, target_items) and os.path.exists(ATIVOS_CSV):
            to_process.append(key)
            
    return sorted(list(set(to_process)))

def run_download_updates():
    """Executa scripts de download para atualizar arquivos locais antes do sync."""
    logger.info("\n" + "="*60)
    logger.info("FORÇANDO DOWNLOAD DE DADOS (Atualização de Cache Local)")
    logger.info("="*60)
    
    # 1. Update B3 Indices (via subprocess para isolar o Selenium)
    update_script = os.path.join(BASE_DIR, "scripts", "update_b3_cache.py")
    if os.path.exists(update_script):
        logger.info("[B3] Executando scripts/update_b3_cache.py...")
        try:
            subprocess.run([sys.executable, update_script], check=True)
            logger.info("[B3] Download de Índices concluído.")
        except Exception as e:
            logger.error(f"[B3] Erro ao rodar update_b3_cache.py: {e}")
    
    # 2. Update BCB Series
    logger.info("\n[BCB] Atualizando séries econômicas...")
    start_date = "1995-01-01"
    today = datetime.now().strftime("%Y-%m-%d")
    for nome, codigo in CATALOGO_BCB.items():
        try:
            logger.info(f"  -> Baixando {nome} ({codigo})...")
            buscar_dados_bcb(codigo, start_date, today, nome=nome)
        except Exception as e:
            logger.error(f"  [BCB] Erro ao baixar {nome}: {e}")

    # 3. Update YFinance Benchmarks
    logger.info("\n[YF] Atualizando benchmarks globais...")
    for nome, ticker in CATALOGO_YF.items():
        try:
            logger.info(f"  -> Baixando {nome} ({ticker})...")
            buscar_dados_benchmark(ticker, start_date, today, nome=nome)
        except Exception as e:
            logger.error(f"  [YF] Erro ao baixar {nome}: {e}")

    logger.info("\nDownload concluído.\n")

def main():
    parser = argparse.ArgumentParser(description="Sincroniza dados locais com Redis (Upstash)")
    parser.add_argument("--indices", action="store_true", help="Sincroniza apenas índices B3")
    parser.add_argument("--raw", action="store_true", help="Sincroniza apenas dados raw (YF, BCB, TD)")
    parser.add_argument("--meta", action="store_true", help="Sincroniza apenas metadados")
    parser.add_argument("--items", type=str, help="Lista de itens (keys ou tickers) separados por vírgula")
    parser.add_argument("--retry", type=int, default=3, help="Número de tentativas por chave")
    parser.add_argument("--download", action="store_true", help="Força o download/atualização dos dados antes de subir")
    
    args = parser.parse_args()
    
    target_items = [i.strip() for i in args.items.split(",")] if args.items else []

    if args.download:
        run_download_updates()

    # Pre-scan
    items_to_process = get_items_to_process(args, target_items)

    start_time = time.time()
    logger.info("="*60)
    logger.info(f"Iniciando Sincronização Redis - {datetime.now()}")
    logger.info(f"Itens identificados para processamento ({len(items_to_process)}):")
    for item in items_to_process:
        logger.info(f"  -> {item}")
    logger.info("="*60)
    
    if not items_to_process:
        logger.warning("Nenhum item encontrado para processar.")
        return

    # Se nenhum argumento de categoria for passado, e nem items específicos, faz tudo
    run_all = not (args.indices or args.raw or args.meta) or target_items

    if run_all or args.indices:
        sync_b3_indices(target_items)
    
    if run_all or args.raw:
        sync_raw_data(target_items)
        
    if run_all or args.meta:
        sync_assets_metadata(target_items)

    end_time = time.time()
    duracao = end_time - start_time
    
    logger.info("\n" + "="*60)
    logger.info("RESUMO DA EXECUÇÃO")
    logger.info("="*60)
    logger.info(f"Duração: {duracao:.2f} segundos")
    logger.info(f"Total de itens processados: {stats['total']}")
    logger.info(f"Sucessos: {stats['success']}")
    logger.info(f"Falhas: {stats['failed']}")
    
    if stats["failed_items"]:
        logger.info("\nItens que falharam:")
        for item in stats["failed_items"]:
            logger.info(f"  - {item}")
        
        # Sugestão de comando para retry
        retry_cmd = f"python scripts/sync_to_redis.py --items {','.join(stats['failed_items'])}"
        logger.info(f"\nPara tentar novamente apenas estes itens, use:\n{retry_cmd}")
    else:
        logger.info("\nTodos os itens foram sincronizados com sucesso!")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()
