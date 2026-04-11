import os
import pandas as pd
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_STATIC_DIR = os.path.join(BASE_DIR, "data", "static")

def get_b3_index(ticker: str, start_date: str, end_date: str) -> pd.Series | None:
    """
    Lê os dados históricos de um índice da B3 a partir dos arquivos CSV estáticos.
    Esses arquivos são gerados via GitHub Actions (scripts/update_b3_cache.py).
    
    Args:
        ticker: O símbolo do índice (ex: 'IBOV', 'IFIX').
        start_date: Data inicial (YYYY-MM-DD).
        end_date: Data final (YYYY-MM-DD).
        
    Returns:
        pd.Series de fechamento diário.
    """
    file_path = os.path.join(DATA_STATIC_DIR, f"{ticker}_all.csv")
    
    if not os.path.exists(file_path):
        logging.getLogger(__name__).warning(f"Arquivo não encontrado para o índice B3: {file_path}")
        return None
        
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        df = df.sort_index()
        
        # Filtro de datas
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        series = df.loc[mask, 'Close'].dropna()
        
        if series.empty:
            return None
            
        series.name = ticker
        return series
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Erro ao ler os dados estáticos da B3 para {ticker}: {e}")
        return None
