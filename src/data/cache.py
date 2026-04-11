import pandas as pd
import json
import time
import logging
import io
import streamlit as st
from upstash_redis import Redis
from typing import Any
from src.utils.logger import logger

import os
from dotenv import load_dotenv

# Carrega .env se existir (para scripts externos)
load_dotenv()

# Cache em memória (Local fallback)
_LOCAL_CACHE = {}

def _get_redis_client():
    try:
        # 1. Tenta Streamlit Secrets (Produção/App)
        if hasattr(st, "secrets") and "upstash_redis" in st.secrets:
            config = st.secrets["upstash_redis"]
            if "url" in config and "token" in config:
                return Redis(url=config["url"], token=config["token"])
        
        # 2. Tenta Variáveis de Ambiente (Scripts locais/CI)
        url = os.getenv("UPSTASH_REDIS_REST_URL")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            return Redis(url=url, token=token)
            
    except Exception:
        pass
    return None

def cache_get(key: str) -> Any:
    """
    Recupera dados do cache. 
    Se o dado for uma string JSON começando com '{"_type": "pd.Series"', 
    converte de volta para pandas.
    """
    # 1. Memória
    if key in _LOCAL_CACHE:
        val, expiry = _LOCAL_CACHE[key]
        if time.time() < expiry:
            logger.debug(f"Cache HIT (Memória): {key}")
            return val
        else:
            del _LOCAL_CACHE[key]

    # 2. Redis
    redis = _get_redis_client()
    if redis:
        try:
            raw = redis.get(key)
            if raw:
                logger.info(f"Cache HIT (Redis): {key}")
                data = json.loads(raw)
                
                # Verifica se é uma série pandas serializada por nós
                # Recupera Séries ou DataFrames pandas
                if isinstance(data, dict) and data.get("_type") in ["pd.Series", "pd.DataFrame"]:
                    # Usa io.StringIO para evitar o FutureWarning do pandas
                    json_payload = json.dumps(data["payload"])
                    result = pd.read_json(io.StringIO(json_payload), orient='table')
                    
                    if data["_type"] == "pd.Series":
                        # Se for série, garante que retorne a primeira coluna como série
                        if isinstance(result, pd.DataFrame):
                            result = result.iloc[:, 0]
                    
                    # Cache L1
                    _LOCAL_CACHE[key] = (result, time.time() + 60)
                    return result
                
                # Caso contrário, retorna o dado como está (dict, list, str, etc)
                _LOCAL_CACHE[key] = (data, time.time() + 60)
                return data
        except Exception as e:
            logger.warning(f"Erro ao ler cache Redis {key}: {e}")
    
    return None

def cache_set(key: str, value: Any, ttl: int = None) -> bool:
    """
    Salva dados no cache. Suporta pd.Series (com metadados), pd.DataFrame ou dados JSON simples.
    Retorna True se conseguir salvar no Redis (ou se Redis estiver offline), False se houver erro de serialização.
    """
    try:
        # 1. L1 Cache
        # Se ttl for None, usamos 1 hora para o cache local (L1) para evitar transbordamento
        l1_ttl = ttl if ttl is not None else 3600
        _LOCAL_CACHE[key] = (value, time.time() + l1_ttl)
        
        # 2. Redis
        redis = _get_redis_client()
        if redis:
            data_to_save = value
            
            if isinstance(value, (pd.Series, pd.DataFrame)):
                # Pre-processamento para evitar erro de MultiIndex em orient='table'
                temp_df = value.to_frame() if isinstance(value, pd.Series) else value.copy()
                
                # Se as colunas forem MultiIndex, achata para colunas simples
                if isinstance(temp_df.columns, pd.MultiIndex):
                    temp_df.columns = [
                        "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else str(col)
                        for col in temp_df.columns.values
                    ]
                
                payload = json.loads(temp_df.to_json(orient='table'))
                data_to_save = {
                    "_type": "pd.Series" if isinstance(value, pd.Series) else "pd.DataFrame",
                    "payload": payload
                }
            
            # Redis SET (ex=None significa sem expiração)
            redis.set(key, json.dumps(data_to_save), ex=ttl)
            logger.info(f"Cache SET (Redis): {key} | TTL: {ttl if ttl else 'PERMANENT'}")
        
        return True
            
    except Exception as e:
        logger.error(f"Erro ao salvar cache {key}: {e}")
        return False

def cache_clear_local():
    _LOCAL_CACHE.clear()
