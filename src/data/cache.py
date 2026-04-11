import pandas as pd
import json
import time
import logging
import io
import streamlit as st
from upstash_redis import Redis
from typing import Any
from src.utils.logger import logger

# Cache em memória (Local fallback)
_LOCAL_CACHE = {}

def _get_redis_client():
    try:
        if "upstash_redis" in st.secrets:
            config = st.secrets["upstash_redis"]
            if "url" in config and "token" in config:
                return Redis(url=config["url"], token=config["token"])
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

def cache_set(key: str, value: Any, ttl: int):
    """
    Salva dados no cache. Suporta pd.Series (com metadados) ou dados JSON simples.
    """
    try:
        # 1. L1 Cache
        _LOCAL_CACHE[key] = (value, time.time() + ttl)
        
        # 2. Redis
        redis = _get_redis_client()
        if redis:
            if isinstance(value, pd.Series):
                # Wrapper para identificar como série no carregamento
                payload = json.loads(value.to_frame().to_json(orient='table'))
                data_to_save = {"_type": "pd.Series", "payload": payload}
            elif isinstance(value, pd.DataFrame):
                 payload = json.loads(value.to_json(orient='table'))
                 data_to_save = {"_type": "pd.DataFrame", "payload": payload}
            else:
                data_to_save = value
            
            redis.set(key, json.dumps(data_to_save), ex=ttl)
            logger.info(f"Cache SET (Redis): {key} | TTL: {ttl}s")
            
    except Exception as e:
        logger.warning(f"Erro ao salvar cache {key}: {e}")

def cache_clear_local():
    _LOCAL_CACHE.clear()
