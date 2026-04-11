import streamlit as st
from src.data.cache import cache_set, cache_get
import logging

logger = logging.getLogger("teste")
# Teste de escrita
cache_set("teste_conexao", "Funcionou!", 300, logger)

# Teste de leitura
resultado = cache_get("teste_conexao", logger)
print(f"Resultado do Cache: {resultado}")
