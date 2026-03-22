import streamlit as st
import os

st.title("🔄 Planejador de Migração (IR)")

st.markdown("Planeje a transição da sua carteira atual para uma nova alocação, respeitando o limite mensal de isenção de IR para Ações (R$ 20.000,00).")

env_token = os.getenv('DLP_TOKEN', '')
token = st.text_input("Token API (DLP)", value=env_token, type="password", key="token_migracao")

if not token:
    st.info("Insira seu token da DLP para carregar a carteira atual.")
else:
    st.success("Configuração ativa. O script de otimização de IR será exibido aqui assim que acionado.")
