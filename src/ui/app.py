import streamlit as st
import os
import sys

# Garante que a raiz do projeto esteja no path para imports de app.* e src.*
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

st.set_page_config(
    page_title="Investimentos V3 (Professional)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializa sessão global
if "custom_indices" not in st.session_state:
    st.session_state.custom_indices = []
if "benchmarks_deleted" not in st.session_state:
    st.session_state.benchmarks_deleted = set()
if "custom_portfolios" not in st.session_state:
    st.session_state.custom_portfolios = [{"name": "Carteira 1", "weights": {}}]

st.title("📊 Dashboard de Investimentos V3")
st.markdown("Bem-vindo à nova versão modularizada e profissionalizada do sistema.")
st.markdown("👈 **Navegue pelas páginas no menu lateral.**")
