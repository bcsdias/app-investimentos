import streamlit as st
import os
import sys

# Garante que a raiz do projeto esteja no path para imports de app.* e src.*
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.logger import logger
from src.data.user_store import load_dlp_token

logger.info("Iniciando Aplicação Investimentos V3")

# Configuração da Página (Deve ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="Investimentos V3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inicialização de Sessão Global (Sempre no topo) ---
if "custom_indices" not in st.session_state:
    st.session_state.custom_indices = []
if "benchmarks_deleted" not in st.session_state:
    st.session_state.benchmarks_deleted = set()
if "custom_portfolios" not in st.session_state:
    st.session_state.custom_portfolios = [{"name": "Carteira 1", "weights": {}}]
if "dlp_token" not in st.session_state:
    st.session_state.dlp_token = None

# Authentication Guard
if not st.user.is_logged_in:
    # Centralizing logo and tagline
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h1 style='text-align: center;'>📊 Investimentos V3</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Sua plataforma inteligente de gestão e análise de ativos.</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.info("Por favor, faça login para acessar o sistema.")
        if st.button("Entrar com Google", use_container_width=True, type="primary"):
            st.login()
        
        # Modo de Desenvolvimento / Convidado para teste local
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Acesso Técnico (Modo Dev)", use_container_width=True):
            st.session_state["dev_mode"] = True
            st.rerun()

    st.stop()

# --- Authenticated Area ---
# Verifica se está logado via Google OU se entrou via modo Dev
is_authenticated = st.user.is_logged_in or st.session_state.get("dev_mode", False)

if not is_authenticated:
    st.stop()

# --- Carregamento Automático do Token após Login ---
if st.session_state.dlp_token is None:
    try:
        user_email = st.user.get("email", "Usuário")
        st.session_state.dlp_token = load_dlp_token(user_email)
        if st.session_state.dlp_token:
            logger.info(f"Token DLP carregado automaticamente para {user_email}")
    except Exception as e:
        logger.error(f"Erro ao carregar token global: {e}")

if "dev_mode" in st.session_state and st.session_state.dev_mode:
    st.warning("⚠️ Você está acessando em **Modo de Desenvolvedor**. Algumas funções de API podem não funcionar.")

# Sidebar User Info and Logout
with st.sidebar:
    st.markdown("---")
    user_email = st.user.get("email", "Usuário")
    st.write(f"👤 **{user_email}**")
    if st.button("Sair (Logout)", type="primary", use_container_width=True):
        st.logout()
    st.markdown("---")

st.title("📊 Dashboard de Investimentos V3")
st.markdown(f"Bem-vindo, **{st.user.get('email', 'Usuário')}**, à nova versão modularizada do sistema.")
st.markdown("👈 **Navegue pelas páginas no menu lateral.**")
