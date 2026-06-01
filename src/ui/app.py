import streamlit as st
import os
import sys
import pandas as pd

# Garante que a raiz do projeto esteja no path para imports de app.* e src.*
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.utils.logger import logger
from src.data.user_store import load_dlp_token
from src.ui.components.theme import apply_theme, render_theme_toggle
from src.ui.components.headers import render_page_header

logger.info("Iniciando Aplicação Investimentos V3")

# Configuração da Página (Deve ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="Investimentos V3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializa e aplica o tema customizado
apply_theme()

# --- Inicialização de Sessão Global ---
if "custom_indices" not in st.session_state:
    st.session_state.custom_indices = []
if "benchmarks_deleted" not in st.session_state:
    st.session_state.benchmarks_deleted = set()
if "custom_portfolios" not in st.session_state:
    st.session_state.custom_portfolios = [{"name": "Carteira 1", "weights": {}}]
if "dlp_token" not in st.session_state:
    st.session_state.dlp_token = None

# Authentication Guard
is_logged_in = st.user.is_logged_in if hasattr(st.user, "is_logged_in") else False
dev_mode = st.session_state.get("dev_mode", False)

if not is_logged_in and not dev_mode:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h1 style='text-align: center;'>📊 Investimentos V3</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Sua plataforma inteligente de gestão e análise de ativos.</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.info("Por favor, faça login para acessar o sistema.")
        if st.button("Entrar com Google", use_container_width=True, type="primary"):
            st.login()
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Acesso Técnico (Modo Dev)", use_container_width=True):
            st.session_state["dev_mode"] = True
            st.rerun()

    st.stop()

# --- Authenticated Area ---
is_authenticated = is_logged_in or dev_mode
if not is_authenticated:
    st.stop()

# Carregamento Automático do Token
if st.session_state.dlp_token is None:
    try:
        user_email = st.user.get("email", "Usuário")
        st.session_state.dlp_token = load_dlp_token(user_email)
    except Exception as e:
        logger.error(f"Erro ao carregar token global: {e}")

# Sidebar UI
with st.sidebar:
    render_theme_toggle()
    st.markdown("---")
    user_email = st.user.get("email", "Usuário")
    st.write(f"👤 **{user_email}**")
    if st.button("Sair (Logout)", type="primary", use_container_width=True):
        st.logout()
    st.markdown("---")

# Main Content
report = st.session_state.get("processed_report")
user_name = st.user.get('name', 'Usuário')

# Header Principal
metrics = []
if report and not report.df_combined.empty:
    try:
        # Tenta pegar a última rentabilidade TWR e TIR
        last_twr = (report.df_combined['Carteira'].iloc[-1] - 1) * 100
        delta_twr = (report.df_combined['Carteira'].iloc[-1] - report.df_combined['Carteira'].iloc[-2]) * 100
        
        metrics = [
            {"label": "Rentabilidade TWR Total", "value": f"{last_twr:.2f}%", "delta": f"{delta_twr:.2f}%"},
            {"label": "Última Atualização", "value": st.session_state.last_calc_at.strftime('%d/%m/%Y'), "delta_color": "off"}
        ]
    except:
        pass

render_page_header(
    title="Dashboard Executivo", 
    icon="🚀", 
    description=f"Bem-vindo, {user_name}. Aqui está o resumo da sua última análise.",
    metrics=metrics
)

if not report:
    st.warning("👋 **Bem-vindo!** Nenhuma análise foi gerada ainda.")
    st.markdown("""
        Para começar, siga estes passos:
        1. Vá em **4. Configurações** no menu lateral.
        2. Insira ou valide seu **Token DLP**.
        3. Selecione os ativos e clique em **Gerar Análises**.
    """)
else:
    # Mostra um pequeno resumo visual se os dados existirem
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("💡 Destaque da Carteira")
            st.write("Sua carteira está ativa e os dados históricos foram processados com sucesso.")
            if "Carteira" in report.df_combined.columns:
                st.info(f"O período analisado vai de {report.df_combined.index[0].strftime('%d/%m/%Y')} até {report.df_combined.index[-1].strftime('%d/%m/%Y')}.")
    
    with col2:
        with st.container(border=True):
            st.subheader("🔔 Atalhos Rápidos")
            c1, c2 = st.columns(2)
            if c1.button("Ver Rentabilidade", use_container_width=True):
                st.switch_page("pages/1_rentabilidade.py")
            if c2.button("Ver Risco", use_container_width=True):
                st.switch_page("pages/2_risco.py")

if "dev_mode" in st.session_state and st.session_state.dev_mode:
    st.toast("⚠️ Modo de Desenvolvedor Ativo", icon="🛠️")
