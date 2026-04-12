import streamlit as st
import pandas as pd
import os
import sys

# Garante que a raiz do projeto esteja no path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.ui.components.theme import apply_theme, render_theme_toggle
from src.ui.components.headers import render_page_header
from src.ui.components.sidebar import render_sidebar_asset_selection, get_expanded_assets
from src.utils.logger import logger
from src.data.user_store import load_dlp_token, save_dlp_token, delete_dlp_token
from src.engine.financial_report import FinancialReport

# Configuração da Página
st.set_page_config(page_title="Configurações", page_icon="⚙️", layout="wide")

# Aplica Tema Visual
apply_theme()

# 1. Inicialização do Token do Usuário
if "dlp_token" not in st.session_state:
    try:
        user_email = st.user.get("email", "Usuário")
        token = load_dlp_token(user_email)
        st.session_state.dlp_token = token
    except Exception:
        st.session_state.dlp_token = None

# Sidebar - Tema e Ativos
with st.sidebar:
    render_theme_toggle()
    st.markdown("---")
    selected_assets = render_sidebar_asset_selection()

# Header
render_page_header(
    title="Configurações e Benchmarks", 
    icon="⚙️", 
    description="Gerencie seu acesso aos dados e selecione os benchmarks para comparação."
)

# --- Seção 1: Token em Card ---
with st.container(border=True):
    st.markdown("#### 🔑 Token DLP (Acesso à Planilha)")
    col_t1, col_t2 = st.columns([0.7, 0.3])
    
    with col_t1:
        if st.session_state.dlp_token:
            masked = st.session_state.dlp_token[:4] + "****"
            st.success(f"Token configurado: `{masked}`")
        else:
            st.warning("⚠️ Nenhum token configurado.")
        
        new_token = st.text_input("Atualizar Token", type="password", placeholder="Insira o seu Token DLP aqui...")
    
    with col_t2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Salvar", use_container_width=True, type="primary"):
            if new_token:
                try:
                    save_dlp_token(st.user.get("email", "Usuário"), new_token)
                    st.session_state.dlp_token = new_token
                    st.success("Salvo!")
                    st.rerun()
                except: st.error("Erro!")
        if st.session_state.dlp_token and st.button("Remover", use_container_width=True):
            delete_dlp_token(st.user.get("email", "Usuário"))
            st.session_state.dlp_token = None
            st.rerun()

# --- Seção 2: Benchmarks em Card ---
st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("#### 📊 Seleção de Benchmarks de Comparação")
    if selected_assets:
        expanded_assets = get_expanded_assets(selected_assets)
        visible_assets = [a for a in expanded_assets if a not in st.session_state.benchmarks_deleted]
        
        df_bench = pd.DataFrame({"Ativo": visible_assets, "Incluir": True})
        edited_bench = st.data_editor(
            df_bench, 
            column_config={
                "Ativo": st.column_config.TextColumn("Ativo", disabled=True),
                "Incluir": st.column_config.CheckboxColumn("Plotar", width="small"),
            },
            hide_index=True,
            use_container_width=True,
            key="editor_bench_individual"
        )
    else:
        st.info("👈 Selecione ativos na aba lateral para começar.")

# --- Seção 3: Parâmetros de Tempo em Card ---
st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("#### 📅 Parâmetros de Tempo")
    col_d1, col_d2 = st.columns(2)
    if "data_inicio" not in st.session_state:
        st.session_state.data_inicio = pd.Timestamp.today() - pd.DateOffset(years=1)
    if "data_fim" not in st.session_state:
        st.session_state.data_fim = pd.Timestamp.today()

    st.session_state.data_inicio = col_d1.date_input("Data Início", value=st.session_state.data_inicio)
    st.session_state.data_fim = col_d2.date_input("Data Fim", value=st.session_state.data_fim)

# --- Pipeline de Processamento ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔄 Gerar Análises", type="primary", use_container_width=True):
    if not st.session_state.get("dlp_token"):
        st.error("❌ Token DLP não configurado!")
        st.stop()
        
    with st.status("🚀 Processando dados financeiros...", expanded=True) as status:
        try:
            active_benchmarks = []
            if 'editor_bench_individual' in st.session_state:
                active_benchmarks = [row['Ativo'] for row in edited_bench.to_dict('records') if row['Incluir']]
            
            report = FinancialReport()
            user_series = report.fetch_user_portfolio(
                st.session_state.dlp_token,
                start_date=st.session_state.data_inicio.strftime('%Y-%m-%d'),
                end_date=st.session_state.data_fim.strftime('%Y-%m-%d')
            )
            report.build_dataset(
                user_series=user_series,
                active_benchmarks=active_benchmarks,
                start_date=st.session_state.data_inicio.strftime('%Y-%m-%d'),
                end_date=st.session_state.data_fim.strftime('%Y-%m-%d')
            )
            
            st.session_state.processed_report = report
            st.session_state.last_calc_at = pd.Timestamp.today()
            status.update(label="✅ Sucesso! Vá para as páginas de análise.", state="complete")
            st.balloons()
            st.rerun()
        except Exception as e:
            logger.error(f"Erro: {e}")
            status.update(label=f"❌ Erro: {e}", state="error")
