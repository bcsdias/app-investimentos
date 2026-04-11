import streamlit as st
import pandas as pd
import os
import sys

# Garante que a raiz do projeto esteja no path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.ui.components.sidebar import render_sidebar_asset_selection, get_expanded_assets
from src.utils.logger import logger
from src.data.user_store import load_dlp_token, save_dlp_token, delete_dlp_token

st.set_page_config(page_title="Configurações", page_icon="⚙️")

# 1. Inicialização do Token do Usuário
if "dlp_token" not in st.session_state:
    try:
        # Carrega o token do Supabase usando o e-mail do usuário logado
        user_email = st.user.get("email", "Usuário")
        token = load_dlp_token(user_email)
        st.session_state.dlp_token = token
    except Exception:
        st.session_state.dlp_token = None

st.title("⚙️ Configurações e Benchmarks")
logger.info("Página Configurações acessada")

# --- Seção 1: Gerenciamento de Token DLP ---
st.subheader("1. Token DLP (Acesso à Planilha)")

col_token1, col_token2 = st.columns([0.7, 0.3])

with col_token1:
    if st.session_state.dlp_token:
        # Mascara o token para segurança (exibe apenas os 4 primeiros caracteres)
        masked = st.session_state.dlp_token[:4] + "****"
        st.success(f"Token configurado: `{masked}`")
    else:
        st.warning("⚠️ Nenhum token configurado para este usuário.")

    new_token = st.text_input(
        "Atualizar Token", 
        type="password", 
        placeholder="Insira o seu Token DLP aqui...",
        help="O token é armazenado de forma criptografada no banco de dados."
    )

with col_token2:
    st.write("") # Alinhamento vertical
    st.write("") # Alinhamento vertical
    if st.button("Salvar", use_container_width=True, type="primary"):
        if new_token:
            try:
                user_email = st.user.get("email", "Usuário")
                save_dlp_token(user_email, new_token)
                st.session_state.dlp_token = new_token
                st.success("Salvo!")
                st.rerun()
            except Exception as e:
                st.error("Erro ao salvar.")
        else:
            st.error("Erro!")
            
    if st.session_state.dlp_token:
        if st.button("Remover", use_container_width=True):
            try:
                user_email = st.user.get("email", "Usuário")
                delete_dlp_token(user_email)
                st.session_state.dlp_token = None
                st.rerun()
            except Exception:
                st.error("Erro!")

st.markdown("---")

# --- Seção 2: Benchmarks (Legado) ---

with st.sidebar:
    st.header("1. Ativos Base")
    selected_assets = render_sidebar_asset_selection()

st.subheader("2. Seleção de Benchmarks de Comparação")

if selected_assets:
    expanded_assets = get_expanded_assets(selected_assets)
    visible_assets = [a for a in expanded_assets if a not in st.session_state.benchmarks_deleted]
    
    with st.expander("Benchmarks Individuais (Plotar Linhas)", expanded=True):
        df_bench = pd.DataFrame({"Ativo": visible_assets, "Incluir": True})
        edited_bench = st.data_editor(
            df_bench, 
            column_config={
                "Ativo": st.column_config.TextColumn("Ativo", disabled=True),
                "Incluir": st.column_config.CheckboxColumn("Plotar", width="small"),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key="editor_bench_individual"
        )
else:
    st.info("👈 Selecione ativos na aba lateral para começar.")

st.markdown("---")
st.subheader("3. Parâmetros de Tempo")
col_d1, col_d2 = st.columns(2)

if "data_inicio" not in st.session_state:
    st.session_state.data_inicio = pd.Timestamp.today() - pd.DateOffset(years=1)
if "data_fim" not in st.session_state:
    st.session_state.data_fim = pd.Timestamp.today()

with col_d1:
    st.session_state.data_inicio = st.date_input("Início", value=st.session_state.data_inicio)
with col_d2:
    st.session_state.data_fim = st.date_input("Fim", value=st.session_state.data_fim)

st.markdown("<br>", unsafe_allow_html=True)

# 4. Pipeline de Processamento Final
from src.engine.financial_report import FinancialReport

if st.button("🔄 Gerar Análises", type="primary", use_container_width=True):
    if not st.session_state.get("dlp_token"):
        st.error("❌ Token DLP não configurado! Por favor, insira o token acima antes de gerar as análises.")
        st.stop()
        
    with st.status("🚀 Processando dados financeiros...", expanded=True) as status:
        try:
            # 1. Preparar lista de benchmarks selecionados
            active_benchmarks = []
            if 'editor_bench_individual' in st.session_state:
                df_editor = st.session_state.editor_bench_individual
                # st.data_editor returns dict with 'edited_rows', etc, or the DF if using standard editor
                # But we defined it with key="editor_bench_individual"
                active_benchmarks = [row['Ativo'] for row in edited_bench.to_dict('records') if row['Incluir']]
            
            st.write("📥 Baixando histórico da carteira...")
            report = FinancialReport()
            user_series = report.fetch_user_portfolio(
                st.session_state.dlp_token,
                start_date=st.session_state.data_inicio.strftime('%Y-%m-%d'),
                end_date=st.session_state.data_fim.strftime('%Y-%m-%d')
            )
            
            st.write("📊 Baixando cotações de benchmarks e índices...")
            report.build_dataset(
                user_series=user_series,
                active_benchmarks=active_benchmarks,
                start_date=st.session_state.data_inicio.strftime('%Y-%m-%d'),
                end_date=st.session_state.data_fim.strftime('%Y-%m-%d')
            )
            
            # Guardamos o report inteiro (ou os dados principais) na sessão
            st.session_state.processed_report = report
            st.session_state.last_calc_at = pd.Timestamp.today()
            
            status.update(label="✅ Análises geradas com sucesso! Vá para as páginas de Rentabilidade ou Risco.", state="complete")
            st.balloons()
            
        except Exception as e:
            logger.error(f"Erro no processamento de análise: {e}")
            status.update(label=f"❌ Erro no processamento: {e}", state="error")
