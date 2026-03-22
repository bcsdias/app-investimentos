import streamlit as st
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from src.ui.components.sidebar import render_sidebar_asset_selection, get_expanded_assets

st.title("⚙️ Configurações e Benchmarks")

with st.sidebar:
    st.header("1. Ativos Base")
    selected_assets = render_sidebar_asset_selection()

st.subheader("2. Seleção de Benchmarks de Comparação")

# Utilizando data_editor (extraído do web_app.py)
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
with col_d1:
    default_start = pd.Timestamp.today() - pd.DateOffset(years=1)
    data_inicio = st.date_input("Início", value=default_start)
with col_d2:
    data_fim = st.date_input("Fim", value="today")

if st.button("Aplicar Configurações"):
    st.success("Dados salvos na sessão! Vá para a aba de Rentabilidade para visualizar os gráficos.")
