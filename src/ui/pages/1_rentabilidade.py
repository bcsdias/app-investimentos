import streamlit as st
import sys
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from src.ui.components.charts import render_altair_line
from src.utils.logger import logger
from src.engine.financial_report import FinancialReport

# Utilizando o novo motor puro refatorado
# Como o usuário pediu para colocar tudo funcionando na estrutura sem mexer na lógica, vou assumir from app.main ou src.engine.financial_report

st.title("💰 Rentabilidade")
logger.info("Página Rentabilidade acessada")

# 1. Recupera o report da sessão
report = st.session_state.get("processed_report")

if not report or report.df_combined.empty:
    st.warning("⚠️ Nenhuma análise encontrada. Vá na página **4. Configurações** e clique em **Gerar Análises**.")
    st.stop()

st.success(f"📈 Análise gerada em: **{st.session_state.last_calc_at.strftime('%d/%m/%Y %H:%M')}**")

# --- Tabs de Rentabilidade ---
tab1, tab2, tab3 = st.tabs(["Evolução Patrimonial (TWR)", "Taxa Interna de Retorno (TIR)", "Simulador de Aportes"])

with tab1:
    st.markdown("### TWR (Time-Weighted Return)")
    st.caption("Performance acumulada (Base 100). Compara a habilidade de escolha de ativos puramente, neutralizando o efeito de aportes.")
    
    # Usando o novo componente Altair interativo
    render_altair_line(
        report.df_combined, 
        title="Evolução TWR da Carteira vs Benchmarks",
        y_title="Retorno Acumulado %",
        y_format=".1f" # Como o report escala para 100, vamos mostrar valor absoluto ou %
    )

with tab2:
    st.markdown("### TIR (Taxa Interna de Retorno)")
    st.caption("Representa a taxa de juros real anualizada que você obteve, considerando o exato momento dos seus aportes.")
    
    fig_irr, df_irr = report.plot_irr_evolution(return_fig=True)
    if fig_irr:
        st.pyplot(fig_irr)
        st.dataframe(df_irr.to_frame("TIR %").T, use_container_width=True)
    else:
        st.info("Dados insuficientes para calcular a evolução da TIR.")

with tab3:
    st.markdown("### Simulador de Shadow Portfolios")
    st.caption("Compara sua Carteira Real contra o que aconteceria se você tivesse investido o mesmo dinheiro nos benchmarks.")
    
    fig_shadow, df_shadow = report.simulate_shadow_portfolios(return_fig=True)
    if fig_shadow:
        st.pyplot(fig_shadow)
        with st.expander("Ver valores patrimoniais simulados"):
            st.dataframe(df_shadow, use_container_width=True)
    else:
        st.info("Dados insuficientes para simulação de aportes.")
