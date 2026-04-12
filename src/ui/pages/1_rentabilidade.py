import streamlit as st
import sys
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from src.ui.components.charts import render_altair_line
from src.ui.components.theme import apply_theme, render_theme_toggle
from src.ui.components.headers import render_page_header
from src.utils.logger import logger

# Configuração da Página
st.set_page_config(page_title="Rentabilidade", page_icon="💰", layout="wide")

# Aplica Tema Visual
apply_theme()

# Sidebar
with st.sidebar:
    render_theme_toggle()

logger.info("Página Rentabilidade acessada")

# 1. Recupera o report da sessão
report = st.session_state.get("processed_report")

if not report or report.df_combined.empty:
    render_page_header(title="Rentabilidade", icon="💰", description="Análise de performance da carteira.")
    st.warning("⚠️ **Nenhuma análise encontrada.** Vá na página **4. Configurações** e clique em **Gerar Análises**.")
    st.stop()

# Cálculo de KPIs para o Header
metrics = []
try:
    if "Carteira" in report.df_combined.columns:
        last_val = (report.df_combined["Carteira"].iloc[-1] - 1) * 100
        prev_val = (report.df_combined["Carteira"].iloc[-2] - 1) * 100
        delta = last_val - prev_val
        metrics.append({"label": "TWR Total", "value": f"{last_val:.2f}%", "delta": f"{delta:.2f}%"})
    
    # Busca TIR se disponível no dataframe de evolução
    # ... aqui poderíamos pegar a TIR final do report ...
    metrics.append({"label": "Última Atualização", "value": st.session_state.last_calc_at.strftime('%d/%m/%Y'), "delta_color": "off"})
except:
    pass

render_page_header(
    title="Rentabilidade", 
    icon="💰", 
    description="Análise detalhada de performance utilizando TWR (Time-Weighted Return) e TIR (Taxa Interna de Retorno).",
    metrics=metrics
)

# --- Conteúdo Principal em Card ---
with st.container(border=True):
    tab1, tab2, tab3 = st.tabs(["📈 Evolução Patrimonial (TWR)", "📊 Taxa Interna de Retorno (TIR)", "🔄 Simulador de Aportes"])

    with tab1:
        st.markdown("#### TWR (Time-Weighted Return)")
        st.caption("A rentabilidade TWR neutraliza o efeito de aportes e retiradas, focando puramente na performance dos ativos escolhidos.")
        
        render_altair_line(
            report.df_combined, 
            title="Evolução TWR da Carteira vs Benchmarks",
            y_title="Acumulado (Base 1.0)",
            y_format=".2f"
        )

    with tab2:
        st.markdown("#### TIR (Taxa Interna de Retorno)")
        st.caption("Representa a taxa de juros real que você obteve, considerando o montante e o exato momento dos seus aportes.")
        
        fig_irr, df_irr = report.plot_irr_evolution(return_fig=True)
        if fig_irr:
            st.pyplot(fig_irr)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_irr.to_frame("TIR %").T, use_container_width=True)
        else:
            st.info("Dados insuficientes para calcular a evolução da TIR.")

    with tab3:
        st.markdown("#### Simulador de Shadow Portfolios")
        st.caption("Compara sua carteira real contra o que aconteceria se você tivesse investido exatamente o mesmo valor nos benchmarks.")
        
        fig_shadow, df_shadow = report.simulate_shadow_portfolios(return_fig=True)
        if fig_shadow:
            st.pyplot(fig_shadow)
            with st.expander("🔍 Ver valores patrimoniais simulados"):
                st.dataframe(df_shadow, use_container_width=True)
        else:
            st.info("Dados insuficientes para simulação de aportes.")
