import streamlit as st
import sys
import os
from src.ui.components.theme import apply_theme, render_theme_toggle
from src.ui.components.headers import render_page_header
from src.utils.logger import logger

# Configuração da Página
st.set_page_config(page_title="Risco e Retorno", page_icon="⚖️", layout="wide")

# Aplica Tema Visual
apply_theme()

# Sidebar
with st.sidebar:
    render_theme_toggle()

logger.info("Página Risco e Retorno acessada")

# 1. Recupera o report da sessão
report = st.session_state.get("processed_report")

if not report or report.df_combined.empty:
    render_page_header(title="Risco e Retorno", icon="⚖️", description="Análise de risco da carteira.")
    st.warning("⚠️ **Nenhuma análise encontrada.** Vá na página **4. Configurações** e clique em **Gerar Análises**.")
    st.stop()

# Cálculo de KPIs para o Header
metrics = []
try:
    if "Carteira" in report.df_combined.columns:
        # Pega o drawdown atual (último ponto)
        _, df_dd = report.plot_drawdown(return_fig=True)
        if df_dd is not None:
            last_dd = df_dd["Carteira"].iloc[-1] * 100
            metrics.append({"label": "Drawdown Atual", "value": f"{last_dd:.2f}%", "delta_color": "inverse"})
        
        # Pega a volatilidade anualizada (última)
        _, df_vol = report.plot_rolling_volatility(window=252, return_fig=True)
        if df_vol is not None:
            last_vol = df_vol["Carteira"].iloc[-1] * 100
            metrics.append({"label": "Volatilidade (252d)", "value": f"{last_vol:.2f}%", "delta_color": "off"})
            
    metrics.append({"label": "Última Atualização", "value": st.session_state.last_calc_at.strftime('%d/%m/%Y'), "delta_color": "off"})
except:
    pass

render_page_header(
    title="Risco e Retorno", 
    icon="⚖️", 
    description="Avaliação de risco histórico, volatilidade móvel e eficiência da carteira (Matriz Scatter).",
    metrics=metrics
)

# --- Conteúdo Principal em Card ---
with st.container(border=True):
    tab1, tab2, tab3 = st.tabs(["📉 Queda Máxima (Drawdown)", "🌊 Volatilidade Móvel", "🎯 Risco x Retorno (Scatter)"])

    with tab1:
        st.markdown("#### Queda Máxima (Drawdown)")
        st.caption("O Drawdown mede a queda percentual do valor da carteira desde o seu último pico máximo. Ilustra o 'pior momento' histórico.")
        
        fig_dd, df_dd = report.plot_drawdown(return_fig=True)
        if fig_dd:
            st.pyplot(fig_dd)
        else:
            st.info("Dados insuficientes para calcular Drawdown.")

    with tab2:
        st.markdown("#### Volatilidade Móvel")
        st.caption("Mede o grau de oscilação (desvio padrão anualizado) da carteira em janelas móveis de 252 dias úteis.")
        
        fig_vol, df_vol = report.plot_rolling_volatility(window=252, return_fig=True)
        if fig_vol:
            st.pyplot(fig_vol)
        else:
            st.info("Dados insuficientes para calcular Volatilidade Móvel.")
        
    with tab3:
        st.markdown("#### Matriz Risco x Retorno")
        st.caption("Compara a rentabilidade anualizada (CAGR) com a volatilidade. O ideal é estar posicionado o mais alto e à esquerda possível.")
        
        fig_scatter, df_scatter = report.plot_risk_return_scatter(return_fig=True)
        if fig_scatter:
            st.pyplot(fig_scatter)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_scatter.style.format("{:.2%}"), use_container_width=True)
        else:
            st.info("Dados insuficientes para gerar a matriz de dispersão.")
