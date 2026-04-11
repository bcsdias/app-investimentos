import streamlit as st
from src.utils.logger import logger

st.title("⚖️ Risco e Retorno")
logger.info("Página Risco e Retorno acessada")

# 1. Recupera o report da sessão
report = st.session_state.get("processed_report")

if not report or report.df_combined.empty:
    st.warning("⚠️ Nenhuma análise encontrada. Vá na página **4. Configurações** e clique em **Gerar Análises**.")
    st.stop()

st.success(f"🛡️ Análise de Risco baseada nos dados de: **{st.session_state.last_calc_at.strftime('%d/%m/%Y %H:%M')}**")

tab1, tab2, tab3 = st.tabs(["Queda Máxima (Drawdown)", "Volatilidade Móvel", "Risco x Retorno (Scatter)"])

with tab1:
    st.markdown("### Queda Máxima (Drawdown)")
    st.caption("O Drawdown mede a queda percentual do preço de um ativo desde o seu último pico máximo. Ilustra o 'pior momento' histórico.")
    
    fig_dd, df_dd = report.plot_drawdown(return_fig=True)
    if fig_dd:
        st.pyplot(fig_dd)
    else:
        st.info("Dados insuficientes para calcular Drawdown.")

with tab2:
    st.markdown("### Volatilidade Móvel")
    st.caption("Mede o grau de oscilação (desvio padrão anualizado) da carteira em janelas móveis de 252 dias úteis.")
    
    fig_vol, df_vol = report.plot_rolling_volatility(window=252, return_fig=True)
    if fig_vol:
        st.pyplot(fig_vol)
    else:
        st.info("Dados insuficientes para calcular Volatilidade Móvel.")
    
with tab3:
    st.markdown("### Matriz Risco x Retorno")
    st.caption("Compara a rentabilidade anualizada (CAGR) com a volatilidade. O ideal é estar no canto superior esquerdo.")
    
    fig_scatter, df_scatter = report.plot_risk_return_scatter(return_fig=True)
    if fig_scatter:
        st.pyplot(fig_scatter)
        st.dataframe(df_scatter.style.format("{:.2%}"), use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar a matriz de dispersão.")
