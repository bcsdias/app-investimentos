import streamlit as st
import os
from src.utils.logger import logger

st.title("🔄 Planejador de Migração (IR)")
logger.info("Página Migração de IR acessada")

st.markdown("Planeje a transição da sua carteira atual para uma nova alocação, respeitando o limite mensal de isenção de IR para Ações.")

# 1. Recupera o report da sessão
report = st.session_state.get("processed_report")

if not report:
    st.warning("⚠️ Nenhuma análise encontrada. Vá na página **4. Configurações** e clique em **Gerar Análises**.")
    st.stop()

st.success(f"✅ Dados da Carteira carregados (**{st.session_state.last_calc_at.strftime('%d/%m/%Y %H:%M')}**)")

if hasattr(report, 'portfolio_df') and report.portfolio_df is not None:
    ultimo_valor = report.portfolio_df['vlr_mercado'].iloc[-1]
    st.metric("Patrimônio Total Atual", f"R$ {ultimo_valor:,.2f}")
    
    st.info("💡 O motor de Otimização de IR (Venda Gradual) será integrado na próxima fase para utilizar estes dados Reais.")
    st.dataframe(report.portfolio_df.tail(10), use_container_width=True)
else:
    st.error("Não foi possível carregar o detalhamento da carteira.")
