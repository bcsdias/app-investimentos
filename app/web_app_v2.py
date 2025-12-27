import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Configuração da Página (Deve ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Investimentos V2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adiciona o diretório raiz ao path para permitir importações dos módulos locais
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Importa a classe de lógica do main_v2
from app.main_v2 import FinancialReport

# Carrega variáveis de ambiente
load_dotenv()

# --- Classe de Logger para Streamlit ---
class StreamlitLogger:
    """Redireciona logs para a interface do Streamlit (Toast e Sidebar)."""
    def info(self, msg):
        # Mostra mensagens curtas como toast flutuante
        if len(msg) < 80:
            st.toast(msg, icon="ℹ️")
        # Mensagens de sistema vão para o console do servidor também
        print(f"[INFO] {msg}")

    def debug(self, msg):
        # Debug só no console para não poluir
        print(f"[DEBUG] {msg}")

    def warning(self, msg):
        st.warning(msg)
        print(f"[WARN] {msg}")

    def error(self, msg):
        st.error(msg)
        print(f"[ERROR] {msg}")

# --- Função Principal do App ---
def main():
    st.title("📊 Dashboard de Investimentos V2")
    st.markdown("---")

    # --- Sidebar: Configurações ---
    with st.sidebar:
        st.header("Configurações")
        
        # Token
        env_token = os.getenv('DLP_TOKEN', '')
        token = st.text_input("Token API (DLP)", value=env_token, type="password")
        
        st.divider()
        
        # Modo de Operação
        modo = st.radio("Modo de Análise", ["Mercado (Benchmarks)", "Carteira DLP Invest"])
        
        ativo = None
        classe = None
        historico = 5
        
        if modo == "Carteira DLP Invest":
            tipo_filtro = st.selectbox("Filtrar por", ["Classe de Ativo", "Ativo Específico"])
            if tipo_filtro == "Classe de Ativo":
                classe = st.text_input("Nome da Classe", value="AÇÃO").upper()
            else:
                ativo = st.text_input("Código do Ativo", value="PETR4").upper()
                
            simular_aportes = st.checkbox("Simular Aportes (Shadow Portfolio)", value=True)
            
        else: # Modo Mercado
            historico = st.slider("Anos de Histórico", min_value=1, max_value=20, value=5)
            simular_aportes = False

        st.divider()
        btn_processar = st.button("🚀 Gerar Relatório", type="primary", use_container_width=True)

    # --- Processamento ---
    if btn_processar:
        if not token and modo == "Carteira DLP Invest":
            st.error("Por favor, informe o Token da API.")
            return

        # Inicializa o Report com nosso Logger customizado
        logger = StreamlitLogger()
        report = FinancialReport(logger)
        
        # Placeholder para status
        status_text = st.empty()
        status_text.info("Iniciando processamento...")

        try:
            # 1. Busca Dados
            user_series = None
            nome_analise = ""
            
            if modo == "Carteira DLP Invest":
                nome_analise = ativo if ativo else classe
                status_text.info(f"Buscando dados da carteira: {nome_analise}...")
                user_series = report.fetch_user_portfolio(token, ativo=ativo, classe=classe)
                
                if user_series is None:
                    st.error("Não foi possível obter dados da carteira. Verifique o Token ou o Ativo/Classe.")
                    return
            else:
                nome_analise = f"Mercado_{historico}anos"
                status_text.info(f"Buscando dados de mercado ({historico} anos)...")

            # 2. Constrói Dataset
            status_text.info("Consolidando benchmarks e calculando indicadores...")
            report.build_dataset(user_series=user_series, years_history=historico if modo != "Carteira DLP Invest" else None)

            if report.df_combined.empty:
                st.error("Nenhum dado disponível para gerar gráficos.")
                return

            status_text.success("Dados processados com sucesso!")
            
            # --- Exibição dos Resultados ---
            
            # Métricas de Topo (KPIs)
            if not report.df_combined.empty:
                df = report.df_combined
                retorno_total = (df.iloc[-1] / df.iloc[0]) - 1
                
                cols = st.columns(len(df.columns))
                for i, col_name in enumerate(df.columns):
                    val = retorno_total[col_name]
                    with cols[i]:
                        st.metric(label=col_name, value=f"{val:.1%}")

            # Abas para organização
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Rentabilidade & Risco", "📉 Drawdown & Volatilidade", "💰 Simulação & TIR", "📋 Dados Brutos"])

            with tab1:
                st.subheader("Evolução TWR (Time-Weighted Return)")
                fig_twr = report.plot_twr_evolution(title_suffix=nome_analise, return_fig=True)
                if fig_twr: st.pyplot(fig_twr)

                st.subheader("Risco x Retorno")
                fig_risk = report.plot_risk_return_scatter(title_suffix=nome_analise, return_fig=True)
                if fig_risk: st.pyplot(fig_risk)

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Drawdown (Queda Máxima)")
                    fig_dd = report.plot_drawdown(title_suffix=nome_analise, return_fig=True)
                    if fig_dd: st.pyplot(fig_dd)
                
                with col2:
                    st.subheader("Volatilidade Móvel (Risco)")
                    fig_vol = report.plot_rolling_volatility(title_suffix=nome_analise, return_fig=True)
                    if fig_vol: st.pyplot(fig_vol)

                st.subheader("Sharpe Ratio Móvel (Eficiência)")
                fig_sharpe = report.plot_rolling_sharpe(title_suffix=nome_analise, return_fig=True)
                if fig_sharpe: st.pyplot(fig_sharpe)

            with tab3:
                if modo == "Carteira DLP Invest":
                    col_tir, col_sim = st.columns(2)
                    
                    with col_tir:
                        st.subheader("Evolução da TIR (Rentabilidade Real)")
                        fig_tir = report.plot_irr_evolution(title_suffix=nome_analise, return_fig=True)
                        if fig_tir: 
                            st.pyplot(fig_tir)
                        else:
                            st.info("Dados insuficientes para cálculo da TIR.")

                    with col_sim:
                        if simular_aportes:
                            st.subheader("Simulação de Aportes (Shadow Portfolio)")
                            fig_sim = report.simulate_shadow_portfolios(title_suffix=nome_analise, return_fig=True)
                            if fig_sim: st.pyplot(fig_sim)
                        else:
                            st.info("Simulação de aportes desativada.")
                else:
                    st.info("Análises de TIR e Simulação de Aportes disponíveis apenas no modo 'Carteira DLP Invest'.")

            with tab4:
                st.subheader("Tabela Resumo de Rentabilidade")
                # Recalcula a tabela resumo para exibir
                # (O método generate_summary_table salva CSV, aqui vamos recriar a lógica simples para exibir)
                df = report.df_combined
                yearly = df.resample('YE').last().pct_change()
                # Ajuste primeiro ano
                yearly.iloc[0] = (df.resample('YE').last().iloc[0] / df.iloc[0]) - 1
                
                # Formatação para exibição
                st.dataframe(yearly.style.format("{:.2%}"), use_container_width=True)
                
                st.subheader("Dataset Consolidado (Download)")
                st.dataframe(report.df_combined)
                
                # Botão de Download
                csv = report.df_combined.to_csv(sep=';', decimal=',').encode('utf-8')
                st.download_button(
                    label="📥 Baixar Dados Consolidados (CSV)",
                    data=csv,
                    file_name=f"dados_consolidados_{nome_analise}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Ocorreu um erro durante a execução: {str(e)}")
            # Em produção, logar o traceback completo no console
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
