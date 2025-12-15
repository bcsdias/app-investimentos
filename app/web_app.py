import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Adiciona o diretório raiz ao path para importar os módulos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.logger import setup_logger
from utils.market_data import (
    buscar_historico,
    processar_benchmarks
)
from app.main import (
    gerar_grafico_twr,
    gerar_grafico_comparativo_twr,
    gerar_twr_historico,
    gerar_analise_risco,
    simular_evolucao_patrimonio,
    gerar_grafico_evolucao,
    gerar_grafico_percentual
)

# Configuração da Página
st.set_page_config(
    page_title="Investimentos Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carrega variáveis de ambiente
load_dotenv()

# Configura Logger (para aparecer no terminal onde o streamlit roda)
logger = setup_logger(log_file='web_app.log')

# --- SIDEBAR: Configurações ---
st.sidebar.title("⚙️ Configurações")

if 'dlp_token' not in st.session_state:
    st.session_state.dlp_token = os.getenv('DLP_TOKEN', '')

if st.session_state.dlp_token:
    st.sidebar.success("Token configurado!")
    if st.sidebar.button("🗑️ Alterar/Remover Token"):
        st.session_state.dlp_token = ""
        st.rerun()
else:
    token_input = st.sidebar.text_input("API Token (DLP_TOKEN)", type="password")
    if token_input:
        st.session_state.dlp_token = token_input
        st.rerun()

token = st.session_state.dlp_token

modo_analise = st.sidebar.radio("Modo de Análise", ["Carteira/Ativo", "Simulação & Histórico Macro"])

# --- CONFIGURAÇÕES DE BENCHMARKS ---
# (Copiado do main.py para manter consistência, mas poderia vir de um arquivo config)
benchmarks_yf_config = {
    'S&P 500': 'SPY',
    'IVVB11': 'IVVB11.SA',
    'IMID': 'IMID.L',
    'Bitcoin': 'BTC-USD'
}
benchmarks_b3_config = {}
benchmarks_bcb_config = {'SELIC': 11, 'IPCA': 433}
benchmarks_td_config = {
    'TD IPCA 2035': {'titulo': 'Tesouro IPCA+', 'vencimento': '15/05/2035'},
    'TD IPCA 2045': {'titulo': 'Tesouro IPCA+', 'vencimento': '15/05/2045'}
}

# --- INTERFACE PRINCIPAL ---
st.title("📈 Dashboard de Investimentos")

if modo_analise == "Carteira/Ativo":
    st.subheader("Análise de Performance de Ativo ou Classe")
    
    col1, col2 = st.columns(2)
    with col1:
        tipo_filtro = st.selectbox("Filtrar por:", ["Ativo Específico", "Classe de Ativo"])
    with col2:
        valor_filtro = st.text_input(f"Digite o código do {tipo_filtro} (ex: KLBN11 ou AÇÃO):")

    anos_historico = st.slider("Anos de Histórico para Comparação (Benchmarks)", 1, 20, 10)

    if st.button("🔍 Gerar Análise"):
        if not token:
            st.warning("⚠️ Token não encontrado. Por favor, configure o Token na barra lateral para continuar.")
        elif not valor_filtro:
            st.error("Por favor, insira um código para análise.")
        else:
            with st.spinner("Buscando dados e gerando gráficos..."):
                # 1. Busca Histórico
                df_historico = None
                nome_analise = valor_filtro.upper()
                
                if tipo_filtro == "Ativo Específico":
                    df_historico = buscar_historico(token, logger, ativo=nome_analise)
                else:
                    df_historico = buscar_historico(token, logger, classe=nome_analise)

                if df_historico is not None and not df_historico.empty:
                    st.success(f"Dados encontrados para {nome_analise}!")
                    
                    # Abas para organizar a visualização
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance (TWR)", "📉 Risco x Retorno", "📅 Histórico Benchmarks", "📋 Dados Brutos"])

                    # --- TAB 1: Performance TWR ---
                    with tab1:
                        col_a, col_b = st.columns(2)
                        
                        # Gráfico de Evolução Patrimonial
                        fig_evol = gerar_grafico_evolucao(df_historico, nome_analise, logger)
                        col_a.pyplot(fig_evol)

                        # Gráfico Percentual Simples
                        fig_perc = gerar_grafico_percentual(df_historico, nome_analise, logger)
                        col_b.pyplot(fig_perc)

                        st.divider()
                        
                        # Cálculo TWR
                        df_twr, fig_twr = gerar_grafico_twr(df_historico, nome_analise, logger)
                        if df_twr is not None:
                            st.pyplot(fig_twr)
                            
                            # Comparativo com Benchmarks
                            start_date = df_twr['date'].min().strftime('%Y-%m-%d')
                            end_date = df_twr['date'].max().strftime('%Y-%m-%d')
                            
                            # Processa Benchmarks
                            benchmarks_data = processar_benchmarks(
                                start_date, end_date, 
                                benchmarks_yf_config, benchmarks_b3_config, 
                                benchmarks_bcb_config, benchmarks_td_config, 
                                {}, logger # Sem carteiras sintéticas aqui por enquanto
                            )
                            
                            fig_comp = gerar_grafico_comparativo_twr(df_twr, benchmarks_data, nome_analise, logger)
                            st.pyplot(fig_comp)

                    # --- TAB 2: Risco ---
                    with tab2:
                        if df_twr is not None:
                            dados_comparativo = benchmarks_data.copy()
                            dados_comparativo[f'Carteira - {nome_analise}'] = df_twr.set_index('date')['twr_acc'] + 1
                            selic_series = benchmarks_data.get('SELIC')
                            
                            fig_risk = gerar_analise_risco(dados_comparativo, selic_series, f'{nome_analise}_comparativo', logger)
                            if fig_risk:
                                st.pyplot(fig_risk)
                            else:
                                st.warning("Dados insuficientes para análise de risco.")

                    # --- TAB 3: Histórico Macro ---
                    with tab3:
                        st.markdown(f"### Visão de Longo Prazo ({anos_historico} anos)")
                        end_dt = pd.Timestamp.today()
                        start_dt_macro = (end_dt - pd.DateOffset(years=anos_historico) + pd.Timedelta(days=1))
                        
                        benchmarks_macro = processar_benchmarks(
                            start_dt_macro.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'),
                            benchmarks_yf_config, benchmarks_b3_config, 
                            benchmarks_bcb_config, benchmarks_td_config, 
                            {}, logger
                        )
                        
                        # Adiciona a carteira atual no contexto histórico (se houver dados)
                        if df_twr is not None:
                            benchmarks_macro[f'Carteira - {nome_analise}'] = df_twr.set_index('date')['twr_acc'] + 1

                        fig_hist = gerar_twr_historico(benchmarks_macro, anos_historico, nome_analise, end_dt, logger)
                        if fig_hist:
                            st.pyplot(fig_hist)

                    # --- TAB 4: Dados ---
                    with tab4:
                        st.dataframe(df_historico)
                        if df_twr is not None:
                            st.markdown("#### Dados TWR Calculados")
                            st.dataframe(df_twr)

                else:
                    st.warning("Nenhum dado encontrado para os filtros aplicados.")

elif modo_analise == "Simulação & Histórico Macro":
    st.subheader("🛠️ Simulador de Carteiras e Benchmarks")

    col1, col2 = st.columns(2)
    with col1:
        anos = st.number_input("Anos de Histórico", min_value=1, max_value=30, value=10)
        aporte = st.number_input("Aporte Mensal (R$)", min_value=0.0, value=1000.0, step=100.0)
    with col2:
        rebal = st.number_input("Meses para Rebalanceamento", min_value=1, value=12)
    
    st.markdown("### 🏗️ Criar Carteira Personalizada")
    st.info("Defina os pesos para os ativos disponíveis (a soma deve ser 1.0 ou 100%).")
    
    # Lista de ativos disponíveis para compor carteira
    ativos_disponiveis = list(benchmarks_yf_config.keys()) + \
                         list(benchmarks_td_config.keys()) + \
                         ['IPCA + 6%'] # Sintético hardcoded no market_data
    
    # Interface dinâmica para pesos
    if 'carteira_custom' not in st.session_state:
        st.session_state.carteira_custom = {}

    col_sel, col_peso, col_add = st.columns([3, 2, 1])
    with col_sel:
        ativo_sel = st.selectbox("Escolher Ativo", ativos_disponiveis)
    with col_peso:
        peso_sel = st.number_input("Peso (%)", min_value=0.0, max_value=100.0, value=0.0)
    with col_add:
        st.write("") # Spacer
        st.write("") 
        if st.button("➕ Adicionar"):
            if peso_sel > 0:
                st.session_state.carteira_custom[ativo_sel] = peso_sel / 100.0

    # Mostra carteira atual
    if st.session_state.carteira_custom:
        st.write("##### Carteira Atual:")
        df_cart = pd.DataFrame(list(st.session_state.carteira_custom.items()), columns=['Ativo', 'Peso'])
        df_cart['Peso'] = df_cart['Peso'].apply(lambda x: f"{x*100:.1f}%")
        st.table(df_cart)
        
        total_peso = sum(st.session_state.carteira_custom.values())
        if abs(total_peso - 1.0) > 0.01:
            st.warning(f"⚠️ A soma dos pesos é {total_peso*100:.1f}%. Ajuste para 100%.")
        
        if st.button("🗑️ Limpar Carteira"):
            st.session_state.carteira_custom = {}
            st.rerun()

    if st.button("🚀 Executar Simulação"):
        with st.spinner("Processando dados de mercado..."):
            end_dt = pd.Timestamp.today()
            start_dt = (end_dt - pd.DateOffset(years=anos) + pd.Timedelta(days=1))
            
            # Configura carteiras para simulação
            carteiras_simulacao = {}
            # Adiciona a customizada se válida
            if st.session_state.carteira_custom and abs(sum(st.session_state.carteira_custom.values()) - 1.0) < 0.01:
                carteiras_simulacao['Minha Carteira'] = st.session_state.carteira_custom
            
            # Adiciona algumas padrão para comparação
            carteiras_simulacao['IMID BRL 60/40'] = {'IMID BRL': 0.6, 'IPCA + 6%': 0.4}
            
            # Busca dados
            benchmarks_data = processar_benchmarks(
                start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'),
                benchmarks_yf_config, benchmarks_b3_config, 
                benchmarks_bcb_config, benchmarks_td_config, 
                carteiras_simulacao, logger
            )

            # 1. Gráfico Histórico (TWR)
            st.subheader("Performance Histórica (Base 100)")
            # Filtra o que mostrar
            benchmarks_exibir = list(carteiras_simulacao.keys()) + ['IMID BRL', 'TD IPCA 2035', 'S&P 500 BRL']
            dados_plot = {k: v for k, v in benchmarks_data.items() if k in benchmarks_exibir}
            
            fig_hist = gerar_twr_historico(dados_plot, anos, "Simulacao", end_dt, logger)
            if fig_hist:
                st.pyplot(fig_hist)

            # 2. Risco x Retorno
            st.subheader("Análise de Risco x Retorno")
            selic = benchmarks_data.get('SELIC')
            fig_risk = gerar_analise_risco(dados_plot, selic, "Simulacao", logger)
            if fig_risk:
                st.pyplot(fig_risk)

            # 3. Simulação de Aportes
            if aporte > 0:
                st.subheader(f"Simulação de Aportes (R${aporte}/mês)")
                figs_simulacao = simular_evolucao_patrimonio(
                    benchmarks_data, carteiras_simulacao, aporte, rebal, logger
                )
                
                # Exibe consolidado primeiro
                if 'Consolidado' in figs_simulacao:
                    st.pyplot(figs_simulacao['Consolidado'])
                
                # Exibe individuais
                cols = st.columns(2)
                for i, (nome, fig) in enumerate(figs_simulacao.items()):
                    if nome != 'Consolidado':
                        cols[i % 2].pyplot(fig)