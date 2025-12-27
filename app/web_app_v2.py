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
from app.benchmarks_config import BENCHMARKS_ATIVOS, CATALOGO_YF, CATALOGO_B3, CATALOGO_BCB, CATALOGO_TD

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

# --- Função de Cache para Dados da Carteira ---
@st.cache_data(ttl=600, show_spinner=False)
def get_wallet_data(token):
    from utils.market_data import buscar_resumo_carteira
    # Logger simples para o cache (evita passar o StreamlitLogger que não é serializável)
    class SimpleLogger:
        def error(self, msg): print(f"ERROR: {msg}")
        def info(self, msg): print(f"INFO: {msg}")
        def debug(self, msg): pass
        def warning(self, msg): print(f"WARN: {msg}")
    return buscar_resumo_carteira(token, SimpleLogger())

# --- Função Principal do App ---
def main():
    st.title("📊 Dashboard de Investimentos V2")
    st.markdown("---")

    # Inicializa variáveis
    ativo = None
    classe = None
    historico = 5
    token = None
    active_benchmarks_list = None # Lista final a ser passada para o report
    simular_aportes = False

    # --- Sidebar: Configurações Gerais ---
    with st.sidebar:
        st.header("Configurações")
        modo = st.radio("Modo de Operação", ["Mercado (Benchmarks)", "Carteira DLP Invest"])
        st.divider()

        if modo == "Carteira DLP Invest":
            env_token = os.getenv('DLP_TOKEN', '')
            token = st.text_input("Token API (DLP)", value=env_token, type="password")
            st.button("Aplicar Token") # Botão para confirmar entrada em dispositivos móveis
        else:
            historico = st.slider("Anos de Histórico", min_value=1, max_value=20, value=5)

    # --- Main Area: Personalização ---
    with st.container(border=True):
        if modo == "Carteira DLP Invest":
            # Tenta buscar dados da carteira se o token estiver presente
            wallet_data = None
            if token:
                wallet_data = get_wallet_data(token)

            st.subheader("Seleção de Ativos")
            col1, col2 = st.columns(2)
            
            with col1:
                tipo_filtro = st.selectbox("Filtrar por", ["Classe de Ativo", "Ativo Específico"])
            
            with col2:
                if wallet_data:
                    # Extrai listas do JSON da API
                    assets_list = sorted(wallet_data.get('summary', {}).get('operations_values', {}).get('assets', []))
                    classes_list = sorted(wallet_data.get('summary', {}).get('operations_values', {}).get('classes', []))
                    
                    if tipo_filtro == "Classe de Ativo":
                        # Seleciona AÇÃO por padrão se existir
                        default_classe = ["AÇÃO"] if "AÇÃO" in classes_list else []
                        classe = st.multiselect("Selecione a(s) Classe(s)", classes_list, default=default_classe)
                    else:
                        # Seleciona PETR4 por padrão se existir
                        default_ativo = ["PETR4"] if "PETR4" in assets_list else []
                        ativo = st.multiselect("Selecione o(s) Ativo(s)", assets_list, default=default_ativo)
                else:
                    # Fallback para texto livre se não carregar dados
                    if tipo_filtro == "Classe de Ativo":
                        classe_input = st.text_input("Nome da Classe", value="AÇÃO").upper()
                        classe = [c.strip() for c in classe_input.split(',') if c.strip()]
                    else:
                        ativo_input = st.text_input("Código do Ativo", value="PETR4").upper()
                        ativo = [a.strip() for a in ativo_input.split(',') if a.strip()]
            
            # Exibe detalhes do ativo selecionado (Setor, Preço, etc)
            if tipo_filtro == "Ativo Específico" and ativo and wallet_data:
                wallet_items = wallet_data.get('wallet', [])
                for a in ativo[:5]:
                    info = next((item for item in wallet_items if item.get('ativo') == a), None)
                    if info:
                        st.info(f"ℹ️ **{a}** | Setor: {info.get('setor', '-')} | Subsetor: {info.get('subsetor', '-')} | Preço Atual: R$ {info.get('price', 0)}")
                    else:
                        st.caption(f"O ativo **{a}** já foi operado, mas não consta na posição atual da carteira.")
                if len(ativo) > 5:
                    st.caption(f"... e mais {len(ativo)-5} ativos selecionados.")
            
            st.markdown("")
            simular_aportes = st.checkbox("Simular Aportes (Shadow Portfolio)", value=True)
            
        else: # Modo Mercado
            st.subheader("Seleção de Benchmarks e Carteiras")
            
            # 1. Prepara opções padrão baseadas no config
            default_options_map = {}
            for item in BENCHMARKS_ATIVOS:
                if isinstance(item, str):
                    default_options_map[item] = item
                elif isinstance(item, dict):
                    default_options_map[item['nome']] = item
            
            # Multiselect para escolher quais exibir
            selected_names = st.multiselect(
                "Benchmarks Disponíveis", 
                options=list(default_options_map.keys()),
                default=list(default_options_map.keys())
            )
            
            # Reconstrói a lista de configuração baseada na seleção
            active_benchmarks_list = [default_options_map[name] for name in selected_names]
            
            st.markdown("#### Criar Carteira Personalizada")
            
            custom_name = st.text_input("Nome da Carteira", value="Minha Carteira")
            
            # Consolida todos os ativos disponíveis nos catálogos
            all_assets = []
            all_assets.extend(CATALOGO_YF.keys())
            all_assets.extend(CATALOGO_B3.keys())
            all_assets.extend(CATALOGO_BCB.keys())
            all_assets.extend(CATALOGO_TD.keys())
            # Adiciona derivados comuns
            all_assets.extend([f"{k} BRL" for k in CATALOGO_YF.keys()])
            all_assets.append("IPCA + 6%")
            
            all_assets = sorted(list(set(all_assets)))
            
            selected_assets = st.multiselect("Selecione os Ativos para Composição", options=all_assets)
            
            custom_composition = {}
            if selected_assets:
                st.caption("Defina os pesos (soma deve ser 1.0):")
                cols = st.columns(min(len(selected_assets), 4))
                
                total_weight = 0.0
                for i, asset in enumerate(selected_assets):
                    col_idx = i % 4
                    with cols[col_idx]:
                        # Peso padrão igualitário
                        default_w = 1.0 / len(selected_assets)
                        w = st.number_input(f"{asset}", min_value=0.0, max_value=1.0, value=default_w, step=0.05, format="%.2f", key=f"w_{asset}")
                        custom_composition[asset] = w
                        total_weight += w
                
                if abs(total_weight - 1.0) < 0.01:
                    st.success("Carteira válida! Será adicionada à análise.")
                    active_benchmarks_list.append({'nome': custom_name, 'composicao': custom_composition})
                else:
                    st.warning(f"A soma dos pesos é {total_weight:.2f}. Ajuste para 1.00.")

        st.markdown("")
        btn_processar = st.button("🚀 Gerar Relatório", type="primary")

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
                ativo_str = ",".join(ativo) if ativo else None
                classe_str = ",".join(classe) if classe else None
                
                nome_analise = ativo_str if ativo_str else classe_str
                if len(nome_analise) > 50:
                    nome_analise = nome_analise[:47] + "..."
                
                status_text.info(f"Buscando dados da carteira: {nome_analise}...")
                user_series = report.fetch_user_portfolio(token, ativo=ativo_str, classe=classe_str)
                
                if user_series is None:
                    st.error("Não foi possível obter dados da carteira. Verifique o Token ou o Ativo/Classe.")
                    return
            else:
                nome_analise = f"Mercado_{historico}anos"
                status_text.info(f"Buscando dados de mercado ({historico} anos)...")

            # 2. Constrói Dataset
            status_text.info("Consolidando benchmarks e calculando indicadores...")
            report.build_dataset(
                user_series=user_series, 
                years_history=historico if modo != "Carteira DLP Invest" else None,
                active_benchmarks=active_benchmarks_list
            )

            if report.df_combined.empty:
                st.error("Nenhum dado disponível para gerar gráficos.")
                return

            status_text.success("Dados processados com sucesso!")
            
            # --- Exibição dos Resultados ---
            
            # Métricas de Topo (KPIs)
            if not report.df_combined.empty:
                df = report.df_combined
                retorno_total = (df.iloc[-1] / df.iloc[0]) - 1
                
                # Layout em Grid: Máximo de 5 métricas por linha para não espremer
                cols_per_row = 5
                col_names = df.columns
                
                for i in range(0, len(col_names), cols_per_row):
                    batch = col_names[i:i+cols_per_row]
                    cols = st.columns(cols_per_row)
                    for j, col_name in enumerate(batch):
                        val = retorno_total[col_name]
                        with cols[j]:
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
