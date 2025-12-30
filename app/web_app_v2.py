import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
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
        # if len(msg) < 80:
        #    st.toast(msg, icon="ℹ️")
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

def render_benchmark_section():
    st.subheader("Seleção de Benchmarks e Carteiras")
    
    # 1. Prepara opções padrão baseadas no config
    default_options_map = {}
    for item in BENCHMARKS_ATIVOS:
        if isinstance(item, str):
            default_options_map[item] = item
        elif isinstance(item, dict):
            default_options_map[item['nome']] = item
    
    # Layout para alinhar o "Selecionar Todos" próximo à coluna de checkbox
    col_txt, col_chk = st.columns([3, 1])
    with col_txt:
        st.caption("Selecione os benchmarks que deseja incluir na análise:")
    with col_chk:
        selecionar_todos = st.checkbox("Selecionar Todos", value=True, key="chk_select_all")
    
    # Cria DataFrame para seleção visual
    df_benchmarks = pd.DataFrame({
        "Benchmark": list(default_options_map.keys()),
        "Selecionar": selecionar_todos
    })

    edited_benchmarks = st.data_editor(
        df_benchmarks,
        column_config={
            "Benchmark": st.column_config.TextColumn("Benchmark", width="large", disabled=True),
            "Selecionar": st.column_config.CheckboxColumn("Incluir", width="small")
        },
        hide_index=True,
        use_container_width=True,
        height=200, # Altura fixa para exibir aprox. 5 linhas (com scroll)
        key="editor_benchmarks"
    )
    
    selected_names = edited_benchmarks[edited_benchmarks["Selecionar"]]["Benchmark"].tolist()
    
    # Reconstrói a lista de configuração baseada na seleção
    active_benchmarks_list = [default_options_map[name] for name in selected_names]
    
    st.markdown("#### Criar Carteira Personalizada")
    
    custom_name = st.text_input("Nome da Carteira", value="Minha Carteira", key="input_custom_name")
    
    # Consolida todos os ativos disponíveis nos catálogos
    all_assets = []
    all_assets.extend(CATALOGO_YF.keys())
    all_assets.extend(CATALOGO_B3.keys())
    all_assets.extend(CATALOGO_BCB.keys())
    all_assets.extend(CATALOGO_TD.keys())
    # Adiciona derivados comuns (Sintéticos suportados pelo market_data.py)
    all_assets.extend(['IMID BRL', 'Bitcoin BRL', 'IPCA + 6%'])
    
    all_assets = sorted(list(set(all_assets)))
    
    st.caption("Adicione ativos e defina os pesos (Total deve ser 100%)")
    
    # Dados iniciais para o editor
    default_data = pd.DataFrame([{"Ativo": "IBOV", "Peso": 100}])

    edited_df = st.data_editor(
        default_data,
        column_config={
            "Ativo": st.column_config.SelectboxColumn(
                "Ativo",
                help="Selecione o ativo",
                width="medium",
                options=all_assets,
                required=True,
            ),
            "Peso": st.column_config.NumberColumn(
                "Peso (%)",
                help="Peso do ativo (0 a 100)",
                min_value=0,
                max_value=100,
                step=1,
                format="%d%%",
                required=True,
            )
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="portfolio_editor"
    )

    custom_composition = {}
    total_weight = 0.0
    
    if edited_df is not None:
        for index, row in edited_df.iterrows():
            asset = row.get("Ativo")
            weight = row.get("Peso")
            
            if asset and pd.notnull(weight) and weight > 0:
                decimal_weight = weight / 100.0
                custom_composition[asset] = custom_composition.get(asset, 0.0) + decimal_weight
                total_weight += decimal_weight
    
    if custom_composition:
        if abs(total_weight - 1.0) < 0.01:
            st.success(f"Carteira '{custom_name}' válida! ({len(custom_composition)} ativos)")
            active_benchmarks_list.append({'nome': custom_name, 'composicao': custom_composition})
        else:
            st.warning(f"A soma dos pesos é {total_weight*100:.1f}%. Ajuste para 100%.")
            
    return active_benchmarks_list

# --- Helper para Gráficos Altair ---
def render_altair_line(df, title, y_format=".0%", y_title="Valor"):
    if df is None or df.empty: return
    
    # Preserva dataframe original para exibição na tabela
    df_display = df.copy()
    
    df = df.copy()
    # Garante que o índice é uma coluna para o Altair
    if df.index.name is None: df.index.name = 'Data'
    df = df.reset_index()
    
    # Sanitiza nomes de colunas para evitar erros no Altair (ex: pontos em tickers)
    safe_cols = [str(c).replace('.', '_') for c in df.columns]
    df.columns = safe_cols
    x_col = safe_cols[0] # Primeira coluna é a Data
    
    # Transformação para formato longo (Tidy Data)
    df_melt = df.melt(id_vars=[x_col], var_name='Ativo', value_name='Valor')
    
    # 1. Configuração do Tooltip Unificado (usando dados Wide)
    tooltip_list = [alt.Tooltip(x_col, type='temporal', title='Data', format='%d/%m/%Y')]
    for col in df.columns:
        if col == x_col: continue
        tooltip_list.append(alt.Tooltip(col, type='quantitative', format=y_format))

    # 2. Seletor de Interação (Nearest X)
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=[x_col], empty=False)

    # 3. Camadas do Gráfico
    lines = alt.Chart(df_melt).mark_line(point=False).encode(
        x=alt.X(f'{x_col}:T', title='Data'),
        y=alt.Y('Valor:Q', title=y_title, axis=alt.Axis(format=y_format)),
        color='Ativo:N'
    )

    points = lines.mark_circle().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Camada invisível (Regra) que captura o mouse e mostra o tooltip com TODOS os dados
    rule = alt.Chart(df).mark_rule(color='gray').encode(
        x=f'{x_col}:T',
        opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
        tooltip=tooltip_list
    ).add_params(nearest)
    
    chart = alt.layer(lines, points, rule).properties(title=title, height=400).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    with st.expander(f"🔍 Ver dados: {title}"):
        st.dataframe(df_display, use_container_width=True)

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

            # Adiciona a seleção de benchmarks também no modo Carteira
            active_benchmarks_list = render_benchmark_section()

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
            
            st.markdown("---")
            
            st.subheader("Período de Análise")
            use_full_period = st.checkbox("Usar todo o histórico disponível", value=True)
            
            data_inicio = None
            data_fim = None
            
            if not use_full_period:
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    default_start = pd.Timestamp.today() - pd.DateOffset(years=1)
                    data_inicio = st.date_input("Data Início", value=default_start)
                with col_d2:
                    data_fim = st.date_input("Data Fim", value="today")
            
            st.markdown("---")
            
            st.markdown("")
            simular_aportes = st.checkbox("Simular Aportes (Shadow Portfolio)", value=True)
            
        else: # Modo Mercado
            active_benchmarks_list = render_benchmark_section()

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
                user_series = report.fetch_user_portfolio(
                    token, 
                    ativo=ativo_str, 
                    classe=classe_str,
                    start_date=data_inicio if not use_full_period else None,
                    end_date=data_fim if not use_full_period else None
                )
                
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
                        #with cols[j]:
                        #    st.metric(
                        #    label=col_name, 
                        #    value=f"{val:.1%}", 
                        #    delta=f"{val:.1%}", # Mostra a variação
                        #    delta_color="normal" # Verde para positivo, Vermelho para negativo
                        #)

            # Abas para organização
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Rentabilidade & Risco", "📉 Drawdown & Volatilidade", "💰 Simulação & TIR", "📋 Dados Brutos"])

            with tab1:
                st.subheader("Evolução TWR (Time-Weighted Return)")
                # O df_twr vem em Base 100 (ex: 105.0). Usamos formato float (.1f)
                _, df_twr = report.plot_twr_evolution(title_suffix=nome_analise, return_fig=True)
                render_altair_line(df_twr, "Evolução TWR (Base 100)", y_format=".1f", y_title="Base 100")

                st.subheader("Risco x Retorno")
                _, df_risk = report.plot_risk_return_scatter(title_suffix=nome_analise, return_fig=True)
                
                if df_risk is not None:
                    df_risk = df_risk.reset_index().rename(columns={'index': 'Ativo'})
                    chart_risk = alt.Chart(df_risk).mark_circle(size=100).encode(
                        x=alt.X('Volatilidade', axis=alt.Axis(format='%')),
                        y=alt.Y('Retorno (CAGR)', axis=alt.Axis(format='%')),
                        color='Ativo',
                        tooltip=['Ativo', alt.Tooltip('Volatilidade', format='.2%'), alt.Tooltip('Retorno (CAGR)', format='.2%'), alt.Tooltip('Sharpe', format='.2f')]
                    ).properties(height=400, title="Risco (Vol) x Retorno (CAGR)").interactive()
                    
                    st.altair_chart(chart_risk, use_container_width=True)
                    with st.expander("🔍 Ver dados: Risco x Retorno"):
                        st.dataframe(df_risk, use_container_width=True)

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Drawdown (Queda Máxima)")
                    _, df_dd = report.plot_drawdown(title_suffix=nome_analise, return_fig=True)
                    # Drawdown é negativo (ex: -0.05), formato % funciona bem
                    render_altair_line(df_dd, "Drawdown", y_format=".1%")
                
                with col2:
                    st.subheader("Volatilidade Móvel (Risco)")
                    _, df_vol = report.plot_rolling_volatility(title_suffix=nome_analise, return_fig=True)
                    render_altair_line(df_vol, "Volatilidade Anualizada", y_format=".1%")

                st.subheader("Sharpe Ratio Móvel (Eficiência)")
                _, df_sharpe = report.plot_rolling_sharpe(title_suffix=nome_analise, return_fig=True)
                render_altair_line(df_sharpe, "Sharpe Ratio", y_format=".2f", y_title="Sharpe")

            with tab3:
                if modo == "Carteira DLP Invest":
                    col_tir, col_sim = st.columns(2)
                    
                    with col_tir:
                        st.subheader("Evolução da TIR (Rentabilidade Real)")
                        _, series_tir = report.plot_irr_evolution(title_suffix=nome_analise, return_fig=True)
                        if series_tir is not None and not series_tir.empty:
                            # TIR vem multiplicada por 100 no main_v2 (ex: 10.5). 
                            # Altair espera decimal para %, ou usamos 'f' com sufixo.
                            # Vamos converter de volta para decimal para usar formatação % padrão
                            df_tir = (series_tir / 100).to_frame(name="TIR")
                            render_altair_line(df_tir, "TIR Histórica", y_format=".2%")
                        else:
                            st.info("Dados insuficientes para cálculo da TIR.")

                    with col_sim:
                        if simular_aportes:
                            st.subheader("Simulação de Aportes (Shadow Portfolio)")
                            _, df_sim = report.simulate_shadow_portfolios(title_suffix=nome_analise, return_fig=True)
                            # Valores monetários
                            render_altair_line(df_sim, "Patrimônio Simulado (R$)", y_format=",.0f", y_title="R$")
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
