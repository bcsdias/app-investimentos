import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from dotenv import load_dotenv

# Configuração da Página (Deve ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Investimentos V3",
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

def get_asset_categories():
    """Organiza os ativos dos catálogos em categorias para exibição."""
    categories = {
        "Bolsa Brasil": [],
        "Bolsa Internacional": [],
        "Índices / Renda Fixa": [],
        "Tesouro Direto": []
    }
    
    # Helper para identificar internacionais no YF
    intl_tickers = ['SPY', 'SPY.BA', 'IVVB11.SA', 'IMID.L', 'BTC-USD']
    
    # YF
    for name, ticker in CATALOGO_YF.items():
        if ticker in intl_tickers or name in ['S&P 500', 'S&P 500 BRL', 'IVVB11', 'IMID', 'Bitcoin']:
            categories["Bolsa Internacional"].append(name)
        else:
            categories["Bolsa Brasil"].append(name)
            
    # B3
    categories["Bolsa Brasil"].extend(list(CATALOGO_B3.keys()))
    
    # BCB
    categories["Índices / Renda Fixa"].extend(list(CATALOGO_BCB.keys()))
    
    # TD
    categories["Tesouro Direto"].extend(list(CATALOGO_TD.keys()))
    
    # Adiciona sintéticos comuns
    categories["Bolsa Internacional"].extend(["IMID BRL", "Bitcoin BRL"])
    categories["Índices / Renda Fixa"].append("IPCA + 6%")
    
    return categories

def render_sidebar_asset_selection():
    """Renderiza a seleção de ativos na sidebar."""
    st.sidebar.header("1. Seleção de Ativos Base")
    st.sidebar.caption("Escolha os ativos que estarão disponíveis para compor as carteiras ou comparar individualmente.")
    
    categories = get_asset_categories()
    selected_assets = []
    
    for cat, assets in categories.items():
        assets = sorted(list(set(assets)))
        with st.sidebar.expander(cat, expanded=False):
            sel = st.multiselect(f"Ativos ({cat})", assets, key=f"sel_{cat}")
            selected_assets.extend(sel)
            
    return list(set(selected_assets))

def render_portfolio_builder(available_assets):
    """Renderiza a área de construção de carteiras no painel principal."""
    st.subheader("2. Construção de Carteiras e Benchmarks")
    
    if not available_assets:
        st.info("👈 Selecione ativos na barra lateral para começar.")
        return []
        
    # --- Benchmarks Individuais ---
    with st.expander("Benchmarks Individuais (Plotar Linhas)", expanded=True):
        df_bench = pd.DataFrame({"Ativo": available_assets, "Incluir": True})
        edited_bench = st.data_editor(
            df_bench, 
            column_config={
                "Ativo": st.column_config.TextColumn("Ativo", disabled=True),
                "Incluir": st.column_config.CheckboxColumn("Plotar", width="small")
            },
            hide_index=True,
            use_container_width=True,
            key="editor_bench_individual"
        )
    active_benchmarks = edited_bench[edited_bench["Incluir"]]["Ativo"].tolist()
    
    # --- Carteiras Personalizadas ---
    st.markdown("##### Carteiras Personalizadas")
    
    if "custom_portfolios" not in st.session_state:
        st.session_state.custom_portfolios = [{"name": "Minha Carteira", "weights": {}}]

    if st.button("➕ Nova Carteira"):
        new_id = len(st.session_state.custom_portfolios) + 1
        st.session_state.custom_portfolios.append({"name": f"Carteira {new_id}", "weights": {}})
    
    final_config_list = list(active_benchmarks)
    indices_to_remove = []
    
    if st.session_state.custom_portfolios:
        tabs = st.tabs([p["name"] if p["name"] else f"Carteira {i+1}" for i, p in enumerate(st.session_state.custom_portfolios)])
        
        for i, tab in enumerate(tabs):
            with tab:
                p = st.session_state.custom_portfolios[i]
                col1, col2 = st.columns([4, 1])
                with col1:
                    new_name = st.text_input(f"Nome", value=p["name"], key=f"name_{i}")
                    st.session_state.custom_portfolios[i]["name"] = new_name
                with col2:
                    if st.button("🗑️", key=f"del_{i}"): indices_to_remove.append(i)
                
                # Prepara dados para o editor
                current_weights = p.get("weights", {})
                data = [{"Ativo": a, "Peso (%)": current_weights.get(a, 0.0) * 100} for a in available_assets]
                
                edited_weights = st.data_editor(
                    pd.DataFrame(data),
                    column_config={
                        "Ativo": st.column_config.TextColumn("Ativo", disabled=True),
                        "Peso (%)": st.column_config.NumberColumn("Peso (%)", min_value=0, max_value=100, format="%.1f")
                    },
                    hide_index=True,
                    use_container_width=True,
                    key=f"editor_port_{i}"
                )
                
                # Processa pesos
                composition = {}
                total_w = 0.0
                for _, row in edited_weights.iterrows():
                    w = row["Peso (%)"]
                    if w > 0:
                        composition[row["Ativo"]] = w / 100.0
                        total_w += w
                
                st.session_state.custom_portfolios[i]["weights"] = composition
                
                if composition:
                    if abs(total_w - 100.0) > 0.1:
                        st.warning(f"Total: {total_w:.1f}%. Ajuste para 100%.")
                    else:
                        st.success(f"Carteira válida.")
                        final_config_list.append({"nome": new_name, "composicao": composition})
                else:
                    st.caption("Defina os pesos acima.")

    if indices_to_remove:
        for index in sorted(indices_to_remove, reverse=True):
            del st.session_state.custom_portfolios[index]
        st.rerun()
        
    return final_config_list

# --- Helper para Gráficos Altair ---
def render_altair_line(df, title, y_format=".0%", y_title="Valor"):
    if df is None or df.empty: return
    
    # --- Ordenação por Rentabilidade Final (Maior para Menor) ---
    # Garante que a tabela e o tooltip mostrem os ativos mais rentáveis primeiro
    if not df.empty:
        # Pega a última linha válida para ordenar
        last_vals = df.ffill().iloc[-1]
        sorted_cols = last_vals.sort_values(ascending=False).index
        df = df[sorted_cols]

    # Preserva dataframe original para exibição na tabela
    df_display = df.copy()
    
    df = df.copy()
    
    # --- OTIMIZAÇÃO DE PERFORMANCE: Downsampling ---
    # Se houver muitos pontos (ex: > 800), reduz a granularidade visual para não travar o navegador.
    # Isso mantém a tendência visual mas reduz drasticamente o peso do JSON/HTML gerado.
    MAX_POINTS = 800
    if len(df) > MAX_POINTS:
        step = len(df) // MAX_POINTS
        # Fatia o dataframe (ex: pega a cada 2, 3, 4... dias)
        df_resampled = df.iloc[::step]
        # Garante que o último ponto (data mais recente) seja incluído para não parecer desatualizado
        if df.index[-1] != df_resampled.index[-1]:
            df_resampled = pd.concat([df_resampled, df.iloc[[-1]]])
        df = df_resampled
    # -----------------------------------------------

    # Garante que o índice é uma coluna para o Altair
    if df.index.name is None: df.index.name = 'Data'
    df = df.reset_index()
    
    # Sanitiza nomes de colunas para evitar erros no Altair (ex: pontos em tickers)
    safe_cols = [str(c).replace('.', '_') for c in df.columns]
    df.columns = safe_cols
    x_col = safe_cols[0] # Primeira coluna é a Data
    
    # Lista ordenada para forçar a legenda a seguir a ordem de rentabilidade (Maior -> Menor)
    legend_sort = [c for c in safe_cols if c != x_col]
    
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
        color=alt.Color('Ativo:N', sort=legend_sort, legend=alt.Legend(title="Ativo"))
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
    st.title("📊 Dashboard de Investimentos V3")
    st.markdown("---")

    # Inicializa variáveis
    ativo = None
    classe = None
    token = None
    active_benchmarks_list = None # Lista final a ser passada para o report
    simular_aportes = False

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configurações")
        
        # 1. Seleção de Ativos (Nova Função)
        selected_assets = render_sidebar_asset_selection()
        
        st.divider()
        st.subheader("2. Período de Análise")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            default_start = pd.Timestamp.today() - pd.DateOffset(years=1)
            data_inicio = st.date_input("Início", value=default_start, min_value=pd.Timestamp("1900-01-01"))
        with col_d2:
            data_fim = st.date_input("Fim", value="today", min_value=pd.Timestamp("1900-01-01"))
            
        st.divider()
        
        # Opção de Comparação com Carteira
        comparar_carteira = st.checkbox("Comparar com Carteira DLP Invest", value=False)
        
        if comparar_carteira:
            env_token = os.getenv('DLP_TOKEN', '')
            token = st.text_input("Token API (DLP)", value=env_token, type="password")

    # --- Main Area: Personalização ---
    with st.container(border=True):
        # 1. Construção de Carteiras (Substitui render_benchmark_section)
        active_benchmarks_list = render_portfolio_builder(selected_assets)

        # 2. Seleção de Ativos da Carteira (Condicional)
        if comparar_carteira:
            st.markdown("---")
            st.subheader("Seleção de Ativos da Carteira")
            
            # Tenta buscar dados da carteira se o token estiver presente
            wallet_data = None
            if token:
                wallet_data = get_wallet_data(token)

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
            
        # else: # Modo Mercado (Benchmarks já renderizados acima)

        st.markdown("")
        btn_processar = st.button("🚀 Gerar Relatório", type="primary")

    # --- Processamento ---
    if btn_processar:
        if comparar_carteira and not token:
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
            # Formata datas para string YYYY-MM-DD
            start_date_str = data_inicio.strftime('%Y-%m-%d')
            end_date_str = data_fim.strftime('%Y-%m-%d')
            
            if comparar_carteira:
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
                    start_date=start_date_str,
                    end_date=end_date_str
                )
                
                if user_series is None:
                    st.error("Não foi possível obter dados da carteira. Verifique o Token ou o Ativo/Classe.")
                    return
            else:
                nome_analise = f"Mercado_{start_date_str}_{end_date_str}"
                status_text.info(f"Buscando dados de mercado ({start_date_str} a {end_date_str})...")

            # 2. Constrói Dataset
            status_text.info("Consolidando benchmarks e calculando indicadores...")
            report.build_dataset(
                user_series=user_series, 
                active_benchmarks=active_benchmarks_list,
                start_date=start_date_str,
                end_date=end_date_str
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
                    df_risk = df_risk.sort_values('Retorno (CAGR)', ascending=False)
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
                if comparar_carteira:
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
