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
from app.benchmarks_config import BENCHMARKS_ATIVOS, CATALOGO_YF, CATALOGO_B3, CATALOGO_BCB, CATALOGO_TD, CATALOGO_CRYPTO

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

@st.cache_data
def load_assets_from_csv():
    """Carrega ativos do CSV local e separa por categoria (Brasil vs Internacional)."""
    csv_path = os.path.join(BASE_DIR, 'data', 'ativos.csv')
    if not os.path.exists(csv_path):
        return {}, {}
    
    try:
        df = pd.read_csv(csv_path, sep=';')
        
        br_assets = {}
        intl_assets = {}
        
        # Definição de classes
        classes_br = ['ACAO', 'ETF', 'ETF RF', 'FI_INFRA', 'FI_SET', 'FIDC', 'FII', 'FIP', 'FIP_IE', 'BDR', 'BDR ETF']
        classes_intl = ['ETF_GB', 'ETF_US', 'REIT', 'STOCK']
        
        for _, row in df.iterrows():
            classe = str(row['classe']).strip()
            ticker = str(row['sigla']).strip()
            market_cod = str(row['market_cod']).strip()
            
            # Lógica de Sufixo YF baseada no market_cod
            yf_ticker = ticker
            if ':' in market_cod:
                exchange = market_cod.split(':')[0]
                if exchange == 'BVMF':
                    yf_ticker = f"{ticker}.SA"
                elif exchange == 'LON':
                    yf_ticker = f"{ticker}.L"
            
            display_name = f"{ticker} - {classe}"
            
            if classe in classes_br:
                br_assets[display_name] = yf_ticker
            elif classe in classes_intl:
                intl_assets[display_name] = yf_ticker
                
        return br_assets, intl_assets
    except Exception as e:
        print(f"[ERROR] Erro ao carregar ativos.csv: {e}")
        return {}, {}

@st.cache_data
def load_tesouro_from_csv():
    """Carrega títulos do Tesouro Direto do CSV local (data/raw/PrecoTaxaTesouroDireto.csv)."""
    csv_path = os.path.join(BASE_DIR, 'data', 'raw', 'PrecoTaxaTesouroDireto.csv')
    if not os.path.exists(csv_path):
        return {}
    
    try:
        # Lê apenas colunas necessárias e remove duplicatas para performance
        # encoding='latin1' é comum para arquivos gerados no Excel/Tesouro
        df = pd.read_csv(csv_path, sep=';', decimal=',', usecols=['Tipo Titulo', 'Data Vencimento'], encoding='latin1')
        df = df.drop_duplicates()
        
        # Identifica se há múltiplos vencimentos do mesmo título no mesmo ano
        # Ex: Prefixado 2007 (Jan) e Prefixado 2007 (Jul)
        df['Ano'] = df['Data Vencimento'].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else '')
        counts = df.groupby(['Tipo Titulo', 'Ano']).size().to_dict()
        
        td_assets = {}
        for _, row in df.iterrows():
            tipo = row['Tipo Titulo']
            venc = row['Data Vencimento']
            ano = row['Ano']
            
            if not ano: continue
            
            # Se houver mais de um título do mesmo tipo vencendo no mesmo ano, usa a data completa no nome
            if counts.get((tipo, ano), 0) > 1:
                key = f"{tipo} {venc}"
            else:
                key = f"{tipo} {ano}"
                
            td_assets[key] = {'titulo': tipo, 'vencimento': venc}
            
        return td_assets
    except Exception as e:
        print(f"[ERROR] Erro ao carregar Tesouro Direto do CSV: {e}")
        return {}

def get_asset_categories():
    """Organiza os ativos dos catálogos em categorias para exibição."""
    # Carrega ativos do CSV e atualiza o catálogo YF
    br_assets, intl_assets = load_assets_from_csv()
    CATALOGO_YF.update(br_assets)
    CATALOGO_YF.update(intl_assets)
    
    # Carrega Tesouro Direto do CSV local
    td_csv = load_tesouro_from_csv()
    CATALOGO_TD.update(td_csv)

    # Adiciona Criptos ao catálogo YF para download
    CATALOGO_YF.update(CATALOGO_CRYPTO)
    
    # Nova estrutura para categories, com "Bolsa Brasil" como um dicionário
    categories = {
        "Bolsa Brasil": {},
        "Bolsa Internacional": {},
        "Criptomoedas": [],
        "Indices": [],
        "Tesouro Direto": []
    }
    
    # Mapeamento e Filtro de Classes Bolsa Brasil
    map_classes_br = {
        'ACAO': 'AÇÕES',
        'FII': 'FII',
        'ETF': 'ETF',
        'ETF RF': 'ETF RENDA FIXA',
        'FI_INFRA': 'FI INFRA',
        'FIP': 'FIP/FIP IE',
        'FIP_IE': 'FIP/FIP IE',
        'BDR': 'BDR',
        'BDR ETF': 'BDR'
    }
    ignore_classes_br = ['FIDC', 'FI_SET']

    # Processa ativos brasileiros do CSV, agrupando por classe
    for display_name in br_assets.keys():
        try:
            # Extrai a classe do display_name (ex: "ITSA4 - ACAO" -> "ACAO")
            parts = display_name.split(' - ')
            if len(parts) < 2: continue
            classe_raw = parts[1]
        except IndexError:
            continue # Pula nomes mal formatados
        
        if classe_raw in ignore_classes_br:
            continue
            
        classe_final = map_classes_br.get(classe_raw, classe_raw)
        
        if classe_final not in categories["Bolsa Brasil"]:
            categories["Bolsa Brasil"][classe_final] = []
        categories["Bolsa Brasil"][classe_final].append(display_name)

    # Mapeamento de Classes Bolsa Internacional
    map_classes_intl = {
        'STOCK': 'STOCK',
        'REIT': 'REIT',
        'ETF': 'ETF',
        'ETF_US': 'ETF',
        'ETF_GB': 'ETF',
    }

    # Processa ativos internacionais do CSV
    for display_name in intl_assets.keys():
        try:
            parts = display_name.split(' - ')
            if len(parts) < 2: continue
            classe_raw = parts[1]
        except IndexError:
            continue
        
        classe_final = map_classes_intl.get(classe_raw, classe_raw)
        if classe_final not in categories["Bolsa Internacional"]:
            categories["Bolsa Internacional"][classe_final] = []
        categories["Bolsa Internacional"][classe_final].append(display_name)
    
    # Helper para identificar internacionais no YF (hardcoded)
    intl_tickers_hardcoded = ['SPY', 'SPY.BA', 'IVVB11.SA', 'IMID.L']
    
    # Processa ativos hardcoded do CATALOGO_YF que não vieram do CSV
    for name, ticker in CATALOGO_YF.items():
        # Pula os que já foram processados do CSV
        if name in br_assets or name in intl_assets or name in CATALOGO_CRYPTO:
            continue
            
        if name == 'S&P 500 BRL': continue # Oculta versão BRL explícita da sidebar
        
        if name in ['S&P 500', 'Ibovespa (YF)']:
            categories["Indices"].append(name)
        elif ticker in intl_tickers_hardcoded or name in ['IVVB11', 'IMID']:
            # Categoriza hardcoded como ETF
            c_name = "ETF"
            
            if c_name not in categories["Bolsa Internacional"]:
                categories["Bolsa Internacional"][c_name] = []
            categories["Bolsa Internacional"][c_name].append(name)
        else:
            # Ativos BR hardcoded vão para a categoria 'Ações' por padrão
            if "AÇÕES" not in categories["Bolsa Brasil"]:
                categories["Bolsa Brasil"]["AÇÕES"] = []
            categories["Bolsa Brasil"]["AÇÕES"].append(name)
            
    # Popula a categoria Criptomoedas com a lista do config
    categories["Criptomoedas"].extend(list(CATALOGO_CRYPTO.keys()))

    # B3
    categories["Indices"].extend(list(CATALOGO_B3.keys()))
    
    # BCB
    categories["Indices"].extend(list(CATALOGO_BCB.keys()))
    
    # TD
    categories["Tesouro Direto"].extend(list(CATALOGO_TD.keys()))
    
    return categories

def render_sidebar_asset_selection():
    """Renderiza a seleção de ativos na sidebar."""
    st.sidebar.header("1. Seleção de Ativos Base")
    st.sidebar.caption("Escolha os ativos que estarão disponíveis para compor as carteiras ou comparar individualmente.")
    
    categories = get_asset_categories()
    selected_assets = []
    
    # Ordem desejada das categorias principais
    main_cat_order = ["Bolsa Brasil", "Bolsa Internacional", "Criptomoedas", "Indices", "Tesouro Direto"]
    
    for cat_name in main_cat_order:
        if cat_name not in categories:
            continue
        
        cat_content = categories[cat_name]
        
        with st.sidebar.expander(cat_name, expanded=False):
            # Se o conteúdo for um dicionário, cria sub-seleções para cada classe
            if isinstance(cat_content, dict):
                # Ordenação específica para Bolsa Brasil
                if cat_name == "Bolsa Brasil":
                    order_br = ['AÇÕES', 'FII', 'BDR', 'ETF', 'ETF RENDA FIXA', 'FI INFRA', 'FIP/FIP IE']
                    sub_cats = sorted(cat_content.keys(), key=lambda x: (order_br.index(x) if x in order_br else 999, x))
                elif cat_name == "Bolsa Internacional":
                    order_intl = ['STOCK', 'REIT', 'ETF']
                    sub_cats = sorted(cat_content.keys(), key=lambda x: (order_intl.index(x) if x in order_intl else 999, x))
                else:
                    sub_cats = sorted(cat_content.keys())

                for sub_cat_name in sub_cats:
                    assets = cat_content[sub_cat_name]
                    sel = st.multiselect(
                        sub_cat_name,
                        sorted(list(set(assets))), 
                        key=f"sel_{cat_name}_{sub_cat_name}"
                    )
                    selected_assets.extend(sel)
            else: # Para as outras categorias, mantém o multiselect único
                assets = sorted(list(set(cat_content)))
                sel = st.multiselect("Selecione", assets, key=f"sel_{cat_name}", label_visibility="collapsed")
                selected_assets.extend(sel)
            
    # --- Índices Personalizados ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Índices Personalizados")
    
    if "custom_indices" not in st.session_state:
        st.session_state.custom_indices = []
        
    with st.sidebar.expander("➕ Criar Novo", expanded=False):
        c_type = st.selectbox("Tipo", ["IPCA +", "CDI +", "% do CDI"], key="idx_type")
        c_val = st.number_input("Taxa (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key="idx_val")
        
        if st.button("Adicionar"):
            if c_val > 0:
                if c_type == "IPCA +":
                    new_idx = f"IPCA + {c_val:.1f}%"
                elif c_type == "CDI +":
                    new_idx = f"CDI + {c_val:.1f}%"
                else: # % do CDI
                    new_idx = f"{c_val:.0f}% do CDI"
                
                if new_idx not in st.session_state.custom_indices:
                    st.session_state.custom_indices.append(new_idx)
                    st.rerun()
    
    if st.session_state.custom_indices:
        st.sidebar.caption("Meus Índices:")
        for idx in st.session_state.custom_indices:
            c1, c2 = st.sidebar.columns([0.85, 0.15])
            c1.text(idx)
            if c2.button("x", key=f"rm_{idx}"):
                st.session_state.custom_indices.remove(idx)
                st.rerun()
    
    selected_assets.extend(st.session_state.custom_indices)
    
    return list(set(selected_assets))

def get_expanded_assets(available_assets):
    # Expande lista de ativos com versões em BRL para internacionais (USD)
    usd_assets = ['S&P 500', 'IMID', 'Bitcoin', 'Ethereum']
    expanded_assets = []
    for a in sorted(available_assets):
        expanded_assets.append(a)
        if a in usd_assets:
            expanded_assets.append(f"{a} BRL")
    return expanded_assets

def render_benchmark_selector(available_assets):
    """Renderiza a tabela de seleção de benchmarks individuais."""
    st.subheader("2. Seleção de Benchmarks")
    
    if not available_assets:
        st.info("👈 Selecione ativos na barra lateral para começar.")
        return []
        
    expanded_assets = get_expanded_assets(available_assets)
    
    # Gerenciamento de Exclusões
    if "benchmarks_deleted" not in st.session_state:
        st.session_state.benchmarks_deleted = set()
    
    visible_assets = [a for a in expanded_assets if a not in st.session_state.benchmarks_deleted]

    # --- Benchmarks Individuais ---
    with st.expander("Benchmarks Individuais (Plotar Linhas)", expanded=True):
        df_bench = pd.DataFrame({"Ativo": visible_assets, "Incluir": True})
        edited_bench = st.data_editor(
            df_bench, 
            column_config={
                "Ativo": st.column_config.TextColumn("Ativo", disabled=True),
                "Incluir": st.column_config.CheckboxColumn("Plotar", width="small"),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key="editor_bench_individual"
        )
        
        # Detecta exclusões feitas diretamente na tabela
        current_assets = set(edited_bench["Ativo"].dropna().tolist())
        deleted_assets = [a for a in visible_assets if a not in current_assets]
        
        if deleted_assets:
            st.session_state.benchmarks_deleted.update(deleted_assets)
            st.rerun()

    active_benchmarks = edited_bench[edited_bench["Incluir"]]["Ativo"].dropna().tolist()
    return active_benchmarks

def render_custom_portfolio_builder(available_assets):
    """Renderiza a área de construção de carteiras personalizadas."""
    st.subheader("3. Carteiras Personalizadas")
    
    if not available_assets:
        return []
    
    expanded_assets = get_expanded_assets(available_assets)
    
    if "custom_portfolios" not in st.session_state:
        # Inicializa com todos os ativos disponíveis (peso 0) para facilitar edição
        initial_weights = {a: 0.0 for a in expanded_assets}
        st.session_state.custom_portfolios = [{"name": "Carteira 1", "weights": initial_weights}]

    if st.button("➕ Nova Carteira"):
        new_id = len(st.session_state.custom_portfolios) + 1
        initial_weights = {a: 0.0 for a in expanded_assets}
        st.session_state.custom_portfolios.append({"name": f"Carteira {new_id}", "weights": initial_weights})
    
    custom_portfolios_list = []
    indices_to_remove = []
    
    if st.session_state.custom_portfolios:
        tabs = st.tabs([p["name"] if p["name"] else f"Carteira {i+1}" for i, p in enumerate(st.session_state.custom_portfolios)])
        
        for i, tab in enumerate(tabs):
            with tab:
                p = st.session_state.custom_portfolios[i]
                col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
                with col1:
                    new_name = st.text_input(f"Nome", value=p["name"], key=f"name_{i}")
                    st.session_state.custom_portfolios[i]["name"] = new_name
                with col2:
                    if st.button("🗑️", key=f"del_{i}"): indices_to_remove.append(i)
                
                # Prepara dados para o editor
                current_weights = p.get("weights", {})
                
                # Garante que as opções incluam ativos já presentes na carteira (mesmo que desmarcados na sidebar)
                all_options = sorted(list(set(expanded_assets + list(current_weights.keys()))))
                
                # Constrói dados baseados apenas no que está salvo em 'weights' (permite exclusão)
                data = [{"Ativo": k, "Peso (%)": v * 100} for k, v in current_weights.items()]
                
                edited_weights = st.data_editor(
                    pd.DataFrame(data),
                    column_config={
                        "Ativo": st.column_config.SelectboxColumn("Ativo", options=all_options, required=True),
                        "Peso (%)": st.column_config.NumberColumn("Peso (%)", min_value=0, max_value=100, format="%.1f")
                    },
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"editor_port_{i}"
                )
                
                # Processa pesos
                composition_state = {} # Para persistência na UI (inclui zeros)
                composition_calc = {}  # Para cálculo (apenas > 0)
                total_w = 0.0
                for _, row in edited_weights.iterrows():
                    asset = row["Ativo"]
                    w = row["Peso (%)"]
                    
                    if asset:
                        composition_state[asset] = w / 100.0
                        total_w += w
                        if w > 0:
                            composition_calc[asset] = w / 100.0
                
                st.session_state.custom_portfolios[i]["weights"] = composition_state
                
                if composition_calc:
                    if abs(total_w - 100.0) > 0.1:
                        st.warning(f"Total: {total_w:.1f}%. Ajuste para 100%.")
                    else:
                        st.success(f"Carteira válida.")
                        custom_portfolios_list.append({"nome": new_name, "composicao": composition_calc})
                else:
                    st.caption("Defina os pesos acima.")

    if indices_to_remove:
        for index in sorted(indices_to_remove, reverse=True):
            del st.session_state.custom_portfolios[index]
        st.rerun()
        
    return custom_portfolios_list

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
        benchmarks_list = render_benchmark_selector(selected_assets)
        
    with st.container(border=True):
        portfolios_list = render_custom_portfolio_builder(selected_assets)
        
    active_benchmarks_list = benchmarks_list + portfolios_list

    # 2. Seleção de Ativos da Carteira (Condicional)
    if comparar_carteira:
        with st.container(border=True):
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
