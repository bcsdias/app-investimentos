import streamlit as st
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from src.data.benchmarks_config import CATALOGO_YF, CATALOGO_B3, CATALOGO_BCB, CATALOGO_TD, CATALOGO_CRYPTO
import pandas as pd

@st.cache_data
def load_assets_from_csv():
    # Simulando load que estava no web_app original
    csv_path = os.path.join(BASE_DIR, 'data', 'ativos.csv')
    if not os.path.exists(csv_path): return {}, {}
    
    try:
        df = pd.read_csv(csv_path, sep=';')
        br_assets, intl_assets = {}, {}
        classes_br = ['ACAO', 'ETF', 'ETF RF', 'FI_INFRA', 'FI_SET', 'FIDC', 'FII', 'FIP', 'FIP_IE', 'BDR', 'BDR ETF']
        classes_intl = ['ETF_GB', 'ETF_US', 'REIT', 'STOCK']
        
        for _, row in df.iterrows():
            classe, ticker, market_cod = str(row['classe']).strip(), str(row['sigla']).strip(), str(row['market_cod']).strip()
            yf_ticker = ticker
            if ':' in market_cod:
                exchange = market_cod.split(':')[0]
                if exchange == 'BVMF': yf_ticker = f"{ticker}.SA"
                elif exchange == 'LON': yf_ticker = f"{ticker}.L"
            
            display_name = f"{ticker} - {classe}"
            if classe in classes_br: br_assets[display_name] = yf_ticker
            elif classe in classes_intl: intl_assets[display_name] = yf_ticker
        return br_assets, intl_assets
    except Exception:
        return {}, {}
        
def get_asset_categories():
    br_assets, intl_assets = load_assets_from_csv()
    CATALOGO_YF.update(br_assets)
    CATALOGO_YF.update(intl_assets)
    CATALOGO_YF.update(CATALOGO_CRYPTO)
    
    categories = {"Bolsa Brasil": {}, "Bolsa Internacional": {}, "Criptomoedas": [], "Indices": [], "Tesouro Direto": []}
    
    map_classes_br = {'ACAO': 'AÇÕES', 'FII': 'FII', 'ETF': 'ETF', 'ETF RF': 'ETF RENDA FIXA', 'FI_INFRA': 'FI INFRA', 'FIP': 'FIP/FIP IE', 'FIP_IE': 'FIP/FIP IE', 'BDR': 'BDR', 'BDR ETF': 'BDR'}
    for name in br_assets.keys():
        parts = name.split(' - ')
        if len(parts) < 2: continue
        c = map_classes_br.get(parts[1], parts[1])
        if c not in categories["Bolsa Brasil"]: categories["Bolsa Brasil"][c] = []
        categories["Bolsa Brasil"][c].append(name)
        
    map_classes_intl = {'STOCK': 'STOCK', 'REIT': 'REIT', 'ETF': 'ETF', 'ETF_US': 'ETF', 'ETF_GB': 'ETF'}
    for name in intl_assets.keys():
        parts = name.split(' - ')
        if len(parts) < 2: continue
        c = map_classes_intl.get(parts[1], parts[1])
        if c not in categories["Bolsa Internacional"]: categories["Bolsa Internacional"][c] = []
        categories["Bolsa Internacional"][c].append(name)

    for name in CATALOGO_YF.keys():
        if name in br_assets or name in intl_assets or name in CATALOGO_CRYPTO: continue
        if name in ['S&P 500', 'Ibovespa (YF)']: categories["Indices"].append(name)
        elif name in ['IVVB11', 'IMID']:
            if "ETF" not in categories["Bolsa Internacional"]: categories["Bolsa Internacional"]["ETF"] = []
            categories["Bolsa Internacional"]["ETF"].append(name)
        else:
            if "AÇÕES" not in categories["Bolsa Brasil"]: categories["Bolsa Brasil"]["AÇÕES"] = []
            if name != 'S&P 500 BRL': categories["Bolsa Brasil"]["AÇÕES"].append(name)

    categories["Criptomoedas"].extend(list(CATALOGO_CRYPTO.keys()))
    categories["Indices"].extend(list(CATALOGO_B3.keys()) + list(CATALOGO_BCB.keys()))
    categories["Tesouro Direto"].extend(list(CATALOGO_TD.keys()))
    
    return categories

def render_sidebar_asset_selection():
    # Garantir que o estado da sessão esteja inicializado (necessário para navegação direta entre páginas)
    if "custom_indices" not in st.session_state:
        st.session_state.custom_indices = []
        
    st.sidebar.header("1. Seleção de Ativos Base")
    categories = get_asset_categories()
    selected_assets = []
    
    for cat_name in ["Bolsa Brasil", "Bolsa Internacional", "Criptomoedas", "Indices", "Tesouro Direto"]:
        if cat_name not in categories: continue
        cat_content = categories[cat_name]
        with st.sidebar.expander(cat_name, expanded=False):
            if isinstance(cat_content, dict):
                for sub_cat in sorted(cat_content.keys()):
                    sel = st.multiselect(sub_cat, sorted(list(set(cat_content[sub_cat]))), key=f"sel_{cat_name}_{sub_cat}")
                    selected_assets.extend(sel)
            else:
                sel = st.multiselect("Selecione", sorted(list(set(cat_content))), key=f"sel_{cat_name}", label_visibility="collapsed")
                selected_assets.extend(sel)

    # Índices Personalizados
    st.sidebar.markdown("---")
    st.sidebar.subheader("Índices Personalizados")
    with st.sidebar.expander("➕ Criar Novo", expanded=False):
        c_type = st.selectbox("Tipo", ["IPCA +", "CDI +", "% do CDI"], key="idx_type")
        c_val = st.number_input("Taxa (%)", min_value=0.0, value=0.0, step=0.1, format="%.2f", key="idx_val")
        if st.button("Adicionar") and c_val > 0:
            new_idx = f"IPCA + {c_val:.1f}%" if c_type == "IPCA +" else (f"CDI + {c_val:.1f}%" if c_type == "CDI +" else f"{c_val:.0f}% do CDI")
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
    categories = get_asset_categories()
    brl_native_assets = set(list(CATALOGO_B3.keys()) + list(CATALOGO_BCB.keys()) + ['Ibovespa (YF)'])
    if "Bolsa Brasil" in categories:
        for s in categories["Bolsa Brasil"].values(): brl_native_assets.update(s)
    if "Tesouro Direto" in categories: brl_native_assets.update(categories["Tesouro Direto"])
    
    expanded_assets = []
    for a in sorted(available_assets):
        expanded_assets.append(a)
        if a not in brl_native_assets and not a.endswith(' BRL') and '+' not in a and '%' not in a:
            expanded_assets.append(f"{a} BRL")
    return expanded_assets
