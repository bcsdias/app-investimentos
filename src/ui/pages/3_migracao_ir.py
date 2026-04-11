import streamlit as st
import pandas as pd
import os
from datetime import datetime
from src.utils.logger import logger
from src.data.sources.market_data import buscar_resumo_carteira as get_wallet_data
logger.info("Página Rentabilidade acessada")
# Page Setup
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] p {
        color: #4b5563 !important;
    }
    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_migration_tool():
    st.title("🔄 Planejador de Migração (Eficiência Fiscal)")
    st.markdown("### Transição de Carteira com Isenção de IR (R$ 20k/mês)")
    st.markdown("Planeje a transição da sua carteira atual para uma alocação simplificada, respeitando o limite mensal de isenção de IR para Ações (R$ 20.000,00).")
    
    # 1. Recuperação de Token (Sessão ou Input)
    env_token = st.session_state.get('dlp_token', os.getenv('DLP_TOKEN', ''))
    
    with st.sidebar:
        st.header("🔑 Autenticação")
        token = st.text_input("Token API (DLP)", value=env_token, type="password", key="token_migracao")
        if not token:
            st.info("Insira seu token da DLP para carregar a carteira atual.")
            st.stop()
            
    # 2. Carregamento de Dados
    with st.status("📥 Carregando carteira...", expanded=False) as status:
        wallet_data = get_wallet_data(token)
        if not wallet_data or 'wallet' not in wallet_data:
            status.update(label="❌ Erro ao carregar carteira", state="error")
            st.error("Não foi possível carregar os dados da DLP. Verifique seu token.")
            return
        status.update(label="✅ Carteira carregada!", state="complete")
        
    wallet_items = wallet_data.get('wallet', [])
    classes_disponiveis = sorted(list(set([item.get('classe', 'Outros') for item in wallet_items])))
    
    # 3. Configurações da Migração
    with st.container(border=True):
        st.markdown("#### ⚙️ Configurações da Estratégia")
        col1, col2, col3 = st.columns([1, 1.2, 0.8])
        
        with col1:
            st.markdown("**Regras de Isenção**")
            # R$ 19.500 de default para dar margem de segurança
            limite_mensal = st.number_input("Limite de Venda Mensal (R$)", min_value=1000, max_value=50000, value=19500, step=100, help="Limite para isenção de IR em Ações no Brasil.")
            classes_venda = st.multiselect("Classes para Vender", classes_disponiveis, default=["AÇÃO"] if "AÇÃO" in classes_disponiveis else classes_disponiveis[:1])
            st.caption("*A isenção aplica-se à classe AÇÃO.*")
        
        with col2:
            st.markdown("**Alocação Alvo (%)**")
            c1, c2, c3 = st.columns(3)
            with c1: peso_b5p2 = st.number_input("B5P211", min_value=0, max_value=100, value=60)
            with c2: peso_vwra = st.number_input("VWRA11", min_value=0, max_value=100, value=35)
            with c3: peso_bith = st.number_input("BITH11", min_value=0, max_value=100, value=5)
            
        with col3:
            st.markdown("**Validação**")
            soma = peso_b5p2 + peso_vwra + peso_bith
            if soma == 100:
                st.success(f"**Soma: {soma}%** ✅")
            else:
                st.error(f"**Soma: {soma}%** (Ajuste para 100%)")
            
    if soma != 100:
        st.warning("⚠️ Ajuste os pesos da alocação alvo para somar 100% antes de prosseguir.")
        st.stop()
        
    if not classes_venda:
        st.info("👆 Selecione pelo menos uma classe para vender.")
        st.stop()
        
    # 4. Processamento dos Itens de Venda
    itens_venda = []
    for item in wallet_items:
        qtd = item.get('quantidade', item.get('qtd', 0))
        if item.get('classe') in classes_venda and qtd > 0:
            preco = item.get('price', 0)
            vlr_mercado = qtd * preco
            if vlr_mercado > 0:
                itens_venda.append({
                    'Ativo': item.get('ativo'),
                    'Classe': item.get('classe'),
                    'Qtd Atual': qtd,
                    'Preço Atual': preco,
                    'Valor Total (R$)': vlr_mercado
                })
                
    if not itens_venda:
        st.warning(f"Nenhum ativo saldo em carteira para as classes selecionadas: {', '.join(classes_venda)}")
        return
        
    df_venda_origem = pd.DataFrame(itens_venda)
    total_vender = df_venda_origem['Valor Total (R$)'].sum()
    
    # 5. Dashboard de Resumo
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric("Total em Liquidação", f"R$ {total_vender:,.2f}")
        st.write(f"💼 **{len(itens_venda)}** ativos identificados.")
        
    with res_col2:
        with st.expander("🔍 Ver Detalhes dos Ativos Origem"):
            st.dataframe(df_venda_origem, use_container_width=True, hide_index=True)
        
    if st.button("🚀 Gerar Plano de Migração Mensal", type="primary", use_container_width=True):
        # Ordenação inteligente: Ativos menores primeiro para limpar o "rabo" da carteira rápido
        df_venda = df_venda_origem.sort_values(by='Valor Total (R$)')
        
        meses = []
        mes_atual_vendas = []
        mes_atual_total = 0.0
        
        for _, row in df_venda.iterrows():
            ativo = row['Ativo']
            qtd_restante = row['Qtd Atual']
            preco = row['Preço Atual']
            
            while qtd_restante > 0:
                capacidade_restante = limite_mensal - mes_atual_total
                
                if preco > limite_mensal and mes_atual_total == 0:
                    # Caso cota única maior que limite (ex: BRK.A)
                    qtd_vender = 1
                elif capacidade_restante < preco:
                    # Inicia novo mês
                    meses.append({'vendas': mes_atual_vendas, 'total': mes_atual_total})
                    mes_atual_vendas = []
                    mes_atual_total = 0.0
                    continue
                else:
                    max_qtd_possivel = int(capacidade_restante // preco)
                    qtd_vender = min(qtd_restante, max_qtd_possivel)
                    
                vlr_venda = qtd_vender * preco
                mes_atual_vendas.append({
                    'Ativo': ativo,
                    'Qtd Vender': int(qtd_vender),
                    'Preço Ref.': preco,
                    'Valor (R$)': vlr_venda
                })
                mes_atual_total += vlr_venda
                qtd_restante -= qtd_vender
                
        if mes_atual_vendas:
            meses.append({'vendas': mes_atual_vendas, 'total': mes_atual_total})
            
        # 6. Exibição do Plano Gerado
        st.markdown(f"### 📅 Plano de Migração ({len(meses)} meses)")
        st.success(f"Plano gerado com sucesso! Tempo estimado: **{len(meses)} meses** para liquidação total com isenção.")
        
        for i, mes in enumerate(meses):
            total_mes = mes['total']
            
            with st.expander(f"Mês {i+1} - Total: R$ {total_mes:,.2f}", expanded=(i==0)):
                col_v, col_c = st.columns(2)
                with col_v:
                    st.markdown("##### 🔴 VENDER")
                    df_mes_venda = pd.DataFrame(mes['vendas'])
                    st.dataframe(df_mes_venda, hide_index=True, use_container_width=True)
                    
                with col_c:
                    st.markdown("##### 🟢 COMPRAR")
                    compras = []
                    if peso_b5p2 > 0: compras.append({"Ativo": "B5P211", "Valor (R$)": total_mes * (peso_b5p2/100)})
                    if peso_vwra > 0: compras.append({"Ativo": "VWRA11", "Valor (R$)": total_mes * (peso_vwra/100)})
                    if peso_bith > 0: compras.append({"Ativo": "BITH11", "Valor (R$)": total_mes * (peso_bith/100)})
                    
                    st.dataframe(pd.DataFrame(compras), hide_index=True, use_container_width=True)

# Main execution
if __name__ == "__main__":
    render_migration_tool()
