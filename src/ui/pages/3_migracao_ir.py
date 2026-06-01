import streamlit as st
import pandas as pd
import os
from src.ui.components.theme import apply_theme, render_theme_toggle
from src.ui.components.headers import render_page_header
from src.utils.logger import logger
from src.data.sources.market_data import buscar_resumo_carteira as get_wallet_data

# Configuração da Página
st.set_page_config(page_title="Migração IR", page_icon="🔄", layout="wide")

# Aplica Tema Visual
apply_theme()

# Sidebar - Tema e Autenticação
with st.sidebar:
    render_theme_toggle()
    st.markdown("---")
    st.header("🔑 Autenticação")
    env_token = st.session_state.get('dlp_token', os.getenv('DLP_TOKEN', ''))
    token = st.text_input("Token API (DLP)", value=env_token, type="password", key="token_migracao")
    if not token:
        st.info("Insira seu token da DLP para carregar a carteira atual.")
        st.stop()

def calcular_compras_rebalanceadas(total_aporte, pesos_alvo, valores_atuais):
    """
    Calcula o valor de compra para cada ativo alvo baseado em rebalanceamento.
    Evita vendas de ativos alvo (rebalanceamento apenas com aportes).
    """
    ativos = list(pesos_alvo.keys())
    soma_pesos = sum(pesos_alvo.values())
    if soma_pesos == 0:
        return {a: 0.0 for a in ativos}
    
    w_alvo = {a: pesos_alvo[a] / soma_pesos for a in ativos}
    valores_pos = {a: valores_atuais.get(a, 0.0) for a in ativos}
    
    ativos_ativos = set(ativos)
    while True:
        soma_w_ativos = sum(w_alvo[a] for a in ativos_ativos)
        if soma_w_ativos == 0:
            for a in ativos:
                valores_pos[a] = valores_atuais.get(a, 0.0)
            break
            
        soma_v_ativos = sum(valores_atuais.get(a, 0.0) for a in ativos_ativos)
        total_alocado_ativos = soma_v_ativos + total_aporte
        
        diferencas = {}
        for a in ativos_ativos:
            v_alvo = (w_alvo[a] / soma_w_ativos) * total_alocado_ativos
            diferencas[a] = v_alvo - valores_atuais.get(a, 0.0)
            
        excesso = [a for a, diff in diferencas.items() if diff < 0]
        if not excesso:
            for a in ativos:
                if a in ativos_ativos:
                    valores_pos[a] = valores_atuais.get(a, 0.0) + diferencas[a]
                else:
                    valores_pos[a] = valores_atuais.get(a, 0.0)
            break
        else:
            ativos_ativos -= set(excesso)
            
    compras = {}
    for a in ativos:
        compras[a] = max(0.0, valores_pos[a] - valores_atuais.get(a, 0.0))
        
    return compras

def render_migration_tool():
    logger.info("Página Migração IR acessada")
    
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
    
    # Obter saldos atuais dos ativos alvo na carteira
    valores_atuais_iniciais = {
        "IMAB11": 0.0,
        "VWRA11": 0.0,
        "BITH11": 0.0
    }
    for item in wallet_items:
        ativo = item.get('ativo')
        if ativo in valores_atuais_iniciais:
            qtd = item.get('quantidade', item.get('qtd', 0))
            preco = item.get('price', 0.0)
            valores_atuais_iniciais[ativo] = qtd * preco
    
    # Header
    render_page_header(
        title="Planejador de Migração", 
        icon="🔄", 
        description="Planeje a transição da sua carteira para uma alocação alvo respeitando o limite mensal de isenção de IR (R$ 20k)."
    )
    
    # 3. Configurações da Migração em Card
    with st.container(border=True):
        st.markdown("#### ⚙️ Estratégia de Transição")
        col1, col2, col3 = st.columns([1, 1.2, 0.8])
        
        with col1:
            st.markdown("**Regras de Isenção**")
            limite_mensal = st.number_input("Limite de Venda Mensal (R$)", min_value=1000, max_value=50000, value=19500, step=100, key="mig_limite_mensal")
            classes_venda = st.multiselect("Classes para Vender", classes_disponiveis, default=["AÇÃO"] if "AÇÃO" in classes_disponiveis else classes_disponiveis[:1], key="mig_classes_venda")
        
        with col2:
            st.markdown("**Alocação Alvo (%)**")
            c1, c2, c3 = st.columns(3)
            with c1: 
                peso_imab = st.number_input("IMAB11", min_value=0, max_value=100, value=60, key="mig_peso_imab")
                st.caption(f"Atual: R$ {valores_atuais_iniciais['IMAB11']:,.2f}")
            with c2: 
                peso_vwra = st.number_input("VWRA11", min_value=0, max_value=100, value=35, key="mig_peso_vwra")
                st.caption(f"Atual: R$ {valores_atuais_iniciais['VWRA11']:,.2f}")
            with c3: 
                peso_bith = st.number_input("BITH11", min_value=0, max_value=100, value=5, key="mig_peso_bith")
                st.caption(f"Atual: R$ {valores_atuais_iniciais['BITH11']:,.2f}")
            
        with col3:
            st.markdown("**Validação**")
            soma = peso_imab + peso_vwra + peso_bith
            if soma == 100:
                st.success(f"**Soma: {soma}%** ✅")
            else:
                st.error(f"**Soma: {soma}%** (Ajuste para 100%)")
            
    if soma != 100 or not classes_venda:
        st.warning("⚠️ Ajuste as configurações acima para prosseguir.")
        st.stop()
        
    # 4. Processamento
    itens_venda = []
    for item in wallet_items:
        qtd = item.get('quantidade', item.get('qtd', 0))
        if item.get('classe') in classes_venda and qtd > 0:
            preco = item.get('price', 1.0) # Fallback 1.0
            vlr_mercado = qtd * preco
            if vlr_mercado > 0:
                itens_venda.append({
                    'Ativo': item.get('ativo'),
                    'Classe': item.get('classe'),
                    'Qtd': qtd,
                    'Preço': preco,
                    'Total (R$)': vlr_mercado
                })
                
    if not itens_venda:
        st.info("Nenhum ativo encontrado para as classes selecionadas.")
        return
        
    df_venda_origem = pd.DataFrame(itens_venda)
    total_vender = df_venda_origem['Total (R$)'].sum()
    
    # 5. Resumo da Origem em Colunas
    st.markdown("<br>", unsafe_allow_html=True)
    c_m1, c_m2 = st.columns([1, 2])
    with c_m1:
        st.metric("Total à Liquidar", f"R$ {total_vender:,.2f}")
    with c_m2:
        with st.expander("🔍 Detalhes da Origem"):
            st.dataframe(df_venda_origem, use_container_width=True, hide_index=True)
        
    if st.button("🚀 Gerar Plano de Migração Mensal", type="primary", use_container_width=True):
        df_venda = df_venda_origem.sort_values(by='Total (R$)')
        meses = []
        mes_atual_vendas = []
        mes_atual_total = 0.0
        
        for _, row in df_venda.iterrows():
            ativo = row['Ativo']
            qtd_restante = row['Qtd']
            preco = row['Preço']
            
            while qtd_restante > 0:
                cap_restante = limite_mensal - mes_atual_total
                if preco > limite_mensal and mes_atual_total == 0:
                    qtd_vender = 1
                elif cap_restante < preco:
                    meses.append({'vendas': mes_atual_vendas, 'total': mes_atual_total})
                    mes_atual_vendas, mes_atual_total = [], 0.0
                    continue
                else:
                    max_qtd = int(cap_restante // preco)
                    qtd_vender = min(qtd_restante, max_qtd)
                    
                vlr_v = qtd_vender * preco
                mes_atual_vendas.append({'Ativo': ativo, 'Qtd': int(qtd_vender), 'Valor': vlr_v})
                mes_atual_total += vlr_v
                qtd_restante -= qtd_vender
        if mes_atual_vendas: meses.append({'vendas': mes_atual_vendas, 'total': mes_atual_total})
        
        # Inicializar os valores acumulados para a projeção de rebalanceamento
        valores_atuais_proj = valores_atuais_iniciais.copy()
        pesos_alvo = {
            "IMAB11": peso_imab,
            "VWRA11": peso_vwra,
            "BITH11": peso_bith
        }
        
        for mes in meses:
            # Calcular compras para este mês usando rebalanceamento dinâmico
            compras_sugeridas = calcular_compras_rebalanceadas(
                mes['total'],
                pesos_alvo,
                valores_atuais_proj
            )
            
            # Registrar saldos finais projetados para exibição
            saldos_finais = {}
            for ativo, v_compra in compras_sugeridas.items():
                valores_atuais_proj[ativo] += v_compra
                saldos_finais[ativo] = valores_atuais_proj[ativo]
                
            mes['compras'] = compras_sugeridas
            mes['saldos_finais'] = saldos_finais
            
        # 6. Exibição do Plano Gerado
        st.markdown(f"### 📅 Plano de Migração ({len(meses)} meses)")
        for i, mes in enumerate(meses):
            with st.expander(f"Mês {i+1} - Total: R$ {mes['total']:,.2f}", expanded=(i==0)):
                cv, cc = st.columns(2)
                with cv:
                    st.markdown("**🔴 VENDER**")
                    st.table(pd.DataFrame(mes['vendas']))
                with cc:
                    st.markdown("**🟢 COMPRAR (Sugestão Rebalanceada)**")
                    compras_data = []
                    soma_valores = sum(mes['saldos_finais'].values())
                    
                    for ativo in ["IMAB11", "VWRA11", "BITH11"]:
                        valor_compra = mes['compras'].get(ativo, 0.0)
                        saldo_final = mes['saldos_finais'].get(ativo, 0.0)
                        part_final = (saldo_final / soma_valores * 100) if soma_valores > 0 else 0.0
                        
                        if valor_compra > 0 or saldo_final > 0:
                            compras_data.append({
                                "Ativo": ativo,
                                "Valor Compra (R$)": f"R$ {valor_compra:,.2f}",
                                "Saldo Final (R$)": f"R$ {saldo_final:,.2f}",
                                "Part. Final (%)": f"{part_final:.1f}%"
                            })
                    
                    if compras_data:
                        st.table(pd.DataFrame(compras_data))
                    else:
                        st.info("Nenhuma compra sugerida para este mês.")

if __name__ == "__main__":
    render_migration_tool()
