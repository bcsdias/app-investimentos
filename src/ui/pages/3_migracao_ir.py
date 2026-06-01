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
    
    # Obter saldos atuais diretos dos ativos alvo na carteira
    imab11_carteira_vlr = 0.0
    vwra11_carteira_vlr = 0.0
    bith11_carteira_vlr = 0.0
    
    for item in wallet_items:
        ativo = item.get('ativo')
        qtd = item.get('quantidade', item.get('qtd', 0))
        preco = item.get('price', 0.0)
        valor_mercado = qtd * preco
        if ativo == "IMAB11":
            imab11_carteira_vlr = valor_mercado
        elif ativo == "VWRA11":
            vwra11_carteira_vlr = valor_mercado
        elif ativo == "BITH11":
            bith11_carteira_vlr = valor_mercado

    # Mapear títulos de Renda Fixa adicionais na carteira
    def is_fixed_income(item):
        classe = str(item.get('classe', '')).upper()
        ativo = str(item.get('ativo', '')).upper()
        termos_rf = [
            "RENDA FIXA", "RENDA_FIXA", "CDB", "LCI", "LCA", "LIG", "LC", "LF", 
            "CRI", "CRA", "DEBENTURE", "DEBÊNTURE", "TESOURO", "TD", "POUPANÇA"
        ]
        for termo in termos_rf:
            if termo in classe or termo in ativo:
                return True
        return False

    # Mapear ativos internacionais adicionais na carteira
    def is_international(item):
        classe = str(item.get('classe', '')).upper()
        ativo = str(item.get('ativo', '')).upper()
        termos_int = [
            "INTERNACIONAL", "GLOBAL", "EXTERIOR", "BDR", "IMID", "VWRA", 
            "IVVB11", "SPXI11", "ACWI", "URTH"
        ]
        for termo in termos_int:
            if termo in classe or termo in ativo:
                return True
        return False

    def obter_preco_atual(ativo):
        # 1. Tentar buscar da carteira atual (DLP)
        for item in wallet_items:
            if item.get('ativo') == ativo:
                preco = item.get('price', 0.0)
                if preco > 0:
                    return preco
                    
        # 2. Caso não esteja na carteira, buscar via yfinance
        ticker_map = {
            "IMAB11": "IMAB11.SA",
            "VWRA11": "VWRA11.SA",
            "BITH11": "BITH11.SA"
        }
        symbol = ticker_map.get(ativo, ativo)
        try:
            import yfinance as yf
            ticker_data = yf.Ticker(symbol)
            price = ticker_data.fast_info.get('lastPrice')
            if price and price > 0:
                return price
            hist = ticker_data.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Erro ao buscar preco de {symbol} via YFinance: {e}")
            
        # 3. Fallbacks razoáveis aproximados se tudo falhar
        fallbacks = {
            "IMAB11": 100.0,
            "VWRA11": 120.0,
            "BITH11": 60.0
        }
        return fallbacks.get(ativo, 1.0)

    rf_items = []
    int_items = []
    for item in wallet_items:
        ativo = item.get('ativo')
        if ativo in ["IMAB11", "VWRA11", "BITH11"]:
            continue
        qtd = item.get('quantidade', item.get('qtd', 0))
        preco = item.get('price', 0.0)
        valor = qtd * preco
        if valor > 0:
            if is_fixed_income(item):
                rf_items.append({
                    "Ativo": ativo,
                    "Classe": item.get('classe', 'Renda Fixa'),
                    "Saldo (R$)": valor,
                    "Somar ao IMAB11": True
                })
            elif is_international(item):
                int_items.append({
                    "Ativo": ativo,
                    "Classe": item.get('classe', 'Internacional'),
                    "Saldo (R$)": valor,
                    "Somar ao VWRA11": True
                })

    # Header
    render_page_header(
        title="Planejador de Migração", 
        icon="🔄", 
        description="Planeje a transição da sua carteira para uma alocação alvo respeitando o limite mensal de isenção de IR (R$ 20k)."
    )

    # 2.5 Integração de Renda Fixa Adicional
    valor_adicional_imab = 0.0
    if rf_items:
        with st.expander("💼 Integrar Renda Fixa Existente (CDB, LCI, LCA, LIG) ao IMAB11", expanded=False):
            st.markdown(
                "Selecione quais títulos de renda fixa você deseja somar ao saldo do **IMAB11** (grupo de Renda Fixa). "
                "CDBs de liquidez diária ou títulos que você não deseja considerar no grupo de renda fixa de longo prazo podem ser desmarcados."
            )
            df_rf = pd.DataFrame(rf_items)
            edited_df = st.data_editor(
                df_rf,
                column_config={
                    "Ativo": st.column_config.TextColumn("Ativo Título", disabled=True),
                    "Classe": st.column_config.TextColumn("Classe", disabled=True),
                    "Saldo (R$)": st.column_config.NumberColumn("Saldo Atual", format="R$ %,.2f", disabled=True),
                    "Somar ao IMAB11": st.column_config.CheckboxColumn("Somar ao IMAB11?", default=True)
                },
                disabled=["Ativo", "Classe", "Saldo (R$)"],
                hide_index=True,
                use_container_width=True,
                key="mig_rf_editor"
            )
            
            ativos_somados = edited_df[edited_df["Somar ao IMAB11"] == True]["Ativo"].tolist()
            valor_adicional_imab = edited_df[edited_df["Somar ao IMAB11"] == True]["Saldo (R$)"].sum()
            
            if ativos_somados:
                st.success(f"✓ **R$ {valor_adicional_imab:,.2f}** integrados ao grupo de Renda Fixa (IMAB11).")

    # 2.6 Integração de Ativos Internacionais Adicionais
    valor_adicional_vwra = 0.0
    if int_items:
        with st.expander("💼 Integrar Ativos Internacionais Existentes (como IMID, BDRs) ao VWRA11", expanded=False):
            st.markdown(
                "Selecione quais ativos internacionais ou globais da sua carteira você deseja somar ao saldo do **VWRA11** (grupo de Ações Internacionais). "
                "Isso fará com que o rebalanceamento considere esses valores como parte do seu colchão de ações globais atual."
            )
            df_int = pd.DataFrame(int_items)
            edited_df_int = st.data_editor(
                df_int,
                column_config={
                    "Ativo": st.column_config.TextColumn("Ativo Título", disabled=True),
                    "Classe": st.column_config.TextColumn("Classe", disabled=True),
                    "Saldo (R$)": st.column_config.NumberColumn("Saldo Atual", format="R$ %,.2f", disabled=True),
                    "Somar ao VWRA11": st.column_config.CheckboxColumn("Somar ao VWRA11?", default=True)
                },
                disabled=["Ativo", "Classe", "Saldo (R$)"],
                hide_index=True,
                use_container_width=True,
                key="mig_int_editor"
            )
            
            ativos_somados_int = edited_df_int[edited_df_int["Somar ao VWRA11"] == True]["Ativo"].tolist()
            valor_adicional_vwra = edited_df_int[edited_df_int["Somar ao VWRA11"] == True]["Saldo (R$)"].sum()
            
            if ativos_somados_int:
                st.success(f"✓ **R$ {valor_adicional_vwra:,.2f}** integrados ao grupo de Ações Internacionais (VWRA11).")

    # Definir valores iniciais finais
    valores_atuais_iniciais = {
        "IMAB11": imab11_carteira_vlr + valor_adicional_imab,
        "VWRA11": vwra11_carteira_vlr + valor_adicional_vwra,
        "BITH11": bith11_carteira_vlr
    }

    # 3. Configurações da Migração em Card
    with st.container(border=True):
        st.markdown("#### ⚙️ Estratégia de Transição")
        col1, col2, col3 = st.columns([1, 1.2, 0.8])
        
        with col1:
            st.markdown("**Regras de Isenção**")
            apenas_aportes = st.checkbox("Apenas Aportes (Sem Vendas)", value=False, key="mig_apenas_aportes")
            if not apenas_aportes:
                limite_mensal = st.number_input("Limite de Venda Mensal (R$)", min_value=1000, max_value=50000, value=19500, step=100, key="mig_limite_mensal")
                classes_venda = st.multiselect("Classes para Vender", classes_disponiveis, default=["AÇÃO"] if "AÇÃO" in classes_disponiveis else classes_disponiveis[:1], key="mig_classes_venda")
            else:
                limite_mensal = 0.0
                classes_venda = []
                
            aporte_inicial = st.number_input("Saldo em Caixa / Novo Aporte (R$)", min_value=0.0, value=5000.0 if apenas_aportes else 0.0, step=500.0, key="mig_aporte_inicial")
        
        with col2:
            st.markdown("**Alocação Alvo (%)**")
            c1, c2, c3 = st.columns(3)
            with c1: 
                peso_imab = st.number_input("IMAB11", min_value=0, max_value=100, value=60, key="mig_peso_imab")
                if valor_adicional_imab > 0:
                    st.caption(f"Atual: R$ {imab11_carteira_vlr:,.2f} + R$ {valor_adicional_imab:,.2f} RF")
                else:
                    st.caption(f"Atual: R$ {valores_atuais_iniciais['IMAB11']:,.2f}")
            with c2: 
                peso_vwra = st.number_input("VWRA11", min_value=0, max_value=100, value=35, key="mig_peso_vwra")
                if valor_adicional_vwra > 0:
                    st.caption(f"Atual: R$ {vwra11_carteira_vlr:,.2f} + R$ {valor_adicional_vwra:,.2f} INT")
                else:
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
            
    if soma != 100 or (not apenas_aportes and not classes_venda):
        st.warning("⚠️ Ajuste as configurações acima para prosseguir.")
        st.stop()
        
    if apenas_aportes and aporte_inicial <= 0:
        st.warning("⚠️ Informe um valor de Saldo em Caixa / Novo Aporte maior que R$ 0,00 para rebalancear.")
        st.stop()
        
    # 4. Processamento
    itens_venda = []
    if not apenas_aportes:
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
                
    if not apenas_aportes and not itens_venda:
        st.info("Nenhum ativo encontrado para as classes selecionadas.")
        return
        
    df_venda_origem = pd.DataFrame(itens_venda)
    total_vender = df_venda_origem['Total (R$)'].sum() if not df_venda_origem.empty else 0.0
    
    # 5. Resumo da Origem em Colunas
    st.markdown("<br>", unsafe_allow_html=True)
    if apenas_aportes:
        st.metric("Total a Aportar (Novo Aporte)", f"R$ {aporte_inicial:,.2f}")
    else:
        c_m1, c_m2, c_m3 = st.columns([1, 1, 2])
        with c_m1:
            st.metric("Total à Liquidar", f"R$ {total_vender:,.2f}")
        with c_m2:
            st.metric("Saldo em Caixa", f"R$ {aporte_inicial:,.2f}")
        with c_m3:
            with st.expander("🔍 Detalhes da Origem"):
                st.dataframe(df_venda_origem, use_container_width=True, hide_index=True)
        
    # Se os parâmetros atuais mudarem, invalidar o plano antigo em cache
    pesos_alvo = {
        "IMAB11": peso_imab,
        "VWRA11": peso_vwra,
        "BITH11": peso_bith
    }
    if 'plano_gerado' in st.session_state:
        plano = st.session_state['plano_gerado']
        if (plano.get('apenas_aportes', False) != apenas_aportes or
            plano['limite_mensal'] != limite_mensal or
            plano['classes_venda'] != classes_venda or
            plano['pesos_alvo'] != pesos_alvo or
            plano['valores_atuais_iniciais'] != valores_atuais_iniciais or
            plano.get('aporte_inicial', 0.0) != aporte_inicial):
            del st.session_state['plano_gerado']

    def gerar_pdf_plano(meses, valores_atuais_iniciais, imab_direct, rf_added, vwra_direct, int_added, pesos_alvo, limite_mensal, classes_venda, aporte_inicial, apenas_aportes, precos_alvo):
        from fpdf import FPDF
        import datetime
        
        class PDFPlan(FPDF):
            def header(self):
                # Banner superior moderno
                self.set_fill_color(26, 54, 93)  # Azul Escuro Premium
                self.rect(0, 0, 210, 22, "F")
                
                self.set_text_color(255, 255, 255)
                self.set_y(4)
                self.set_font("helvetica", "B", 13)
                self.cell(0, 5, "PLANO DE MIGRAÇÃO MENSAL E ISENÇÃO DE IR", align="C", ln=True)
                self.set_font("helvetica", "I", 8)
                self.cell(0, 5, "Simulação de Transição Patrimonial Otimizada", align="C", ln=True)
                
                self.set_text_color(0, 0, 0)
                self.set_y(26)
                
            def footer(self):
                self.set_y(-15)
                self.set_font("helvetica", "I", 7)
                self.set_text_color(128, 128, 128)
                data_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                self.cell(100, 10, f"Gerado em {data_str} | Antigravity AI Portfolio", align="L")
                self.cell(90, 10, f"Página {self.page_no()}/{{nb}}", align="R")

        pdf = PDFPlan()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)
        
        # --- Seção 1: Parâmetros da Estratégia ---
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(26, 54, 93)
        pdf.cell(0, 7, "1. Parâmetros da Estratégia", ln=True)
        pdf.set_draw_color(26, 54, 93)
        pdf.set_line_width(0.3)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        if apenas_aportes:
            pdf.cell(0, 5, "Modo da Estratégia: Apenas Aportes (Sem Vendas)", ln=True)
            pdf.cell(0, 5, f"Total a Aportar (Novo Aporte): R$ {aporte_inicial:,.2f}", ln=True)
        else:
            pdf.cell(95, 5, f"Limite de Venda Mensal: R$ {limite_mensal:,.2f}", ln=False)
            pdf.cell(95, 5, f"Saldo em Caixa: R$ {aporte_inicial:,.2f}", ln=True)
            pdf.cell(0, 5, f"Classes de Venda: {', '.join(classes_venda)}", ln=True)
        
        pesos_str = " | ".join([f"{k}: {v}%" for k, v in pesos_alvo.items()])
        pdf.cell(0, 5, f"Alocação Alvo: {pesos_str}", ln=True)
        pdf.ln(4)
        
        # --- Seção 2: Saldos Iniciais Considerados ---
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(26, 54, 93)
        pdf.cell(0, 7, "2. Saldos Iniciais Considerados", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        pdf.set_font("helvetica", "B", 8)
        pdf.set_fill_color(230, 237, 245)
        pdf.set_draw_color(180, 180, 180)
        pdf.cell(50, 6, "Ativo Alvo", border=1, fill=True)
        pdf.cell(90, 6, "Composição do Saldo Inicial", border=1, fill=True)
        pdf.cell(50, 6, "Saldo Inicial Total", border=1, fill=True, align="R")
        pdf.ln()
        
        pdf.set_font("helvetica", "", 8)
        # IMAB11
        pdf.cell(50, 5, "IMAB11", border=1)
        if rf_added > 0:
            comp_str = f"Carteira (R$ {imab_direct:,.2f}) + RF Integrada (R$ {rf_added:,.2f})"
        else:
            comp_str = "Saldo Direto em Carteira"
        pdf.cell(90, 5, comp_str, border=1)
        pdf.cell(50, 5, f"R$ {valores_atuais_iniciais['IMAB11']:,.2f}", border=1, align="R")
        pdf.ln()
        
        # VWRA11
        pdf.cell(50, 5, "VWRA11", border=1)
        if int_added > 0:
            comp_str_int = f"Carteira (R$ {vwra_direct:,.2f}) + INT Integrada (R$ {int_added:,.2f})"
        else:
            comp_str_int = "Saldo Direto em Carteira"
        pdf.cell(90, 5, comp_str_int, border=1)
        pdf.cell(50, 5, f"R$ {valores_atuais_iniciais['VWRA11']:,.2f}", border=1, align="R")
        pdf.ln()
        
        # BITH11
        pdf.cell(50, 5, "BITH11", border=1)
        pdf.cell(90, 5, f"Saldo Direto em Carteira (R$ {valores_atuais_iniciais['BITH11']:,.2f})", border=1)
        pdf.cell(50, 5, f"R$ {valores_atuais_iniciais['BITH11']:,.2f}", border=1, align="R")
        pdf.ln()
        pdf.ln(5)
        
        # --- Seção 3: Cronograma Mensal ---
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(26, 54, 93)
        pdf.cell(0, 7, f"3. Cronograma de Migração ({len(meses)} meses)" if not apenas_aportes else "3. Cronograma de Aportes", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        for i, mes in enumerate(meses):
            pdf.set_font("helvetica", "B", 9)
            pdf.set_text_color(44, 82, 130)
            compra_total = mes.get('valor_compra_efetivo', mes['total'])
            if apenas_aportes:
                pdf.cell(0, 6, f"Aporte Único - Total a Aportar: R$ {compra_total:,.2f}", ln=True)
            else:
                pdf.cell(0, 6, f"MÊS {i+1} - Total a Liquidar: R$ {mes['total']:,.2f} | Total a Comprar: R$ {compra_total:,.2f}", ln=True)
            pdf.set_text_color(0, 0, 0)
            
            # Subtabelas de Vendas e Compras lado a lado
            y_start = pdf.get_y()
            
            # Vendas (Esquerda)
            pdf.set_font("helvetica", "B", 7)
            pdf.set_fill_color(254, 235, 235)  # Vermelho suave
            pdf.cell(92, 5, "VENDER (Origem)", border=1, fill=True, ln=True)
            
            if apenas_aportes:
                pdf.set_font("helvetica", "I", 7)
                pdf.cell(92, 5, " Nenhuma venda sugerida (Modo Apenas Aportes)", border=1, ln=True)
            else:
                pdf.cell(42, 4, "Ativo", border=1)
                pdf.cell(20, 4, "Qtd", border=1, align="C")
                pdf.cell(30, 4, "Valor", border=1, align="R")
                pdf.ln()
                
                pdf.set_font("helvetica", "", 7)
                for v in mes['vendas']:
                    pdf.cell(42, 4, str(v['Ativo']), border=1)
                    pdf.cell(20, 4, str(v['Qtd']), border=1, align="C")
                    pdf.cell(30, 4, f"R$ {v['Valor']:,.2f}", border=1, align="R")
                    pdf.ln()
                
            y_end_vendas = pdf.get_y()
            
            # Compras (Direita)
            pdf.set_y(y_start)
            pdf.set_x(108)
            
            pdf.set_font("helvetica", "B", 7)
            pdf.set_fill_color(240, 253, 244)  # Verde suave
            pdf.cell(92, 5, "COMPRAR (Destino Rebalanceado)", border=1, fill=True, ln=True)
            
            pdf.set_x(108)
            pdf.cell(15, 4, "Ativo", border=1)
            pdf.cell(18, 4, "Cotac. (R$)", border=1, align="R")
            pdf.cell(12, 4, "Qtd", border=1, align="C")
            pdf.cell(17, 4, "Compra", border=1, align="R")
            pdf.cell(18, 4, "Saldo Fin.", border=1, align="R")
            pdf.cell(12, 4, "Part. %", border=1, align="R")
            pdf.ln()
            
            pdf.set_font("helvetica", "", 7)
            soma_saldos = sum(mes['saldos_finais'].values())
            for ativo in ["IMAB11", "VWRA11", "BITH11"]:
                val_compra = mes['compras'].get(ativo, 0.0)
                sal_final = mes['saldos_finais'].get(ativo, 0.0)
                part_final = (sal_final / soma_saldos * 100) if soma_saldos > 0 else 0.0
                
                preco_unit = precos_alvo.get(ativo, 1.0)
                qtd_sug = int(val_compra // preco_unit) if val_compra > 0 and preco_unit > 0 else 0
                
                pdf.set_x(108)
                pdf.cell(15, 4, ativo, border=1)
                pdf.cell(18, 4, f"R$ {preco_unit:,.2f}", border=1, align="R")
                pdf.cell(12, 4, str(qtd_sug) if qtd_sug > 0 else "-", border=1, align="C")
                pdf.cell(17, 4, f"R$ {val_compra:,.2f}" if val_compra > 0 else "-", border=1, align="R")
                pdf.cell(18, 4, f"R$ {sal_final:,.2f}", border=1, align="R")
                pdf.cell(12, 4, f"{part_final:.1f}%", border=1, align="R")
                pdf.ln()
                
            y_end_compras = pdf.get_y()
            
            # Próxima linha
            pdf.set_y(max(y_end_vendas, y_end_compras) + 4)
            
            # Evitar quebrar órfão
            if pdf.get_y() > 250:
                pdf.add_page()
                pdf.ln(3)
                
        return bytes(pdf.output())

    if st.button("🚀 Gerar Plano de Migração Mensal" if not apenas_aportes else "🚀 Calcular Aportes de Rebalanceamento", type="primary", use_container_width=True):
        if apenas_aportes:
            meses = [{'vendas': [], 'total': 0.0}]
        else:
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
        
        # Obter cotações atuais para os ativos alvo
        precos_alvo_cotacoes = {}
        for ativo in ["IMAB11", "VWRA11", "BITH11"]:
            precos_alvo_cotacoes[ativo] = obter_preco_atual(ativo)
            
        # Inicializar os valores acumulados para a projeção de rebalanceamento
        valores_atuais_proj = valores_atuais_iniciais.copy()
        
        for idx, mes in enumerate(meses):
            # No primeiro mês, somamos o saldo em caixa (aporte inicial) ao valor a comprar
            valor_compra_mes = mes['total']
            if idx == 0:
                valor_compra_mes += aporte_inicial
                
            # Calcular compras para este mês usando rebalanceamento dinâmico
            compras_sugeridas = calcular_compras_rebalanceadas(
                valor_compra_mes,
                pesos_alvo,
                valores_atuais_proj
            )
            
            # Registrar saldos finais projetados para exibição
            saldos_finais = {}
            for ativo, v_compra in compras_sugeridas.items():
                valores_atuais_proj[ativo] += v_compra
                saldos_finais[ativo] = valores_atuais_proj[ativo]
                
            mes['valor_compra_efetivo'] = valor_compra_mes
            mes['compras'] = compras_sugeridas
            mes['saldos_finais'] = saldos_finais
            
        # Salvar no session state
        st.session_state['plano_gerado'] = {
            'meses': meses,
            'valores_atuais_iniciais': valores_atuais_iniciais,
            'imab11_carteira_vlr': imab11_carteira_vlr,
            'valor_adicional_imab': valor_adicional_imab,
            'vwra11_carteira_vlr': vwra11_carteira_vlr,
            'valor_adicional_vwra': valor_adicional_vwra,
            'pesos_alvo': pesos_alvo,
            'limite_mensal': limite_mensal,
            'classes_venda': classes_venda,
            'aporte_inicial': aporte_inicial,
            'apenas_aportes': apenas_aportes,
            'precos_alvo': precos_alvo_cotacoes
        }

    # Renderizar plano se existir no session state
    if 'plano_gerado' in st.session_state:
        plano = st.session_state['plano_gerado']
        meses = plano['meses']
        
        p_valores_atuais_iniciais = plano['valores_atuais_iniciais']
        p_imab11_carteira_vlr = plano['imab11_carteira_vlr']
        p_valor_adicional_imab = plano['valor_adicional_imab']
        p_vwra11_carteira_vlr = plano.get('vwra11_carteira_vlr', 0.0)
        p_valor_adicional_vwra = plano.get('valor_adicional_vwra', 0.0)
        p_pesos_alvo = plano['pesos_alvo']
        p_limite_mensal = plano['limite_mensal']
        p_classes_venda = plano['classes_venda']
        p_aporte_inicial = plano.get('aporte_inicial', 0.0)
        p_apenas_aportes = plano.get('apenas_aportes', False)
        p_precos_alvo = plano.get('precos_alvo', {"IMAB11": 100.0, "VWRA11": 120.0, "BITH11": 60.0})
        
        # 6. Exibição do Plano Gerado
        st.markdown("<br>", unsafe_allow_html=True)
        col_title, col_download = st.columns([2, 1])
        with col_title:
            if p_apenas_aportes:
                st.markdown("### 📅 Sugestão de Aporte Único")
            else:
                st.markdown(f"### 📅 Plano de Migração ({len(meses)} meses)")
            
        with col_download:
            try:
                pdf_bytes = gerar_pdf_plano(
                    meses=meses,
                    valores_atuais_iniciais=p_valores_atuais_iniciais,
                    imab_direct=p_imab11_carteira_vlr,
                    rf_added=p_valor_adicional_imab,
                    vwra_direct=p_vwra11_carteira_vlr,
                    int_added=p_valor_adicional_vwra,
                    pesos_alvo=p_pesos_alvo,
                    limite_mensal=p_limite_mensal,
                    classes_venda=p_classes_venda,
                    aporte_inicial=p_aporte_inicial,
                    apenas_aportes=p_apenas_aportes,
                    precos_alvo=p_precos_alvo
                )
                
                import datetime
                st.download_button(
                    label="📥 Baixar Plano em PDF",
                    data=pdf_bytes,
                    file_name=f"plano_migracao_ir_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                logger.error(f"Erro ao gerar PDF: {e}")
                st.error(f"Não foi possível gerar o PDF para download.")
                
        for i, mes in enumerate(meses):
            compra_total = mes.get('valor_compra_efetivo', mes['total'])
            venda_total = mes['total']
            
            expander_title = f"Sugestão de Aporte - Total a Comprar: R$ {compra_total:,.2f}" if p_apenas_aportes else f"Mês {i+1} - Vendas: R$ {venda_total:,.2f} | Compras: R$ {compra_total:,.2f}"
            
            with st.expander(expander_title, expanded=(i==0)):
                cv, cc = st.columns(2)
                with cv:
                    st.markdown("**🔴 VENDER**")
                    if p_apenas_aportes:
                        st.info("Nenhuma venda sugerida (Modo Apenas Aportes).")
                    else:
                        st.table(pd.DataFrame(mes['vendas']))
                with cc:
                    st.markdown("**🟢 COMPRAR (Sugestão Rebalanceada)**")
                    compras_data = []
                    soma_valores = sum(mes['saldos_finais'].values())
                    
                    for ativo in ["IMAB11", "VWRA11", "BITH11"]:
                        valor_compra = mes['compras'].get(ativo, 0.0)
                        saldo_final = mes['saldos_finais'].get(ativo, 0.0)
                        part_final = (saldo_final / soma_valores * 100) if soma_valores > 0 else 0.0
                        
                        preco_unit = p_precos_alvo.get(ativo, 1.0)
                        qtd_sug = int(valor_compra // preco_unit) if valor_compra > 0 and preco_unit > 0 else 0
                        
                        if valor_compra > 0 or saldo_final > 0:
                            compras_data.append({
                                "Ativo": ativo,
                                "Cotação (R$)": f"R$ {preco_unit:,.2f}",
                                "Qtd Sugerida": qtd_sug if qtd_sug > 0 else "-",
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
