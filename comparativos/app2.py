import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Função para ler a planilha
def ler_planilha(caminho):
    df = pd.read_excel(caminho)
    return df

# Função para buscar o preço atual do ativo
def buscar_preco_atual(codigo):
    try:
        ticker = yf.Ticker(f"{codigo}.SA")
        dados = ticker.history(period='1d')
        
        if not dados.empty:
            return dados['Close'].iloc[-1]
        else:
            raise ValueError(f"Sem dados disponíveis para o ativo {codigo}.")
    
    except Exception as e:
        print(f"Erro ao buscar preço de {codigo}: {e}")
        return None

# Função para calcular o valor da carteira ao longo do tempo
def calcular_valor_carteira(df):
    df['Data operação'] = pd.to_datetime(df['Data operação'], dayfirst=True)

    df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce')
    df['Preço unitário'] = pd.to_numeric(df['Preço unitário'], errors='coerce')
    df['Corretagem'] = pd.to_numeric(df['Corretagem'], errors='coerce')
    df['Taxas'] = pd.to_numeric(df['Taxas'], errors='coerce')
    df['Impostos'] = pd.to_numeric(df['Impostos'], errors='coerce')

    df = df.sort_values('Data operação')

    patrimonio = []
    datas = []
    valor_carteira = 0
    carteira = {}

    for index, row in df.iterrows():
        codigo = row['Código Ativo']
        quantidade = row['Quantidade']
        preco_unitario = row['Preço unitário']
        corretagem = row['Corretagem']
        taxas = row['Taxas']
        impostos = row['Impostos']

        custo_operacao = quantidade * preco_unitario + corretagem + taxas + impostos

        if row['Operação C/V'] == 'C':
            if codigo in carteira:
                carteira[codigo] += quantidade
            else:
                carteira[codigo] = quantidade
            valor_carteira += custo_operacao
        elif row['Operação C/V'] == 'V':
            if codigo in carteira:
                carteira[codigo] -= quantidade
            valor_carteira -= custo_operacao

        patrimonio.append(valor_carteira)
        datas.append(row['Data operação'])

    # Atualiza o valor da carteira com os preços atuais
    valor_atualizado = 0
    for codigo in carteira:
        preco_atual = buscar_preco_atual(codigo)
        if preco_atual:
            valor_atualizado += carteira[codigo] * preco_atual

    patrimonio.append(valor_atualizado)
    datas.append(datetime.now())

    return datas, patrimonio

# Função para buscar os dados de um índice
def buscar_dados_indice(codigo, start, end):
    try:
        ticker = yf.Ticker(codigo)
        dados = ticker.history(start=start, end=end)
        if not dados.empty:
            return dados['Close']
        else:
            raise ValueError(f"Sem dados disponíveis para o índice {codigo}.")
    except Exception as e:
        print(f"Erro ao buscar dados de {codigo}: {e}")
        return None

# Função para gerar gráfico comparativo da evolução do patrimônio e índices
def gerar_grafico_comparativo(datas, patrimonio, ibov, idiv=None, ibsd=None):
    datas = pd.to_datetime(datas)

    plt.figure(figsize=(10, 6))

    # Normalizando os dados para começar em 100
    patrimonio_normalizado = [v / patrimonio[0] * 100 for v in patrimonio]
    plt.plot(datas, patrimonio_normalizado, marker='o', linestyle='-', color='b', label='Patrimônio')

    if ibov is not None:
        ibov_normalizado = (ibov / ibov.iloc[0]) * 100
        plt.plot(ibov.index, ibov_normalizado, linestyle='-', color='r', label='IBOV')

    if idiv is not None:
        idiv_normalizado = (idiv / idiv.iloc[0]) * 100
        plt.plot(idiv.index, idiv_normalizado, linestyle='-', color='g', label='IDIV')

    if ibsd is not None:
        ibsd_normalizado = (ibsd / ibsd.iloc[0]) * 100
        plt.plot(ibsd.index, ibsd_normalizado, linestyle='-', color='orange', label='IBSD')

    plt.title('Comparação da Evolução do Patrimônio e Índices')
    plt.xlabel('Data')
    plt.ylabel('Valor Normalizado (Base 100)')
    plt.legend()

    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.show()

# Caminho da planilha
caminho_planilha = 'movimentacoes.xlsx'

# Execução do script
df = ler_planilha(caminho_planilha)
datas, patrimonio = calcular_valor_carteira(df)

# Definindo o período de análise
start_date = min(datas).strftime('%Y-%m-%d')
end_date = max(datas).strftime('%Y-%m-%d')

# Buscando dados dos índices
ibov = buscar_dados_indice('^BVSP', start=start_date, end=end_date)
idiv = buscar_dados_indice('IDIV.SA', start=start_date, end=end_date)  # Atualizado para IDIV.SA
ibsd = buscar_dados_indice('IBSD.SA', start=start_date, end=end_date)  # Atualizado para IBSD.SA

# Gerando o gráfico comparativo
gerar_grafico_comparativo(datas, patrimonio, ibov, idiv, ibsd)
