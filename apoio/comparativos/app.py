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
        # Tenta buscar o preço do ativo
        ticker = yf.Ticker(f"{codigo}.SA")
        # Define o período como 1 dia
        dados = ticker.history(period='1d')
        
        # Verifica se o DataFrame retornado tem dados
        if not dados.empty:
            # Retorna o preço de fechamento mais recente
            return dados['Close'].iloc[-1]
        else:
            # Se não houver dados, gera uma exceção
            raise ValueError(f"Sem dados disponíveis para o ativo {codigo}.")
    
    except Exception as e:
        print(f"Erro ao buscar preço de {codigo}: {e}")
        return None

# Função para calcular o valor da carteira ao longo do tempo

def calcular_valor_carteira(df):
    # Corrigindo a conversão de data com dayfirst=True
    df['Data operação'] = pd.to_datetime(df['Data operação'], dayfirst=True)

    # Convertendo as colunas relevantes para o tipo numérico
    df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce')
    df['Preço unitário'] = pd.to_numeric(df['Preço unitário'], errors='coerce')
    df['Corretagem'] = pd.to_numeric(df['Corretagem'], errors='coerce')
    df['Taxas'] = pd.to_numeric(df['Taxas'], errors='coerce')
    df['Impostos'] = pd.to_numeric(df['Impostos'], errors='coerce')

    # Ordena pela data
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

        # Se for compra (C), adiciona à carteira
        if row['Operação C/V'] == 'C':
            if codigo in carteira:
                carteira[codigo] += quantidade
            else:
                carteira[codigo] = quantidade
            valor_carteira += custo_operacao

        # Se for venda (V), subtrai da carteira
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



# Função para gerar gráfico da evolução do patrimônio
def gerar_grafico(datas, patrimonio):
    # Converta as datas para o formato correto (caso não esteja)
    datas = pd.to_datetime(datas)
    
    # Cria o gráfico
    plt.figure(figsize=(10, 6))
    
    # Plotando o patrimônio ao longo do tempo
    plt.plot(datas, patrimonio, marker='o', linestyle='-', color='b')
    
    # Definindo o título e os rótulos
    plt.title('Evolução do Patrimônio ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Patrimônio (R$)')
    
    # Formatando o eixo x (datas) para garantir que mostre corretamente
    plt.gcf().autofmt_xdate()  # Rotaciona as datas para não sobrepor
    
    # Ajustando o limite do eixo x com base nos dados fornecidos
    plt.xlim([min(datas), max(datas)])

    # Exibindo o gráfico
    plt.grid(True)
    plt.show()

# Caminho da planilha
caminho_planilha = 'movimentacoes.xlsx'

# Execução do script
df = ler_planilha(caminho_planilha)
datas, patrimonio = calcular_valor_carteira(df)
gerar_grafico(datas, patrimonio)
print(f'patrimonio: {patrimonio}')
print(f'datas: {datas}')

# Converter colunas para numéricas, forçando erros a serem NaN e substituindo NaN por 0
colunas_numericas = ['Quantidade', 'Preço unitário', 'Corretagem', 'Taxas', 'Impostos']
df[colunas_numericas] = df[colunas_numericas].apply(pd.to_numeric, errors='coerce').fillna(0)

# Verifique se os dados estão corretos após a conversão
print(df[colunas_numericas].dtypes)
print(df[colunas_numericas].head())

'''
# Simulando datas e patrimônio para teste
datas = pd.date_range(start='2023-01-01', periods=12, freq='M')  # 12 meses a partir de janeiro de 2023
patrimonio = [1000, 1050, 1100, 1200, 1300, 1250, 1350, 1450, 1500, 1600, 1700, 1800]  # Valores simulados

# Função para gerar gráfico
def gerar_grafico_teste(datas, patrimonio):
    plt.figure(figsize=(10, 6))
    plt.plot(datas, patrimonio, marker='o', linestyle='-', color='b')
    plt.title('Evolução do Patrimônio ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Patrimônio (R$)')
    plt.gcf().autofmt_xdate()
    plt.xlim([min(datas), max(datas)])
    plt.grid(True)
    plt.show()

# Executando a função com dados simulados
gerar_grafico_teste(datas, patrimonio)
'''
