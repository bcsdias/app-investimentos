import yfinance as yf
import requests
import pandas as pd

# Função para pegar o IPCA esperado de uma fonte pública (exemplo, API do Banco Central)
def get_ipca():
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json"
    response = requests.get(url)
   
    # Verificando se a resposta foi bem-sucedida
    if response.status_code != 200:
        print(f"Erro ao acessar a API do Banco Central. Status code: {response.status_code}")
        return None
    
    try:
        ipca_data = response.json()
        # Pegando o último valor de IPCA disponível
        ipca_atual = float(ipca_data[-1]['valor']) / 100  # Convertemos de percentual para decimal
        return ipca_atual
    except (ValueError, IndexError) as e:
        print(f"Erro ao processar a resposta da API: {e}")
        return None
    

# Função para calcular o ROE de uma empresa
def calcular_roe(ticker):
    empresa = yf.Ticker(ticker)
    balanco = empresa.balance_sheet
    
    # Verificando as linhas disponíveis no balanço
    #print(f"Linhas do balanço para {ticker}:")
    #print(balanco.index)
    
    # Tentando diferentes alternativas para encontrar o patrimônio líquido
    patrimonio_liquido = None
    possiveis_nomes = ['Total Stockholder Equity', 'Total Equity', 'Equity', 'Stockholders Equity']

    for nome in possiveis_nomes:
        if nome in balanco.index:
            patrimonio_liquido = balanco.loc[nome, balanco.columns[0]]  # Pega o valor mais recente
            break
    
    if patrimonio_liquido is None:
        print(f"Não foi possível encontrar o patrimônio líquido para {ticker}")
        return None

    # Pegando o lucro líquido
    try:
        lucro_liquido = empresa.financials.loc['Net Income'].iloc[0]
    except KeyError:
        print(f"Não foi possível encontrar 'Net Income' para {ticker}")
        return None
    
    # Cálculo do ROE
    try:
        roe = lucro_liquido / patrimonio_liquido
        return roe
    except (TypeError, ZeroDivisionError) as e:
        print(f"Erro ao calcular o ROE para {ticker}: {e}")
        return None

# Função principal que faz a comparação com o critério IPCA + 12%
def calcular_rentabilidade(ticker):
    # Calculando o ROE da empresa
    roe = calcular_roe(ticker)
    
    # Pegando o IPCA atual
    ipca = get_ipca()
    
    if ipca is None:
        print("Não foi possível obter o IPCA, cálculo não realizado.")
        return
    
    # Cálculo do retorno requerido
    ipca_mais = 12 ## INFORMAR AQUI O QUANTO ACIMA DA INFLACAO QUER CALCULAR. EX: IPCA+12%
    retorno_requerido = ipca + ipca_mais/100
    
    # Comparando ROE com o retorno requerido, apenas se o ROE não for None
    if roe is not None:
        if roe > retorno_requerido:
            print(f"{ticker}: ROE = {roe*100:.2f}% > IPCA + {ipca_mais}% -> ATRATIVO")
        else:
            print(f"{ticker}: ROE = {roe*100:.2f}%, IPCA + 12% = {retorno_requerido*100:.2f}% -> NÃO ATRATIVO")
    else:
        print(f"{ticker}: Não foi possível calcular o ROE.")

# Lista de empresas da B3 para análise
empresas = ['PRIO3.SA', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'MRFG3.SA']

# Rodando a análise para cada empresa
for empresa in empresas:
    calcular_rentabilidade(empresa)
