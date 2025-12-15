import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, time, timedelta, date

# Função para pegar o IPCA dos últimos 12 meses
def get_ipca():
    print("Calculando IPCA dos ultimos 12 meses...")
    mes = date.today().month
    ano_anterior = date.today().year - 1

    # Criar a URL com a data dinâmica
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json&dataInicial=01/{mes}/{ano_anterior}"

    # Tentativas de acesso à API
    max_tentativas = 3
    for tentativa in range(max_tentativas):
        try:
            response = requests.get(url)
            # Verificando se a resposta foi bem-sucedida
            if response.status_code == 200:
                break  # Se a resposta for bem-sucedida, sai do loop
            else:
                print(f"Erro ao acessar a API do Banco Central. Status code: {response.status_code}")
                if tentativa < max_tentativas - 1:
                    print("Tentando novamente...")
                    time.sleep(2)  # Espera 2 segundos antes da próxima tentativa
        except (ValueError, IndexError) as e:
            print(f"Erro ao processar a resposta da API: {e}")
            if tentativa < max_tentativas - 1:
                print("Tentando novamente...")
                time.sleep(2)  # Espera 2 segundos antes da próxima tentativa
    else:
        # Se não conseguiu após 3 tentativas
        print("Não foi possível acessar a API após 3 tentativas.")
        return None

    try:
        ipca_data = response.json()
        # Transformando os dados em um DataFrame para fácil manipulação
        df_ipca = pd.DataFrame(ipca_data)
        df_ipca['data'] = pd.to_datetime(df_ipca['data'], format='%d/%m/%Y')
        df_ipca['valor'] = df_ipca['valor'].astype(float)

        # Calculando a variação do IPCA nos últimos 12 meses
        if len(df_ipca) >= 12:
            ipca = df_ipca['valor'].sum()
            return ipca
        else:
            print("Dados insuficientes para calcular o IPCA anual.")
            return None

    except (ValueError, IndexError) as e:
        print(f"Erro ao processar a resposta da API: {e}")
        return None


# Função para calcular o ROE de uma empresa
def calcular_roe(ticker):
    empresa = yf.Ticker(ticker)
    balanco = empresa.balance_sheet

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


# Função para calcular o Fluxo de Caixa Livre
def calcular_fcf(ticker):
    empresa = yf.Ticker(ticker)
    fluxo_caixa = empresa.cashflow

    # Verificando os nomes das linhas disponíveis no fluxo de caixa
    # print(f"Dados de fluxo de caixa para {ticker}:\n", fluxo_caixa)

    # O Fluxo de Caixa Livre é calculado como: FCF = Fluxo de Caixa das Operações - Investimentos
    try:
        fluxo_caixa_operacoes = fluxo_caixa.loc['Operating Cash Flow'].iloc[0]
        investimentos = fluxo_caixa.loc['Capital Expenditure'].iloc[0]
        fcf = fluxo_caixa_operacoes - investimentos
        return fcf
    except KeyError as e:
        print(f"Não foi possível encontrar os dados de Fluxo de Caixa Livre para {ticker}. Error: {e}")
        return None


# Função principal que faz a comparação com o critério IPCA + 12%
def calcular_rentabilidade(ticker, ipca):
    # Calculando o ROE da empresa
    roe = calcular_roe(ticker)
    # Calculando o Fluxo de Caixa Livre da empresa
    fcf = calcular_fcf(ticker)

    # Cálculo do retorno requerido
    ipca_mais = 12  # Quanto acima da inflação calcular. Ex: IPCA+12%
    retorno_requerido = (ipca + ipca_mais) / 100

    # Comparando ROE com o retorno requerido
    print("--------------------")
    if roe is not None:
        if roe <= 0:
            print(f"{ticker}\n ROE = {roe * 100:.2f}%\n ROE negativo -> NÃO ATRATIVO")
        elif roe > retorno_requerido:
            print(f"{ticker}\n ROE = {roe * 100:.2f}%\n IPCA + {ipca_mais}% = {retorno_requerido * 100:.2f}%\n ROE > IPCA+{ipca_mais} -> ATRATIVO")
        else:
            print(f"{ticker}\n ROE = {roe * 100:.2f}%\n IPCA + {ipca_mais}% = {retorno_requerido * 100:.2f}%\n ROE < IPCA+{ipca_mais} -> NÃO ATRATIVO")
    else:
        print(f"{ticker}: Não foi possível calcular o ROE.")

    # Comparando FCF com o IPCA
    if fcf is not None:
        if fcf > ipca:
            print(f"{ticker}\n Fluxo de Caixa Livre = {fcf:.2f}\n FCF > IPCA -> ATRATIVO")
        else:
            print(f"{ticker}\n Fluxo de Caixa Livre = {fcf:.2f}\n FCF < IPCA -> NÃO ATRATIVO")
    else:
        print(f"{ticker}: Não foi possível calcular o Fluxo de Caixa Livre.")


# Lista de empresas da B3 para análise
empresas = ['PRIO3.SA', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'MRFG3.SA']

# Rodando a análise para cada empresa
# Pegando o IPCA atual
ipca = get_ipca()
for empresa in empresas:
    calcular_rentabilidade(empresa, ipca)
