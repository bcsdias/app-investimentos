import requests
import os
from dotenv import load_dotenv
import pandas as pd
from requests.exceptions import RequestException
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import yfinance as yf

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém o token da variável de ambiente
token = os.getenv('DLP_TOKEN')

if not token:
    raise ValueError("A variável de ambiente 'DLP_TOKEN' não foi encontrada ou está vazia. Verifique seu arquivo .env.")

def gerar_grafico_twr(df: pd.DataFrame, ativo: str) -> pd.DataFrame | None:
    """
    Calcula o Time-Weighted Return (TWR) e gera um gráfico da sua evolução.

    Args:
        df (pd.DataFrame): DataFrame com o histórico, contendo 'date', 'vlr_mercado' e 'vlr_investido'.
        ativo (str): Nome do ativo para o título do gráfico.
    Returns:
        pd.DataFrame | None: DataFrame com os cálculos do TWR ou None se falhar.
    """
    colunas_necessarias = ['date', 'vlr_mercado', 'vlr_investido', 'proventos']
    if not all(col in df.columns for col in colunas_necessarias):
        print(f"DataFrame não contém as colunas necessárias ({', '.join(colunas_necessarias)}) para calcular o TWR.")
        return None

    # Garante que a pasta de gráficos exista
    pasta_graficos = "dlombello/graficos"
    os.makedirs(pasta_graficos, exist_ok=True)

    # 1. Preparar os dados
    df_twr = df.copy()
    df_twr['date'] = pd.to_datetime(df_twr['date'])
    df_twr = df_twr.sort_values(by='date').reset_index(drop=True)

    # 2. Calcular o fluxo de caixa (aportes/retiradas) em cada período
    # O fluxo de caixa é a variação do capital investido MENOS os proventos recebidos no período.
    # Proventos são parte do retorno, não do aporte, e são tratados como retirada de caixa.
    df_twr['fluxo_mes'] = df_twr['vlr_investido'].diff().fillna(df_twr['vlr_investido'].iloc[0]) - df_twr['proventos']

    # 3. Calcular valores do período anterior e fluxos acumulados
    # O .shift(1) "puxa" o valor da linha anterior.
    df_twr['valor_inicial_periodo'] = df_twr['vlr_mercado'].shift(1).fillna(0)
    df_twr['fluxo_acc'] = df_twr['fluxo_mes'].cumsum()

    # 4. Calcular o Retorno do Período (Holding Period Return - HPR)
    # HPR = (Valor Final de Mercado) / (Valor Inicial de Mercado + Fluxo de Caixa do Período)
    # O denominador é o valor de mercado do mês anterior somado ao dinheiro que entrou/saiu neste mês.
    denominador = df_twr['valor_inicial_periodo'] + df_twr['fluxo_mes']
    
    # Evita divisão por zero. Se o denominador for 0, o retorno do período é 0.
    # O HPR é o fator de multiplicação (ex: 1.05 para 5% de ganho).
    df_twr['hpr'] = 1.0 # Inicia com 1 (sem ganho/perda)
    df_twr.loc[denominador != 0, 'hpr'] = df_twr['vlr_mercado'] / denominador

    # 5. Calcular o TWR acumulado
    # twr_mes é o retorno do período (HPR - 1)
    df_twr['twr_mes'] = df_twr['hpr'] - 1
    # O .cumprod() multiplica acumuladamente os valores da série.
    df_twr['twr_acc'] = (df_twr['hpr'].cumprod() - 1)

    # 5.1 Calcular o lucro do mês
    df_twr['lucro_mes'] = df_twr['vlr_mercado'] - df_twr['valor_inicial_periodo'] - df_twr['fluxo_mes']

    # 6. Gerar o gráfico
    plt.figure(figsize=(12, 7))
    # Multiplicamos por 100 apenas para a exibição no gráfico
    plt.plot(df_twr['date'], df_twr['twr_acc'] * 100, marker='o', linestyle='-', color='darkorange')
    plt.title(f'Rentabilidade (TWR Acumulado) - {ativo}', fontsize=16)
    plt.ylabel('Rentabilidade Acumulada (%)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    caminho_arquivo = os.path.join(pasta_graficos, f'evolucao_twr_{ativo}.png')

    # Adiciona os rótulos de valor em cada ponto
    for index, row in df_twr.iterrows():
        plt.text(row['date'], row['twr_acc'] * 100, f' {row["twr_acc"]*100:.2f}%', va='bottom', ha='left', fontsize=9)

    # Salva os dados em um arquivo CSV (formato para Excel brasileiro)
    caminho_csv = os.path.join(pasta_graficos, f'evolucao_twr_{ativo}.csv')
    
    # Seleciona e renomeia as colunas para o formato desejado
    colunas_csv = [
        'date', 'vlr_investido', 'vlr_mercado', 'fluxo_mes', 
        'fluxo_acc', 'lucro_mes', 'twr_mes', 'twr_acc'
    ]
    df_csv = df_twr[colunas_csv].copy()

    # Formata a data para DD/MM/AAAA
    df_csv['date'] = df_csv['date'].dt.strftime('%d/%m/%Y')

    # Salva o CSV com separador de ponto e vírgula e decimal com vírgula
    df_csv.to_csv(caminho_csv, index=False, sep=';', decimal=',')

    print(f"Dados do gráfico de TWR salvos em: {caminho_csv}")

    plt.savefig(caminho_arquivo)
    print(f"Gráfico de TWR salvo com sucesso em: {caminho_arquivo}")
    
    return df_twr

def gerar_grafico_percentual(df: pd.DataFrame, ativo: str):
    """
    Gera e salva um gráfico da evolução percentual (lucro/prejuízo) de um ativo.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados do histórico.
        ativo (str): O nome do ativo para usar no título e nome do arquivo.
    """
    colunas_necessarias = ['date', 'vlr_mercado', 'vlr_investido']
    if not all(col in df.columns for col in colunas_necessarias):
        print(f"DataFrame não contém as colunas necessárias ({', '.join(colunas_necessarias)}) para gerar o gráfico percentual.")
        return

    # Garante que a pasta de gráficos exista
    pasta_graficos = "dlombello/graficos"
    os.makedirs(pasta_graficos, exist_ok=True)

    # Prepara os dados para o gráfico
    df_grafico = df.copy()
    df_grafico['date'] = pd.to_datetime(df_grafico['date'])
    df_grafico = df_grafico.sort_values(by='date')

    # Calcula a evolução percentual, tratando divisão por zero
    df_grafico['evolucao_%'] = 0.0
    df_grafico.loc[df_grafico['vlr_investido'] != 0, 'evolucao_%'] = \
        ((df_grafico['vlr_mercado'] / df_grafico['vlr_investido']) - 1) * 100

    # Cria o gráfico
    plt.figure(figsize=(12, 7))
    plt.plot(df_grafico['date'], df_grafico['evolucao_%'], marker='o', linestyle='-', color='seagreen')
    
    # Customiza o gráfico
    plt.title(f'Evolução Percentual (Lucro/Prejuízo) - {ativo}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Lucro / Prejuízo (%)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--') # Linha do 0%
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter()) # Formata o eixo Y como porcentagem
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    # Salva o gráfico em um arquivo
    caminho_arquivo = os.path.join(pasta_graficos, f'evolucao_percentual_{ativo}.png')

    # Adiciona os rótulos de valor em cada ponto
    for index, row in df_grafico.iterrows():
        plt.text(row['date'], row['evolucao_%'], f' {row["evolucao_%"]:.2f}%', va='bottom', ha='left', fontsize=9)

    # Salva os dados em um arquivo CSV (formato para Excel brasileiro)
    caminho_csv = os.path.join(pasta_graficos, f'evolucao_percentual_{ativo}.csv')
    df_grafico[['date', 'evolucao_%']].to_csv(caminho_csv, index=False, decimal=',', sep=';')
    print(f"Dados do gráfico percentual salvos em: {caminho_csv}")

    plt.savefig(caminho_arquivo)
    print(f"Gráfico percentual salvo com sucesso em: {caminho_arquivo}")

def gerar_grafico_evolucao(df: pd.DataFrame, ativo: str):
    """
    Gera e salva um gráfico da evolução do valor de mercado de um ativo.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados do histórico.
        ativo (str): O nome do ativo para usar no título e nome do arquivo.
    """
    if 'date' not in df.columns or 'vlr_mercado' not in df.columns:
        print("DataFrame não contém as colunas 'date' e 'vlr_mercado' para gerar o gráfico.")
        return

    # Garante que a pasta de gráficos exista
    pasta_graficos = "dlombello/graficos"
    os.makedirs(pasta_graficos, exist_ok=True)

    # Prepara os dados para o gráfico
    df_grafico = df.copy()
    df_grafico['date'] = pd.to_datetime(df_grafico['date'])
    df_grafico = df_grafico.sort_values(by='date')

    # Cria o gráfico
    plt.figure(figsize=(12, 7))
    plt.plot(df_grafico['date'], df_grafico['vlr_mercado'], marker='o', linestyle='-', color='royalblue')
    
    # Customiza o gráfico
    plt.title(f'Evolução do Patrimônio - {ativo}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Valor de Mercado (R$)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.gcf().autofmt_xdate() # Melhora a visualização das datas

    # Salva o gráfico em um arquivo
    caminho_arquivo = os.path.join(pasta_graficos, f'evolucao_{ativo}.png')

    # Adiciona os rótulos de valor em cada ponto
    for index, row in df_grafico.iterrows():
        plt.text(row['date'], row['vlr_mercado'], f' R${row["vlr_mercado"]:.2f}', va='bottom', ha='left', fontsize=9)

    # Salva os dados em um arquivo CSV (formato para Excel brasileiro)
    caminho_csv = os.path.join(pasta_graficos, f'evolucao_{ativo}.csv')
    df_grafico[['date', 'vlr_mercado']].to_csv(caminho_csv, index=False, decimal=',', sep=';')
    print(f"Dados do gráfico de evolução salvos em: {caminho_csv}")

    plt.savefig(caminho_arquivo)
    print(f"\nGráfico salvo com sucesso em: {caminho_arquivo}")

def buscar_historico(token: str, ativo: str = None, classe: str = None, corretora: str = None) -> pd.DataFrame | None:    
    """
    Busca o histórico de investimentos na API e retorna como um DataFrame do pandas.

    Args:
        token (str): Token de autorização para a API.
        ativo (str, optional): Filtra por um ativo específico. Defaults to None.
        classe (str, optional): Filtra por uma classe de ativo (ex: 'R.FIXA', 'ACOES'). Defaults to None.
        corretora (str, optional): Filtra por corretora. Defaults to None.

    Returns:
        pd.DataFrame | None: Um DataFrame com os dados do histórico ou None em caso de erro.
    """
    url = "https://users.dlombelloplanilhas.com/historico"
    headers = {"Content-Type": "application/json", "Authorization": token}
    
    # Monta os parâmetros de consulta apenas com os valores que foram fornecidos
    params = {
        "ativo": ativo,
        "classe": classe,
        "corretora": corretora
        #"date_ini": date_ini,
        #"date_fim": date_fim
    }
    # Filtra para não enviar parâmetros vazios
    params = {k: v for k, v in params.items() if v is not None}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Lança um erro para status HTTP 4xx/5xx
        
        dados_json = response.json()
        
        # Converte a lista 'historico' do JSON em um DataFrame
        df = pd.DataFrame(dados_json.get("historico", []))
        
        if df.empty:
            print("Nenhum dado de histórico encontrado para os filtros aplicados.")
            return None
            
        return df

    except RequestException as e:
        print(f"Erro na requisição à API: {e}")
        return None
    except ValueError: # Erro de decodificação do JSON
        print("Erro ao processar a resposta da API. Não é um JSON válido.")
        return None

def buscar_dados_benchmark(ticker: str, start_date: str, end_date: str) -> pd.Series | None:
    """
    Busca dados históricos de fechamento para um ticker de benchmark.

    Args:
        ticker (str): O ticker do benchmark (ex: '^BVSP' para IBOV).
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de fim no formato 'YYYY-MM-DD'.

    Returns:
        pd.Series | None: Uma série com os preços de fechamento ou None em caso de erro.
    """
    try:
        print(f"Buscando dados para o benchmark: {ticker}...")
        dados = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if dados.empty:
            print(f"Nenhum dado encontrado para o benchmark {ticker} no período especificado.")
            return None
        return dados['Close']
    except Exception as e:
        print(f"Erro ao buscar dados do benchmark {ticker}: {e}")
        return None

def gerar_grafico_comparativo_twr(df_twr: pd.DataFrame, benchmarks_data: dict, ativo: str):
    """
    Gera um gráfico comparando o TWR da carteira com outros benchmarks.

    Args:
        df_twr (pd.DataFrame): DataFrame com a coluna 'twr_acc' e 'date'.
        benchmarks_data (dict): Dicionário onde a chave é o nome do benchmark (ex: 'IBOV')
                                e o valor é uma pd.Series com os dados de preço.
        ativo (str): Nome do ativo principal para o título do gráfico.
    """
    pasta_graficos = "dlombello/graficos"
    os.makedirs(pasta_graficos, exist_ok=True)

    plt.figure(figsize=(14, 8))

    # 1. Plotar o TWR da carteira (normalizado em base 100)
    # (twr_acc + 1) transforma o percentual de retorno em um fator de crescimento
    carteira_normalizada = (df_twr['twr_acc'] + 1) * 100
    plt.plot(df_twr['date'], carteira_normalizada, label=f'Carteira - {ativo}', color='red', linewidth=2.5)

    # 2. Plotar cada benchmark (normalizado em base 100)
    for nome, dados_benchmark in benchmarks_data.items():
        if dados_benchmark is not None and not dados_benchmark.empty:
            # Normaliza o benchmark para começar em 100
            benchmark_normalizado = (dados_benchmark / dados_benchmark.iloc[0]) * 100
            plt.plot(benchmark_normalizado.index, benchmark_normalizado, label=nome, linestyle='--')

    # 3. Customizar o gráfico
    plt.title(f'Comparativo de Rentabilidade: {ativo} vs. Benchmarks', fontsize=16)
    plt.ylabel('Performance (Base 100)', fontsize=12)
    plt.xlabel('Data', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    # 4. Salvar o gráfico
    caminho_arquivo = os.path.join(pasta_graficos, f'comparativo_twr_{ativo}.png')
    plt.savefig(caminho_arquivo)
    print(f"Gráfico comparativo de TWR salvo com sucesso em: {caminho_arquivo}")


def main():
    """
    Função principal que orquestra a execução do script.
    """
    # Busca o histórico do ativo 'KLBN11'
    df_historico = buscar_historico(token, ativo="KLBN11")

    if df_historico is not None:
        print("Dados capturados com sucesso! Exibindo as 5 primeiras linhas:")
        print(df_historico.head())
        gerar_grafico_evolucao(df_historico, ativo="KLBN11")
        
        # Calcula o TWR e obtém o dataframe com os resultados
        df_twr = gerar_grafico_twr(df_historico, ativo="KLBN11")

        if df_twr is not None:
            # Define os benchmarks para comparação
            benchmarks = {
                'IBOV': '^BVSP',
                'S&P 500': 'SPY'
            }
            
            # Define o período para a busca dos benchmarks
            start_date = df_twr['date'].min().strftime('%Y-%m-%d')
            end_date = df_twr['date'].max().strftime('%Y-%m-%d')

            # Busca os dados dos benchmarks
            benchmarks_data = {nome: buscar_dados_benchmark(ticker, start_date, end_date) for nome, ticker in benchmarks.items()}

            gerar_grafico_comparativo_twr(df_twr, benchmarks_data, ativo="KLBN11")

# Garante que a função main() só seja executada quando o script for rodado diretamente
if __name__ == "__main__":
    main()