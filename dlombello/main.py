import requests
import os
import sys
import argparse
from dotenv import load_dotenv
import pandas as pd
from requests.exceptions import RequestException
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import yfinance as yf
from bcb import sgs

# Define o diretório raiz do projeto 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Adiciona a raiz ao PYTHONPATH para permitir imports como "from utils.logger"
sys.path.append(BASE_DIR)
from utils.logger import setup_logger

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém o token da variável de ambiente
token = os.getenv('DLP_TOKEN')

if not token:
    msg = "A variável de ambiente 'DLP_TOKEN' não foi encontrada ou está vazia. Verifique seu arquivo .env."
    # O logger ainda não foi configurado, então usamos print e raise para um erro crítico.
    print(f"CRITICAL: {msg}")
    raise ValueError(msg)


def gerar_grafico_twr(df: pd.DataFrame, nome_grafico: str, logger) -> pd.DataFrame | None:
    """
    Calcula o Time-Weighted Return (TWR) e gera um gráfico da sua evolução.

    Args:
        df (pd.DataFrame): DataFrame com o histórico, contendo 'date', 'vlr_mercado' e 'vlr_investido'.
        nome_grafico (str): Nome para o título e arquivo do gráfico (ex: 'KLBN11' ou 'Ações').
    Returns:
        pd.DataFrame | None: DataFrame com os cálculos do TWR ou None se falhar.
    """
    colunas_necessarias = ['date', 'vlr_mercado', 'vlr_investido', 'proventos']
    if not all(col in df.columns for col in colunas_necessarias):
        logger.error(f"DataFrame não contém as colunas necessárias ({', '.join(colunas_necessarias)}) para calcular o TWR.")
        return None

    # Garante que a pasta de gráficos exista
    pasta_graficos = "dlombello/graficos"
    os.makedirs(pasta_graficos, exist_ok=True)

    # 1. Preparar e consolidar os dados por data
    logger.info("Iniciando cálculo de TWR...")
    # Se a análise for de uma classe de ativos, haverá múltiplas linhas por data (uma para cada ativo).
    # Precisamos agrupar por data e somar os valores para ter a visão consolidada da carteira.
    df_consolidado = df.copy()
    df_consolidado['date'] = pd.to_datetime(df_consolidado['date'])
    
    colunas_para_somar = ['vlr_mercado', 'vlr_investido', 'proventos']
    df_twr = df_consolidado.groupby('date')[colunas_para_somar].sum().reset_index()

    # Ordena por data para garantir a sequência cronológica correta
    df_twr = df_twr.sort_values(by='date').reset_index(drop=True)
    logger.debug(f"DataFrame consolidado para TWR:\n{df_twr.to_string()}")

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
    
    # Calcula o lucro/prejuízo do mês para usar no cálculo do HPR em caso de venda total.
    df_twr['lucro_mes'] = df_twr['vlr_mercado'] - df_twr['valor_inicial_periodo'] - df_twr['fluxo_mes']

    # Calcula o HPR (Holding Period Return)
    # O HPR é o fator de multiplicação (ex: 1.05 para 5% de ganho).
    df_twr['hpr'] = 0.0 # Inicializa a coluna

    # Caso 1: Cálculo normal, onde o denominador não é zero.
    mask_normal = denominador != 0
    df_twr.loc[mask_normal, 'hpr'] = df_twr.loc[mask_normal, 'vlr_mercado'] / denominador[mask_normal]

    # Caso 2: Venda total do ativo (vlr_mercado vai a zero, mas valor_inicial não era).
    # O HPR reflete o retorno até o momento da venda.
    mask_venda_total = (df_twr['vlr_mercado'] == 0) & (df_twr['valor_inicial_periodo'] != 0)
    df_twr.loc[mask_venda_total, 'hpr'] = (df_twr.loc[mask_venda_total, 'lucro_mes'] - df_twr.loc[mask_venda_total, 'fluxo_mes']) / df_twr.loc[mask_venda_total, 'valor_inicial_periodo']

    # Caso 3: Posição estava e continua zerada. O HPR deve ser 1.
    # Isso "congela" o TWR acumulado até que um novo aporte seja feito.
    mask_zerado = (df_twr['valor_inicial_periodo'] == 0) & (df_twr['vlr_mercado'] == 0)
    df_twr.loc[mask_zerado, 'hpr'] = 1.0

    # 5. Calcular o TWR acumulado
    # twr_mes é o retorno do período (HPR - 1)
    df_twr['twr_mes'] = df_twr['hpr'] - 1
    # O .cumprod() multiplica acumuladamente os valores da série.
    df_twr['twr_acc'] = (df_twr['hpr'].cumprod() - 1)
    logger.debug(f"DataFrame final com cálculo de TWR:\n{df_twr.to_string()}")

    # 6. Gerar o gráfico
    plt.figure(figsize=(12, 7))
    # Multiplicamos por 100 apenas para a exibição no gráfico
    plt.plot(df_twr['date'], df_twr['twr_acc'] * 100, marker='o', linestyle='-', color='darkorange', label=nome_grafico)
    plt.title(f'Rentabilidade (TWR Acumulado) - {nome_grafico}', fontsize=16)
    plt.ylabel('Rentabilidade Acumulada (%)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    caminho_arquivo = os.path.join(pasta_graficos, f'evolucao_twr_{nome_grafico}.png')

    # Adiciona os rótulos de valor em cada ponto
    for index, row in df_twr.iterrows():
        plt.text(row['date'], row['twr_acc'] * 100, f' {row["twr_acc"]*100:.2f}%', va='bottom', ha='left', fontsize=9)

    # Salva os dados em um arquivo CSV (formato para Excel brasileiro)
    caminho_csv = os.path.join(pasta_graficos, f'evolucao_twr_{nome_grafico}.csv')
    
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

    logger.info(f"Dados do gráfico de TWR salvos em: {caminho_csv}")

    plt.savefig(caminho_arquivo)
    logger.info(f"Gráfico de TWR salvo com sucesso em: {caminho_arquivo}")
    
    return df_twr

def gerar_grafico_percentual(df: pd.DataFrame, nome_grafico: str, logger):
    """
    Gera e salva um gráfico da evolução percentual (lucro/prejuízo) de um ativo.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados do histórico.
        nome_grafico (str): O nome para usar no título e nome do arquivo.
    """
    colunas_necessarias = ['date', 'vlr_mercado', 'vlr_investido']
    if not all(col in df.columns for col in colunas_necessarias):
        logger.error(f"DataFrame não contém as colunas necessárias ({', '.join(colunas_necessarias)}) para gerar o gráfico percentual.")
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
    plt.plot(df_grafico['date'], df_grafico['evolucao_%'], marker='o', linestyle='-', color='seagreen', label=nome_grafico)
    
    # Customiza o gráfico
    plt.title(f'Evolução Percentual (Lucro/Prejuízo) - {nome_grafico}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Lucro / Prejuízo (%)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--') # Linha do 0%
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter()) # Formata o eixo Y como porcentagem
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    # Salva o gráfico em um arquivo
    caminho_arquivo = os.path.join(pasta_graficos, f'evolucao_percentual_{nome_grafico}.png')

    # Adiciona os rótulos de valor em cada ponto
    for index, row in df_grafico.iterrows():
        plt.text(row['date'], row['evolucao_%'], f' {row["evolucao_%"]:.2f}%', va='bottom', ha='left', fontsize=9)

    # Salva os dados em um arquivo CSV (formato para Excel brasileiro)
    caminho_csv = os.path.join(pasta_graficos, f'evolucao_percentual_{nome_grafico}.csv')
    df_grafico[['date', 'evolucao_%']].to_csv(caminho_csv, index=False, decimal=',', sep=';')
    logger.info(f"Dados do gráfico percentual salvos em: {caminho_csv}")

    plt.savefig(caminho_arquivo)
    logger.info(f"Gráfico percentual salvo com sucesso em: {caminho_arquivo}")

def gerar_grafico_evolucao(df: pd.DataFrame, nome_grafico: str, logger):
    """
    Gera e salva um gráfico da evolução do valor de mercado de um ativo.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados do histórico.
        nome_grafico (str): O nome para usar no título e nome do arquivo.
    """
    if 'date' not in df.columns or 'vlr_mercado' not in df.columns:
        logger.error("DataFrame não contém as colunas 'date' e 'vlr_mercado' para gerar o gráfico.")
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
    plt.plot(df_grafico['date'], df_grafico['vlr_mercado'], marker='o', linestyle='-', color='royalblue', label=nome_grafico)
    
    # Customiza o gráfico
    plt.title(f'Evolução do Patrimônio - {nome_grafico}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Valor de Mercado (R$)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.gcf().autofmt_xdate() # Melhora a visualização das datas

    # Salva o gráfico em um arquivo
    caminho_arquivo = os.path.join(pasta_graficos, f'evolucao_{nome_grafico}.png')

    # Adiciona os rótulos de valor em cada ponto
    for index, row in df_grafico.iterrows():
        plt.text(row['date'], row['vlr_mercado'], f' R${row["vlr_mercado"]:.2f}', va='bottom', ha='left', fontsize=9)

    # Salva os dados em um arquivo CSV (formato para Excel brasileiro)
    caminho_csv = os.path.join(pasta_graficos, f'evolucao_{nome_grafico}.csv')
    df_grafico[['date', 'vlr_mercado']].to_csv(caminho_csv, index=False, decimal=',', sep=';')
    logger.info(f"Dados do gráfico de evolução salvos em: {caminho_csv}")

    plt.savefig(caminho_arquivo)
    logger.info(f"Gráfico de evolução salvo com sucesso em: {caminho_arquivo}")

def buscar_historico(token: str, logger, ativo: str = None, classe: str = None, corretora: str = None) -> pd.DataFrame | None:    
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

    logger.info(f"Buscando histórico na API com os parâmetros: {params}")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Lança um erro para status HTTP 4xx/5xx
        
        dados_json = response.json()
        
        # Converte a lista 'historico' do JSON em um DataFrame
        df = pd.DataFrame(dados_json.get("historico", []))
        
        if df.empty:
            logger.warning("Nenhum dado de histórico encontrado para os filtros aplicados.")
            return None
            
        return df

    except RequestException as e:
        logger.error(f"Erro na requisição à API: {e}")
        return None
    except ValueError: # Erro de decodificação do JSON
        logger.error("Erro ao processar a resposta da API. Não é um JSON válido.")
        return None

def buscar_dados_benchmark(ticker: str, start_date: str, end_date: str, logger) -> pd.Series | None:
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
        logger.info(f"Buscando dados para o benchmark: {ticker}...")
        dados = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if dados.empty:
            logger.warning(f"Nenhum dado encontrado para o benchmark {ticker} no período especificado.")
            return None
        return dados['Close']
    except Exception as e:
        logger.error(f"Erro ao buscar dados do benchmark {ticker}: {e}")
        return None

def buscar_dados_bcb(codigo_bcb: int, start_date: str, end_date: str, logger) -> pd.Series | None:
    """
    Busca uma série temporal do Banco Central do Brasil (BCB) e calcula o retorno acumulado.

    Args:
        codigo_bcb (int): O código da série no sistema SGS do BCB.
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de fim no formato 'YYYY-MM-DD'.

    Returns:
        pd.Series | None: Uma série com o índice de retorno acumulado ou None em caso de erro.
    """
    try:
        logger.info(f"Buscando dados da série {codigo_bcb} do BCB...")
        # O nome da coluna será o próprio código
        df = sgs.get({str(codigo_bcb): codigo_bcb}, start=start_date, end=end_date)
        if df.empty:
            logger.warning(f"Nenhum dado encontrado para a série {codigo_bcb} do BCB no período.")
            return None

        # A API retorna a taxa em % ao dia ou ao mês.
        # Precisamos converter para um fator de retorno e calcular o produto acumulado.
        # Ex: 0.05% -> 1.0005
        retorno_fator = (df[str(codigo_bcb)] / 100) + 1

        # Calcula o retorno acumulado (índice)
        retorno_acumulado = retorno_fator.cumprod()

        # A primeira data pode ter um valor NaN se não houver dados anteriores, preenche com 1
        retorno_acumulado = retorno_acumulado.fillna(1)

        # Renomeia o índice para 'Date' para consistência
        retorno_acumulado.index.name = 'Date'

        return retorno_acumulado
    except Exception as e:
        logger.error(f"Erro ao buscar dados da série {codigo_bcb} do BCB: {e}")
        return None

def gerar_grafico_comparativo_twr(df_twr: pd.DataFrame, benchmarks_data: dict, nome_grafico: str, logger):
    """
    Gera um gráfico comparando o TWR da carteira com outros benchmarks.

    Args:
        df_twr (pd.DataFrame): DataFrame com a coluna 'twr_acc' e 'date'.
        benchmarks_data (dict): Dicionário onde a chave é o nome do benchmark (ex: 'IBOV')
                                e o valor é uma pd.Series com os dados de preço.
        nome_grafico (str): Nome do ativo principal para o título do gráfico.
    """
    pasta_graficos = "dlombello/graficos"
    os.makedirs(pasta_graficos, exist_ok=True)

    plt.figure(figsize=(14, 8))

    # 1. Plotar o TWR da carteira (normalizado em base 100)
    # (twr_acc + 1) transforma o percentual de retorno em um fator de crescimento
    carteira_normalizada = (df_twr['twr_acc'] + 1) * 100
    plt.plot(df_twr['date'], carteira_normalizada, label=f'Carteira - {nome_grafico}', color='red', linewidth=2.5)

    # 2. Plotar cada benchmark (normalizado em base 100)
    for nome, dados_benchmark in benchmarks_data.items():
        logger.debug(f"Processando benchmark '{nome}' para o gráfico comparativo.")
        if dados_benchmark is not None and not dados_benchmark.empty:
            # Normaliza o benchmark para começar em 100
            benchmark_normalizado = (dados_benchmark / dados_benchmark.iloc[0]) * 100
            logger.debug(f"Dados para '{nome}' encontrados. Normalizando e plotando.")
            plt.plot(benchmark_normalizado.index, benchmark_normalizado, label=nome, linestyle='--')

            # Adiciona rótulos nos pontos mensais correspondentes ao df_twr
            # Usamos reindex para alinhar as datas diárias do benchmark com as datas mensais da carteira
            logger.debug(f"Alinhando benchmark '{nome}' com as datas da carteira: {df_twr['date'].to_list()}")
            benchmark_mensal = benchmark_normalizado.reindex(df_twr['date'], method='ffill')
            logger.debug(f"Valores mensais para '{nome}' após alinhamento:\n{benchmark_mensal.to_string()}")

            # Se reindex com datas duplicadas criou um DataFrame, converte de volta para Series.
            # A coluna terá o nome da Series original ou '0' se não tiver nome.
            if isinstance(benchmark_mensal, pd.DataFrame):
                # Pega a primeira (e única) coluna do DataFrame resultante.
                benchmark_mensal = benchmark_mensal.iloc[:, 0]

            # Itera sobre a Series para adicionar os rótulos.
            # Usamos .items() que é o método moderno para iterar sobre (índice, valor).
            for data, valor_ponto in benchmark_mensal.items():
                # A verificação pd.notna() agora funciona corretamente com um valor escalar.
                if pd.notna(valor_ponto):
                    # O valor é a performance base 100. Para exibir o ganho/perda, subtraímos 100.
                    plt.text(data, valor_ponto, f' {valor_ponto-100:.1f}%', va='top', ha='center', fontsize=8, alpha=0.7)
        else:
            logger.warning(f"Nenhum dado válido para o benchmark '{nome}'. Não será plotado.")

    # Adiciona os rótulos para a carteira por último para que fiquem por cima
    for index, row in df_twr.iterrows():
        valor_normalizado = (row['twr_acc'] + 1) * 100
        plt.text(row['date'], valor_normalizado, f' {valor_normalizado-100:.1f}%', va='bottom', ha='center', fontsize=8, color='red', weight='bold')


    # 3. Customizar o gráfico
    plt.title(f'Comparativo de Rentabilidade: {nome_grafico} vs. Benchmarks', fontsize=16)
    plt.ylabel('Performance (Base 100)', fontsize=12)
    plt.xlabel('Data', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    # 4. Salvar o gráfico
    caminho_arquivo = os.path.join(pasta_graficos, f'comparativo_twr_{nome_grafico}.png')
    plt.savefig(caminho_arquivo)
    logger.info(f"Gráfico comparativo de TWR salvo com sucesso em: {caminho_arquivo}")


def main():
    """
    Função principal que orquestra a execução do script.
    """
    parser = argparse.ArgumentParser(description="Gera análises de carteira de investimentos a partir da API.")
    parser.add_argument('--debug', action='store_true', help='Ativa o modo de log detalhado (debug).')
    
    # Grupo de argumentos mutuamente exclusivos: ou --ativo ou --classe deve ser fornecido.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ativo', type=str, help='Código do ativo a ser analisado (ex: KLBN11).')
    group.add_argument('--classe', type=str, help='Classe de ativos a ser analisada (ex: AÇÃO).')

    args = parser.parse_args()

    logger = setup_logger(debug=args.debug, log_file='main.log')
    if args.debug:
        logger.info("Modo de depuração ativado.")

    df_historico = None
    nome_analise = ""

    if args.ativo:
        nome_analise = args.ativo
        df_historico = buscar_historico(token, logger, ativo=args.ativo)
    elif args.classe:
        nome_analise = args.classe
        df_historico = buscar_historico(token, logger, classe=args.classe)

    if df_historico is not None:
        logger.info("Dados capturados com sucesso!")
        logger.debug(f"Cabeçalho do DataFrame de histórico:\n{df_historico.head().to_string()}")

        # Salva o histórico completo em um arquivo .csv (formato para Excel brasileiro)
        pasta_saida = "dlombello/graficos"
        os.makedirs(pasta_saida, exist_ok=True)
        caminho_csv = os.path.join(pasta_saida, f'historico_completo_{nome_analise}.csv')
        df_historico.to_csv(caminho_csv, sep=';', decimal=',', index=False)
        logger.info(f"Histórico completo salvo em: {caminho_csv}")

        logger.info("Gerando gráficos...")
        gerar_grafico_evolucao(df_historico, nome_grafico=nome_analise, logger=logger)
        
        # Calcula o TWR e obtém o dataframe com os resultados
        df_twr = gerar_grafico_twr(df_historico, nome_grafico=nome_analise, logger=logger)

        if df_twr is not None:
            # Define os benchmarks para comparação
            benchmarks_yf = {
                'IBOV': '^BVSP',
                'S&P 500': 'SPY', # ETF que replica o S&P 500
                'IMID': 'IMID.L' # SPDR MSCI All Country World Investable Market UCITS ETF
            }
            # Códigos das séries no SGS do Banco Central
            # 11: Taxa SELIC diária
            # 433: IPCA mensal
            benchmarks_bcb = {
                'SELIC': 11,
                'IPCA': 433
            }
            
            # Define o período para a busca dos benchmarks
            start_date = df_twr['date'].min().strftime('%Y-%m-%d')
            end_date = df_twr['date'].max().strftime('%Y-%m-%d')
            logger.info(f"Período da análise: {start_date} a {end_date}")

            benchmarks_data = {}

            # 1. Busca os dados dos benchmarks do Yahoo Finance
            for nome, ticker in benchmarks_yf.items():
                benchmarks_data[nome] = buscar_dados_benchmark(ticker, start_date, end_date, logger)

            # 2. Busca os dados dos benchmarks do Banco Central
            for nome, codigo in benchmarks_bcb.items():
                benchmarks_data[nome] = buscar_dados_bcb(codigo, start_date, end_date, logger)

            # 3. Calcula o benchmark sintético "IPCA + 6%"
            if 'IPCA' in benchmarks_data and benchmarks_data['IPCA'] is not None:
                logger.info("Calculando benchmark sintético 'IPCA + 6%'...")
                # Converte a taxa anual de 6% para uma taxa mensal equivalente
                taxa_real_mensal = (1.06 ** (1/12)) - 1
                
                # Cria um fator de crescimento mensal constante
                fator_crescimento_real = pd.Series(taxa_real_mensal + 1, index=benchmarks_data['IPCA'].index).cumprod()
                
                # Multiplica o índice IPCA pelo fator de crescimento real para obter o IPCA + 6%
                ipca_mais_6 = benchmarks_data['IPCA'] * fator_crescimento_real
                benchmarks_data['IPCA + 6%'] = ipca_mais_6

            gerar_grafico_comparativo_twr(df_twr, benchmarks_data, nome_grafico=nome_analise, logger=logger)

    else:
        logger.error(f"Não foi possível obter o histórico para a análise '{nome_analise}'. Encerrando o script.")

# Garante que a função main() só seja executada quando o script for rodado diretamente
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Captura qualquer exceção não tratada para garantir que seja logada.
        setup_logger().exception("Ocorreu um erro fatal não capturado na execução principal.")
        raise e