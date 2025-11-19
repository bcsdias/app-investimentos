import requests
import os
import sys
import argparse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from requests.exceptions import RequestException
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import yfinance as yf
from bcb import sgs
from adjustText import adjust_text

# Define o diretório raiz do projeto 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Adiciona a raiz ao PYTHONPATH para permitir imports como "from utils.logger"
sys.path.append(BASE_DIR)
from utils.logger import setup_logger
# Importa a função que acabamos de refatorar em import_b3.py
from dlombello.import_b3 import run_b3_downloader


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

    # Adiciona uma linha separadora com quebras de linha para melhor visualização no log
    logger.info(f"\n\n==================== NOVA ANÁLISE ====================")
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

def buscar_dados_b3(indice: str, start_date: str, end_date: str, logger) -> pd.Series | None:
    """
    Orquestra o download e o processamento de dados de índices da B3.
    """
    logger.info(f"Iniciando processo de obtenção de dados da B3 para o índice '{indice}'.")
    
    # 1. Determinar os anos necessários para o download
    ano_inicio = pd.to_datetime(start_date).year
    ano_fim = pd.to_datetime(end_date).year
    anos_necessarios = list(range(ano_inicio, ano_fim + 1))
    logger.debug(f"Anos necessários para a análise de '{indice}': {anos_necessarios}")
    
    # 2. Verificar quais arquivos já existem e quais precisam ser baixados
    pasta_dados = os.path.join(os.path.dirname(__file__), 'dados')
    anos_para_baixar = []
    for ano in anos_necessarios:
        caminho_arquivo = os.path.join(pasta_dados, f'{indice}-{ano}.csv')
        if not os.path.exists(caminho_arquivo):
            anos_para_baixar.append(ano)
    
    # 3. Chamar o downloader apenas se houver arquivos faltando
    if anos_para_baixar:
        logger.info(f"Arquivos não encontrados para os anos: {anos_para_baixar}. Iniciando download...")
        indices_para_baixar = {indice: anos_para_baixar}
        run_b3_downloader(indices_para_baixar, logger)
    else:
        logger.info(f"Todos os arquivos necessários para o índice '{indice}' já existem localmente. Download pulado.")

    
    # 4. Ler, processar e consolidar todos os arquivos CSV necessários (existentes + baixados)
    dados_completos = pd.DataFrame()

    mapa_meses = {
        'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Mai': 5, 'Jun': 6,
        'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12
    }

    for ano in anos_necessarios:
        caminho_arquivo = os.path.join(pasta_dados, f'{indice}-{ano}.csv')
        if not os.path.exists(caminho_arquivo):
            logger.warning(f"Arquivo '{caminho_arquivo}' não encontrado após tentativa de download. Pulando ano {ano}.")
            continue
        
        logger.debug(f"Processando arquivo: {caminho_arquivo}")
        # Lê o CSV, pulando a primeira linha de título e tratando o formato brasileiro
        df_ano = pd.read_csv(caminho_arquivo, sep=';', decimal=',', skiprows=1, encoding='latin-1')
        
        # Remove linhas de rodapé como 'MÍNIMO'
        df_ano = df_ano[pd.to_numeric(df_ano['Dia'], errors='coerce').notna()]
        df_ano['Dia'] = df_ano['Dia'].astype(int)

        # Unpivot: Transforma a tabela de meses em colunas para linhas de data e valor
        df_unpivoted = df_ano.melt(id_vars=['Dia'], var_name='Mes', value_name='Close')
        df_unpivoted.dropna(subset=['Close'], inplace=True)

        # **CORREÇÃO APLICADA AQUI**
        # A coluna 'Close' é lida como string (ex: '3.320,47').
        # 1. Removemos o separador de milhar ('.').
        # 2. Substituímos o separador decimal (',') por um ponto ('.').
        # 3. Convertemos para float.
        df_unpivoted['Close'] = df_unpivoted['Close'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

        # Monta a data completa
        df_unpivoted['Mes_Num'] = df_unpivoted['Mes'].map(mapa_meses)
        df_unpivoted['Ano'] = ano
        df_unpivoted['Date'] = pd.to_datetime(df_unpivoted[['Ano', 'Mes_Num', 'Dia']].rename(columns={'Ano': 'year', 'Mes_Num': 'month', 'Dia': 'day'}))
        
        dados_completos = pd.concat([dados_completos, df_unpivoted[['Date', 'Close']]])

    if dados_completos.empty:
        logger.error(f"Nenhum dado pôde ser processado para o índice '{indice}' da B3.")
        return None

    # Finaliza o processamento
    dados_completos = dados_completos.sort_values('Date').set_index('Date')
    
    # Filtra novamente pelo período exato e retorna a série
    serie_final = dados_completos.loc[start_date:end_date, 'Close']
    logger.info(f"Dados do índice '{indice}' da B3 processados com sucesso.")
    return serie_final

def calcular_rentabilidades_resumo(df_twr: pd.DataFrame, benchmarks_data: dict, nome_carteira: str, logger) -> pd.DataFrame | None:
    """
    Calcula a rentabilidade anual e total para a carteira e benchmarks.

    Returns:
        pd.DataFrame: DataFrame formatado com as rentabilidades para a tabela.
    """
    logger.debug("Iniciando cálculo de rentabilidades para a tabela de resumo.")
    resumo_data = {}

    # 1. Processar a carteira
    df_carteira = df_twr.set_index('date')['twr_acc']
    
    # Adiciona 1 para ter o fator de crescimento
    fator_carteira = df_carteira + 1
    
    # Pega o valor no final de cada ano
    anual_carteira = fator_carteira.resample('A').last()
    # Calcula o retorno anual
    retornos_anuais_carteira = anual_carteira.pct_change().fillna(anual_carteira.iloc[0] - 1)
    logger.debug(f"Tipo de dados para 'Carteira': {type(retornos_anuais_carteira)}")
    resumo_data[f'Carteira - {nome_carteira}'] = retornos_anuais_carteira

    # 2. Processar cada benchmark
    def _ensure_series(dados):
        """Garante que `dados` seja uma pd.Series com valores numéricos: seleciona
        a primeira coluna numérica de um DataFrame ou converte iteráveis para Series."""
        if dados is None or getattr(dados, 'empty', False):
            return None
        if isinstance(dados, pd.DataFrame):
            if dados.shape[1] == 1:
                s = dados.iloc[:, 0]
            else:
                numeric = dados.select_dtypes(include=[np.number])
                if not numeric.empty:
                    s = numeric.iloc[:, 0]
                else:
                    s = pd.to_numeric(dados.iloc[:, 0], errors='coerce')
                    s = pd.Series(s.values, index=dados.index)
        elif isinstance(dados, pd.Series):
            s = dados
        else:
            try:
                s = pd.Series(dados)
            except Exception:
                return None

        # Converte para float e remove NaNs intermediários
        s = pd.to_numeric(s, errors='coerce')
        s = s.dropna()
        if s.empty:
            return None
        return s

    for nome, dados in benchmarks_data.items():
        s = _ensure_series(dados)
        if s is None:
            logger.debug(f"Benchmark '{nome}' vazio ou inválido para cálculo anual.")
            continue

        # Normaliza o benchmark como fator a partir do primeiro valor (base 1)
        fator_bench = s / s.iloc[0]
        anual_bench = fator_bench.resample('A').last()
        # Retorno anual = pct_change; para o primeiro ano, usa o fator final do ano menos 1
        if not anual_bench.empty:
            retornos_anuais_bench = anual_bench.pct_change().fillna(anual_bench.iloc[0] - 1)
        else:
            retornos_anuais_bench = anual_bench.pct_change()

        # Se for DataFrame resultante (caso raro), reduz para Series
        if isinstance(retornos_anuais_bench, pd.DataFrame):
            retornos_anuais_bench = retornos_anuais_bench.iloc[:, 0]

        logger.debug(f"Tipo de dados para o benchmark '{nome}': {type(retornos_anuais_bench) if 'retornos_anuis_bench' in locals() else type(retornos_anuais_bench)}")
        resumo_data[nome] = retornos_anuais_bench if 'retornos_anuis_bench' in locals() else retornos_anuais_bench

    if not resumo_data:
        return None

    # 3. Montar o DataFrame final
    df_resumo = pd.DataFrame(resumo_data).T # Transpõe para ter os anos como colunas
    
    # 4. Calcular a rentabilidade total acumulada
    rentabilidades_totais = {}
    # Calcula para a carteira (garante float)
    try:
        rentabilidades_totais[f'Carteira - {nome_carteira}'] = float(fator_carteira.iloc[-1] - 1)
    except Exception:
        rentabilidades_totais[f'Carteira - {nome_carteira}'] = np.nan

    # Calcula para cada benchmark usando _ensure_series
    for nome, dados in benchmarks_data.items():
        s = _ensure_series(dados)
        if s is None:
            rentabilidades_totais[nome] = np.nan
        else:
            try:
                rentabilidades_totais[nome] = float((s.iloc[-1] / s.iloc[0]) - 1)
            except Exception:
                rentabilidades_totais[nome] = np.nan

    logger.debug(f"Rentabilidades totais calculadas: {rentabilidades_totais}")

    # Adiciona a coluna 'Total' ao DataFrame mapeando os valores pelo índice
    df_resumo['Total'] = df_resumo.index.map(rentabilidades_totais)
    
    # 5. Formatar o DataFrame para exibição
    # Renomeia as colunas para apenas o ano
    df_resumo.columns = [col.year if isinstance(col, pd.Timestamp) else str(col) for col in df_resumo.columns]
    
    # Agrupa por nomes de coluna para remover duplicatas, pegando o primeiro valor
    df_resumo = df_resumo.groupby(level=0, axis=1).first()

    # Garante que todas as colunas de dados sejam numéricas antes de formatar
    for col in df_resumo.columns:
        df_resumo[col] = pd.to_numeric(df_resumo[col], errors='coerce')
    # Calcula rentabilidade acumulada por linha (apenas para colunas de anos)
    # Identifica colunas de ano (exclui 'Total' se presente)
    cols = [c for c in df_resumo.columns if str(c).lower() != 'total']

    # Tenta ordenar as colunas por ano (converte para int quando possível)
    def _col_key(c):
        try:
            return int(c)
        except Exception:
            # Mantém a ordem original se não for conversível
            return float('inf')

    cols_sorted = sorted(cols, key=_col_key)

    # Reordena o DataFrame (anos em ordem crescente, depois 'Total' se existir)
    ordered_cols = cols_sorted + (['Total'] if 'Total' in df_resumo.columns else [])
    df_resumo = df_resumo.reindex(columns=ordered_cols)

    # Calcula acumulado: (1 + r).cumprod() - 1 ao longo das colunas de anos
    if cols_sorted:
        acumulado_df = (1 + df_resumo[cols_sorted]).cumprod(axis=1) - 1
    else:
        acumulado_df = pd.DataFrame(index=df_resumo.index)

    # Função de formatação: 'anual (acumulado)'
    def _format_cell(annual, acc):
        if pd.isna(annual):
            return '-'
        try:
            if pd.isna(acc):
                return f'{float(annual):.1%}'
            return f'{float(annual):.1%} ({float(acc):.1%})'
        except Exception:
            return str(annual)

    # Monta um DataFrame de strings com a mesma estrutura
    df_formatted = pd.DataFrame(index=df_resumo.index, columns=df_resumo.columns, dtype=object)

    # Preenche colunas de ano com 'anual (acumulado)'
    for c in cols_sorted:
        for idx in df_resumo.index:
            annual = df_resumo.at[idx, c]
            acc = acumulado_df.at[idx, c] if (idx in acumulado_df.index and c in acumulado_df.columns) else np.nan
            df_formatted.at[idx, c] = _format_cell(annual, acc)

    # Formata a coluna 'Total' como porcentagem simples
    if 'Total' in df_resumo.columns:
        for idx in df_resumo.index:
            total_val = df_resumo.at[idx, 'Total']
            df_formatted.at[idx, 'Total'] = f'{float(total_val):.1%}' if pd.notna(total_val) else '-'

    df_resumo = df_formatted
    
    return df_resumo


def gerar_grafico_comparativo_twr(df_twr: pd.DataFrame, benchmarks_data: dict, nome_grafico: str, logger):
    """
    Gera um gráfico comparando o TWR da carteira com outros benchmarks.
                logger.debug(f"Tipo de dados para o benchmark '{nome}': {type(retornos_anuais_bench)}")
    Args:
        df_twr (pd.DataFrame): DataFrame com a coluna 'twr_acc' e 'date'.
        benchmarks_data (dict): Dicionário onde a chave é o nome do benchmark (ex: 'IBOV')
                                e o valor é uma pd.Series com os dados de preço.
        nome_grafico (str): Nome do ativo principal para o título do gráfico.
    """
    pasta_graficos = "dlombello/graficos"
    os.makedirs(pasta_graficos, exist_ok=True)

    # --- VARIÁVEL DE CONTROLE ---
    # Define a cada quantos meses um rótulo de performance será exibido no gráfico.
    # Use 1 para exibir todos, 6 para exibir a cada semestre, 12 para anual, etc.
    intervalo_meses_rotulo = 6

    fig, ax = plt.subplots(figsize=(15, 10))

    # Lista para armazenar todos os objetos de texto que serão ajustados
    text_labels = []

    # Mapa para guardar a cor utilizada em cada linha (benchmark)
    color_map = {}

    # 1. Plotar o TWR da carteira (normalizado em base 100)
    # (twr_acc + 1) transforma o percentual de retorno em um fator de crescimento
    carteira_normalizada = (df_twr['twr_acc'] + 1) * 100
    linha_carteira, = ax.plot(df_twr['date'], carteira_normalizada, label=f'Carteira - {nome_grafico}', color='red', linewidth=2.5)
    color_map[f'Carteira - {nome_grafico}'] = linha_carteira.get_color()

    # 2. Plotar cada benchmark (normalizado em base 100)
    for nome, dados_benchmark in benchmarks_data.items():
        logger.debug(f"Processando benchmark '{nome}' para o gráfico comparativo.")
        if dados_benchmark is not None and not dados_benchmark.empty:
            # Normaliza o benchmark para começar em 100
            benchmark_normalizado = (dados_benchmark / dados_benchmark.iloc[0]) * 100
            logger.debug(f"Dados para '{nome}' encontrados. Normalizando e plotando.")
            linha_bench, = ax.plot(benchmark_normalizado.index, benchmark_normalizado, label=nome, linestyle='--')
            color_map[nome] = linha_bench.get_color()

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
            # Adicionamos enumerate para ter um contador e plotar a cada 6 meses.
            for i, (data, valor_ponto) in enumerate(benchmark_mensal.items()):
                # A verificação pd.notna() agora funciona corretamente com um valor escalar.
                # Adiciona o rótulo apenas para o primeiro ponto (i==0) e a cada 6 meses.
                if pd.notna(valor_ponto) and (i % intervalo_meses_rotulo == 0 or i == 0):
                    # Cria o objeto de texto e o adiciona à lista para ajuste posterior
                    label = ax.text(data, valor_ponto, f' {valor_ponto-100:.1f}%', va='top', ha='center', fontsize=8, alpha=0.7)
                    text_labels.append(label)
        else:
            logger.warning(f"Nenhum dado válido para o benchmark '{nome}'. Não será plotado.")

    # Adiciona os rótulos para a carteira a cada 6 meses
    for index, row in df_twr.iterrows():
        # O índice do DataFrame (0, 1, 2...) nos serve como contador.
        # Adiciona o rótulo apenas para o primeiro ponto (index==0) e a cada 6 meses.
        if index % intervalo_meses_rotulo == 0 or index == 0:
            valor_normalizado = (row['twr_acc'] + 1) * 100
            # Cria o objeto de texto e o adiciona à lista
            label = ax.text(row['date'], valor_normalizado, f' {valor_normalizado-100:.1f}%', va='bottom', ha='center', fontsize=8, color='red', weight='bold')
            text_labels.append(label)


    # 3. Customizar o gráfico
    ax.set_title(f'Comparativo de Rentabilidade: {nome_grafico} vs. Benchmarks', fontsize=16)
    ax.set_ylabel('Performance (Base 100)', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.autofmt_xdate()

    # 4. Ajustar a posição dos rótulos para evitar sobreposição
    # A função adjust_text irá reposicionar os textos da lista 'text_labels'
    adjust_text(text_labels, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # 5. Adicionar a tabela de resumo de rentabilidade
    df_resumo = calcular_rentabilidades_resumo(df_twr, benchmarks_data, nome_grafico, logger)
    if df_resumo is not None:
        # Prepara os dados para a função table
        cell_text = df_resumo.values
        row_labels = list(df_resumo.index)
        col_labels = list(df_resumo.columns)

        # Ajusta os rótulos das colunas para indicar que entre parênteses está o acumulado
        col_labels_display = [f"{c} (acc)" if str(c).lower() != 'total' else 'Total' for c in col_labels]

        # Adiciona a tabela na parte de baixo do gráfico
        tabela = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels_display,
                  loc='bottom', cellLoc='center', bbox=[0, -0.4, 1, 0.3])
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(10)
        tabela.scale(1, 1.5) # Ajusta a altura das células

        # Colorir a célula do rótulo da linha com a cor da linha do gráfico, quando disponível
        try:
            celld = tabela.get_celld()
            for (r, c), cell in celld.items():
                # Em matplotlib o índice de coluna das rowLabels costuma ser -1
                if c == -1 and 0 <= r < len(row_labels):
                    nome_linha = row_labels[r]
                    cor = color_map.get(nome_linha)
                    if cor:
                        cell.get_text().set_color(cor)
                        cell.get_text().set_weight('bold')
        except Exception:
            logger.debug('Não foi possível colorir os rótulos das linhas na tabela.')

        # Ajusta o layout para dar espaço para a tabela
        fig.subplots_adjust(bottom=0.3)

    # 6. Salvar o gráfico
    caminho_arquivo = os.path.join(pasta_graficos, f'comparativo_twr_{nome_grafico}.png')
    plt.savefig(caminho_arquivo, bbox_inches='tight')
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
                'IBOV (YFinance)': '^BVSP',
                'S&P 500': 'SPY', # ETF que replica o S&P 500
                'IMID': 'IMID.L' # SPDR MSCI All Country World Investable Market UCITS ETF
            }
            # Benchmarks que serão baixados e processados da B3
            benchmarks_b3 = {
                'IFIX': 'IFIX'
                #'IBOV (B3)': 'IBOV'
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
            
            # 2. Busca e processa os dados dos benchmarks da B3
            for nome, indice in benchmarks_b3.items():
                benchmarks_data[nome] = buscar_dados_b3(indice, start_date, end_date, logger)

            # 3. Busca os dados dos benchmarks do Banco Central
            for nome, codigo in benchmarks_bcb.items():
                benchmarks_data[nome] = buscar_dados_bcb(codigo, start_date, end_date, logger)

            # 4. Calcula o benchmark sintético "IPCA + 6%"
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