import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
from requests.exceptions import RequestException
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.wait import WebDriverWait 
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Define o diretório raiz do projeto (assumindo que utils está um nível abaixo da raiz)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DOWNLOADS_DIR = os.path.join(BASE_DIR, "data", "downloads")

def _ensure_series(x) -> pd.Series | None:
    """Garante que x seja uma pd.Series 1D com valores numéricos quando possível.
    Retorna None se não for possível converter.
    """
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors='coerce')
    if isinstance(x, pd.DataFrame):
        # prioriza colunas numéricas
        numeric = x.select_dtypes(include=[np.number])
        if not numeric.empty:
            return numeric.iloc[:, 0]
        # fallback: pega a primeira coluna e converte
        return pd.to_numeric(x.iloc[:, 0], errors='coerce')
    try:
        return pd.to_numeric(pd.Series(x), errors='coerce')
    except Exception:
        return None

def buscar_historico(token: str, logger, ativo: str = None, classe: str = None, corretora: str = None) -> pd.DataFrame | None:    
    """
    Busca o histórico de investimentos na API e retorna como um DataFrame do pandas.
    """
    url = "https://users.dlombelloplanilhas.com/historico"
    headers = {"Content-Type": "application/json", "Authorization": token}
    
    # Monta os parâmetros de consulta apenas com os valores que foram fornecidos
    params = {
        "ativo": ativo,
        "classe": classe,
        "corretora": corretora
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
        logger.debug(f"Traceback YFinance {ticker}:", exc_info=True)
        return None

def buscar_dados_tesouro(titulo_nome: str, vencimento_str: str, start_date: str, end_date: str, logger) -> pd.Series | None:
    """
    Busca dados históricos de títulos do Tesouro Direto via Tesouro Transparente.
    Faz download do CSV oficial e filtra pelo título e vencimento.
    """
    # Garante que o diretório de dados brutos exista antes de tentar salvar o arquivo
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    arquivo_csv = os.path.join(DATA_RAW_DIR, "PrecoTaxaTesouroDireto.csv")
    url_tesouro = "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"

    # Verifica se precisa baixar (se não existe ou se é mais antigo que 24h)
    precisa_baixar = True
    if os.path.exists(arquivo_csv):
        tempo_arquivo = os.path.getmtime(arquivo_csv)
        if (time.time() - tempo_arquivo) < 86400: # 24 horas
            precisa_baixar = False
            logger.info("Usando cache local dos dados do Tesouro Direto.")

    if precisa_baixar:
        logger.info("Baixando dados históricos do Tesouro Direto (pode demorar um pouco)...")
        try:
            response = requests.get(url_tesouro, stream=True)
            response.raise_for_status()
            with open(arquivo_csv, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download do Tesouro Direto concluído.")
        except Exception as e:
            logger.error(f"Erro ao baixar dados do Tesouro Direto: {e}")
            logger.debug("Traceback download Tesouro:", exc_info=True)
            if not os.path.exists(arquivo_csv):
                return None

    try:
        logger.info(f"Processando dados do Tesouro: {titulo_nome} {vencimento_str}...")
        # Lê apenas colunas necessárias para otimizar memória
        df = pd.read_csv(
            arquivo_csv, 
            sep=';', 
            decimal=',', 
            encoding='latin-1',
            usecols=['Tipo Titulo', 'Data Vencimento', 'Data Base', 'PU Base Manha']
        )
        
        # Filtra pelo título e vencimento
        df['Data Vencimento'] = pd.to_datetime(df['Data Vencimento'], dayfirst=True)
        df['Data Base'] = pd.to_datetime(df['Data Base'], dayfirst=True)
        
        vencimento_dt = pd.to_datetime(vencimento_str, dayfirst=True)
        
        mask = (df['Tipo Titulo'] == titulo_nome) & (df['Data Vencimento'] == vencimento_dt)
        df_filtrado = df[mask].set_index('Data Base').sort_index()
        
        # Retorna a série no período solicitado (PU Base Manha é o preço de referência)
        return df_filtrado.loc[start_date:end_date, 'PU Base Manha']
        
    except Exception as e:
        logger.error(f"Erro ao processar dados do Tesouro Direto: {e}")
        logger.debug("Traceback processamento Tesouro:", exc_info=True)
        return None

def buscar_dados_bcb(codigo_bcb: int, start_date: str, end_date: str, logger) -> pd.Series | None:
    """
    Busca uma série temporal do Banco Central do Brasil (BCB) e calcula o retorno acumulado.
    """
    try:
        logger.info(f"Buscando dados da série {codigo_bcb} do BCB...")
        logger.debug(f"Chamada sgs.get: codigo={codigo_bcb}, start={start_date}, end={end_date}")
        
        # O nome da coluna será o próprio código
        df = sgs.get({str(codigo_bcb): codigo_bcb}, start=start_date, end=end_date)
        
        if df is None:
            logger.warning(f"A função sgs.get retornou None para a série {codigo_bcb}.")
            return None
            
        if df.empty:
            logger.warning(f"Nenhum dado encontrado para a série {codigo_bcb} do BCB no período.")
            return None
            
        logger.debug(f"Dados brutos recebidos do BCB (head):\n{df.head().to_string()}")

        # Lista de códigos que retornam Número Índice (já acumulado) e não Taxa %
        # IMA-B (12466, 12467, 12468), IRF-M (12461, 12463, 12464), IMA-S (12469), IMA-Geral (12462)
        codigos_indices = [12466, 12467, 12468, 12461, 12463, 12464, 12469, 12462]

        if codigo_bcb in codigos_indices:
            # Se já é índice, usa o valor direto
            retorno_acumulado = df[str(codigo_bcb)]
            # Preenche eventuais falhas com o valor anterior
            retorno_acumulado = retorno_acumulado.ffill()
        else:
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
        logger.debug("Traceback detalhado do erro BCB:", exc_info=True)
        return None

def download_b3_index_year(driver, index, year, download_folder, folder, logger):
    logger.info(f"Iniciando download para o índice '{index}', ano {year}.")
    driver.get(f'https://sistemaswebb3-listados.b3.com.br/indexStatisticsPage/daily-evolution/{index}?language=pt-br')
    logger.debug(f"Navegou para a URL do índice {index}.")

    element = WebDriverWait(driver, 30).until(lambda x: x.find_element(By.ID, 'selectYear'))
    year_select = Select(element)
    year_select.select_by_value(str(year))
    logger.debug(f"Ano {year} selecionado no dropdown.")

    element = WebDriverWait(driver, 30).until(lambda x: x.find_element(By.XPATH, '//a[text()="Download (ano selecionado)"]'))
    element.click()
    logger.info("Botão de download clicado. Aguardando o arquivo...")
    
    start_time = time.time()
    timeout_seconds = 30 
    while 'Evolucao_Diaria.csv' not in os.listdir(download_folder):
        time.sleep(.1)
        if time.time() - start_time > timeout_seconds:
            logger.error(f"Tempo de espera excedido ({timeout_seconds}s). O arquivo 'Evolucao_Diaria.csv' não foi encontrado em '{download_folder}'.")
            raise TimeoutError("Download do arquivo demorou demais.")
        
    logger.info("Arquivo 'Evolucao_Diaria.csv' encontrado na pasta de downloads.")

    with open(f'{download_folder}/Evolucao_Diaria.csv', 'r', encoding='latin') as file:
        line = file.readline()
        # Validação simples do conteúdo
        # index_file = line[:4] 
        # assert str(year) == line[-5:-1]

    # Define o caminho do arquivo de destino
    destination_file = f'{folder}/{index}-{year}.csv'
    # Se o arquivo de destino já existir, remove-o para evitar o FileExistsError
    if os.path.exists(destination_file):
        os.remove(destination_file)
        logger.warning(f"Arquivo de destino existente '{destination_file}' foi removido para ser substituído.")

    os.rename(f'{download_folder}/Evolucao_Diaria.csv', destination_file)
    logger.info(f"Arquivo renomeado e movido para: {destination_file}")

def run_b3_downloader(indices_anos: dict, logger):
    """
    Orquestra o download de múltiplos arquivos de índices da B3 para diferentes anos.
    """
    logger.info("Iniciando orquestrador de downloads da B3.")
    
    download_folder = DATA_DOWNLOADS_DIR
    destination_folder = DATA_RAW_DIR

    # Garante que as pastas existam
    os.makedirs(download_folder, exist_ok=True)
    os.makedirs(destination_folder, exist_ok=True)

    # Configura o Chrome para usar a pasta de download personalizada
    chrome_options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_folder}
    chrome_options.add_experimental_option("prefs", prefs)

    driver = None
    try:
        logger.info("Inicializando driver do Chrome...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        for index, years in indices_anos.items():
            for year in years:
                try:
                    download_b3_index_year(driver, index, year, download_folder, destination_folder, logger)
                except Exception as e:
                    logger.error(f"Falha ao baixar dados para o índice '{index}' ano {year}. Erro: {e}")

        logger.info("--- Processo de download da B3 finalizado. ---")

    except Exception as e:
        logger.exception("Ocorreu um erro durante a execução do script de download.")
    finally:
        if driver:
            driver.quit()
            logger.info("Driver do Chrome encerrado.")

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
    pasta_dados = DATA_RAW_DIR
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

def buscar_dolar_bcb(start_date: str, end_date: str, logger) -> pd.Series | None:
    """
    Busca a cotação do dólar (PTAX venda) na API do Banco Central (Olinda).
    Retorna uma série com as cotações diárias.
    """
    try:
        # A API espera datas no formato 'MM-DD-YYYY'
        dt_ini = pd.to_datetime(start_date).strftime('%m-%d-%Y')
        dt_fim = pd.to_datetime(end_date).strftime('%m-%d-%Y')
        
        url = f"https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?@dataInicial='{dt_ini}'&@dataFinalCotacao='{dt_fim}'&$top=10000&$format=json&$select=cotacaoVenda,dataHoraCotacao"
        
        logger.info(f"Buscando cotação do dólar na API do BCB (Olinda)...")
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if 'value' not in data or not data['value']:
            logger.warning("Nenhum dado de dólar encontrado no período.")
            return None
            
        df = pd.DataFrame(data['value'])
        if df.empty:
             logger.warning("DataFrame de dólar vazio.")
             return None

        df['dataHoraCotacao'] = pd.to_datetime(df['dataHoraCotacao'])
        df['Date'] = df['dataHoraCotacao'].dt.normalize() # Remove hora
        # Ordena por data/hora e remove duplicatas de dia mantendo a última (evita erro no reindex)
        df = df.sort_values('dataHoraCotacao')
        df = df.drop_duplicates(subset=['Date'], keep='last')
        df = df.set_index('Date').sort_index()
        
        return df['cotacaoVenda']
        
    except Exception as e:
        logger.error(f"Erro ao buscar cotação do dólar: {e}")
        logger.debug("Traceback Dolar BCB:", exc_info=True)
        return None

def processar_benchmarks(start_date: str, end_date: str, benchmarks_yf: dict, benchmarks_b3: dict, benchmarks_bcb: dict, benchmarks_td: dict, carteiras_config: dict, logger) -> dict:
    """
    Centraliza a busca e cálculo de benchmarks e índices sintéticos.
    """
    benchmarks_data = {}

    # 1. Busca YF
    for nome, ticker in benchmarks_yf.items():
        benchmarks_data[nome] = buscar_dados_benchmark(ticker, start_date, end_date, logger)

    # 2. Busca B3
    for nome, indice in benchmarks_b3.items():
        benchmarks_data[nome] = buscar_dados_b3(indice, start_date, end_date, logger)

    # 3. Busca BCB
    for nome, codigo in benchmarks_bcb.items():
        benchmarks_data[nome] = buscar_dados_bcb(codigo, start_date, end_date, logger)

    # 4. Busca Tesouro Direto
    for nome, config in benchmarks_td.items():
        benchmarks_data[nome] = buscar_dados_tesouro(config['titulo'], config['vencimento'], start_date, end_date, logger)

    # 4. Dolar & Conversões para BRL
    dolar_ptax = buscar_dolar_bcb(start_date, end_date, logger)
    if dolar_ptax is not None:
        # IMID BRL
        if 'IMID' in benchmarks_data and benchmarks_data['IMID'] is not None:
            imid_series = _ensure_series(benchmarks_data['IMID'])
            if imid_series is not None:
                logger.info("Calculando benchmark 'IMID BRL'...")
                dolar_aligned = dolar_ptax.reindex(imid_series.index, method='ffill')
                benchmarks_data['IMID BRL'] = imid_series * dolar_aligned

        # S&P 500 BRL
        if 'S&P 500' in benchmarks_data and benchmarks_data['S&P 500'] is not None:
            spy_series = _ensure_series(benchmarks_data['S&P 500'])
            if spy_series is not None:
                logger.info("Calculando benchmark 'S&P 500 BRL'...")
                dolar_aligned = dolar_ptax.reindex(spy_series.index, method='ffill')
                benchmarks_data['S&P 500 BRL'] = spy_series * dolar_aligned

        # Bitcoin BRL
        if 'Bitcoin' in benchmarks_data and benchmarks_data['Bitcoin'] is not None:
            btc_series = _ensure_series(benchmarks_data['Bitcoin'])
            if btc_series is not None:
                logger.info("Calculando benchmark 'Bitcoin BRL'...")
                dolar_aligned = dolar_ptax.reindex(btc_series.index, method='ffill')
                benchmarks_data['Bitcoin BRL'] = btc_series * dolar_aligned

    # 5. IPCA + 6%
    if 'IPCA' in benchmarks_data and benchmarks_data['IPCA'] is not None:
        logger.info("Calculando benchmark sintético 'IPCA + 6%'...")
        taxa_real_mensal = (1.06 ** (1/12)) - 1
        fator_crescimento_real = pd.Series(taxa_real_mensal + 1, index=benchmarks_data['IPCA'].index).cumprod()
        benchmarks_data['IPCA + 6%'] = benchmarks_data['IPCA'] * fator_crescimento_real

    # Helper interno para cálculos de portfólio sintético
    def _calc_portfolio(assets_weights: dict, name: str):
        series_list = []
        valid = True
        for nome, peso in assets_weights.items():
            if nome not in benchmarks_data or benchmarks_data[nome] is None:
                valid = False
                break
            s = _ensure_series(benchmarks_data[nome])
            if s is None:
                valid = False
                break
            # Calcula retorno ponderado
            series_list.append(s.pct_change().fillna(0) * peso)
        
        if valid:
            logger.info(f"Calculando benchmark sintético '{name}'...")
            combined = pd.concat(series_list, axis=1).fillna(0)
            portfolio_ret = combined.sum(axis=1)
            benchmarks_data[name] = (1 + portfolio_ret).cumprod()

    # 6. Sintéticos Compostos (Carteiras Teóricas) - Gerados dinamicamente via config
    if carteiras_config:
        for nome_carteira, pesos in carteiras_config.items():
            _calc_portfolio(pesos, nome_carteira)

    return benchmarks_data
