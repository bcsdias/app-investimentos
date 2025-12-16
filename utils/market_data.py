import os
import time
import requests
import re
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

# Configurações de Cache
CACHE_EXPIRY = 86400  # 24 horas em segundos
CACHE_START_DATE = '1995-01-01'  # Data inicial padrão para popular o cache com histórico longo

def _get_cache_path(filename):
    return os.path.join(DATA_RAW_DIR, filename)

def _is_cache_valid(filepath):
    if os.path.exists(filepath):
        if (time.time() - os.path.getmtime(filepath)) < CACHE_EXPIRY:
            return True
    return False

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
        # Sanitiza o ticker para nome de arquivo
        safe_ticker = ticker.replace('^', '').replace('.', '_')
        cache_file = _get_cache_path(f"YF_{safe_ticker}.csv")
        
        df = None
        if _is_cache_valid(cache_file):
            logger.info(f"Usando cache local para benchmark YF: {ticker}")
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
                # Se o CSV tiver cabeçalhos extras do yfinance (Price, Ticker), o índice pode ficar sujo.
                # Tenta limpar convertendo o índice para datetime e removendo o que falhar.
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df.dropna(how='all') # Remove linhas onde o índice virou NaT
                # Garante índice único e ordenado
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
            except Exception:
                pass # Se falhar ao ler, baixa novamente
        
        if df is None:
            logger.info(f"Baixando dados YF para: {ticker} (Histórico Completo)...")
            # Baixa histórico longo para popular o cache
            dados = yf.download(ticker, start=CACHE_START_DATE, progress=False, auto_adjust=True)
            if not dados.empty:
                # Garante índice único e ordenado antes de salvar
                dados = dados[~dados.index.duplicated(keep='last')]
                dados = dados.sort_index()
                df = dados[['Close']]
                df.to_csv(cache_file)
            else:
                logger.warning(f"Nenhum dado encontrado para o benchmark {ticker}.")
                return None

        # Filtra pelo período solicitado
        # Garante que o índice é datetime para o slice funcionar
        df.index = pd.to_datetime(df.index)
        # Ordena novamente para garantir slice correto
        df = df.sort_index()
        # Slice seguro usando máscara booleana
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df.loc[mask, 'Close'].dropna()

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

    # --- CACHE NÍVEL 2: Série Processada ---
    # Verifica se já temos o arquivo processado deste título específico (muito mais rápido)
    safe_title = titulo_nome.replace(' ', '_').replace('+', 'mais')
    safe_venc = vencimento_str.replace('/', '-')
    cache_series_file = _get_cache_path(f"TD_{safe_title}_{safe_venc}.csv")

    if _is_cache_valid(cache_series_file):
        logger.info(f"Usando cache processado para Tesouro: {titulo_nome} {vencimento_str}")
        try:
            s = pd.read_csv(cache_series_file, index_col=0, parse_dates=True, sep=';', decimal=',', date_format='%Y-%m-%d %H:%M:%S')
            # Garante índice único e ordenado
            s = s[~s.index.duplicated(keep='last')]
            s = s.sort_index()
            if not s.empty:
                return s.iloc[:, 0].loc[start_date:end_date]
        except Exception:
            pass

    # Verifica se precisa baixar (se não existe ou se é mais antigo que 24h)
    precisa_baixar = not _is_cache_valid(arquivo_csv)
    if not precisa_baixar:
        logger.info("Usando cache local do arquivo mestre do Tesouro Direto.")

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
        
        # Remove duplicatas se houver
        df_filtrado = df_filtrado[~df_filtrado.index.duplicated(keep='last')]

        # Salva o cache processado para a próxima vez
        full_series = df_filtrado['PU Base Manha']
        full_series.to_csv(cache_series_file, sep=';', decimal=',')

        # Retorna a série no período solicitado (PU Base Manha é o preço de referência)
        return full_series.loc[start_date:end_date]
        
    except Exception as e:
        logger.error(f"Erro ao processar dados do Tesouro Direto: {e}")
        logger.debug("Traceback processamento Tesouro:", exc_info=True)
        return None

def buscar_dados_bcb(codigo_bcb: int, start_date: str, end_date: str, logger) -> pd.Series | None:
    """
    Busca uma série temporal do Banco Central do Brasil (BCB) e calcula o retorno acumulado.
    """
    try:
        cache_file = _get_cache_path(f"BCB_{codigo_bcb}.csv")
        df = None

        # 1. Tenta carregar o cache existente (independente da validade temporal do arquivo)
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
                df = df[~df.index.duplicated(keep='last')].sort_index()
            except Exception:
                logger.warning(f"Arquivo de cache {cache_file} corrompido. Será baixado novamente.")
                df = None

        # 2. Verifica a necessidade de atualização (Incremental)
        today = pd.Timestamp.today().normalize()
        update_needed = False
        start_download = pd.to_datetime(CACHE_START_DATE)

        if df is not None and not df.empty:
            last_date = df.index[-1]
            # Se a última data for anterior a (hoje - 5 dias), tenta atualizar.
            if last_date < (today - pd.Timedelta(days=5)):
                update_needed = True
                start_download = last_date + pd.Timedelta(days=1)
                logger.info(f"Cache BCB {codigo_bcb} desatualizado (último dado: {last_date.date()}). Buscando atualizações...")
            else:
                logger.info(f"Usando cache local para série BCB: {codigo_bcb} (atualizado até {last_date.date()})")
        else:
            update_needed = True
            logger.info(f"Buscando dados da série {codigo_bcb} do BCB (Histórico Completo)...")
            df = pd.DataFrame()

        # 3. Realiza o download incremental se necessário
        if update_needed:
            dfs_new = []
            current_start = start_download
            
            while current_start < today:
                current_end = current_start + pd.DateOffset(years=5)
                if current_end > today:
                    current_end = today
                
                logger.debug(f"Buscando BCB {codigo_bcb} de {current_start.date()} a {current_end.date()}")
                try:
                    chunk_df = sgs.get({str(codigo_bcb): codigo_bcb}, start=current_start, end=current_end)
                    if chunk_df is not None and not chunk_df.empty:
                        chunk_df.index = pd.to_datetime(chunk_df.index)
                        dfs_new.append(chunk_df)
                except Exception as e:
                    logger.debug(f"Falha ao buscar chunk {current_start.date()}-{current_end.date()} para BCB {codigo_bcb}: {e}")

                current_start = current_end + pd.DateOffset(days=1)
            
            if dfs_new:
                df_new = pd.concat(dfs_new)
                df = pd.concat([df, df_new])
                df = df[~df.index.duplicated(keep='last')]
                df = df.sort_index()
                df.to_csv(cache_file)
                logger.info(f"Cache BCB {codigo_bcb} atualizado com sucesso.")
            elif df.empty:
                logger.warning(f"Nenhum dado encontrado para a série {codigo_bcb} do BCB.")
                return None
            else:
                # Se não achou dados novos mas já tinha antigos, apenas atualiza mtime do arquivo
                if os.path.exists(cache_file):
                    os.utime(cache_file, None)

            # --- FALLBACK: Extensão via Proxy (ETFs) para índices descontinuados no BCB ---
            # A ANBIMA parou de fornecer dados públicos ao BCB em meados de 2023.
            # Usamos ETFs equivalentes para projetar a variação do índice no período faltante.
            if df is not None and not df.empty:
                last_date = df.index[-1]
                if last_date < (today - pd.Timedelta(days=5)):
                    BCB_PROXIES = {
                        12466: 'IMAB11.SA',  # IMA-B
                        12461: 'IRFM11.SA',  # IRF-M
                        12467: 'B5P211.SA',  # IMA-B 5
                        12468: 'IB5M11.SA',  # IMA-B 5+ (Usando IB5M11 como proxy aproximado)
                        12469: 'LFTS11.SA',  # IMA-S
                    }
                    proxy_ticker = BCB_PROXIES.get(codigo_bcb)
                    
                    if proxy_ticker:
                        logger.info(f"Série BCB {codigo_bcb} estagnada em {last_date.date()}. Tentando estender com proxy {proxy_ticker} via YFinance...")
                        try:
                            # Baixa dados do proxy a partir da última data válida
                            proxy_data = yf.download(proxy_ticker, start=last_date, progress=False, auto_adjust=True)
                            
                            if not proxy_data.empty and 'Close' in proxy_data:
                                # Trata colunas MultiIndex se existirem (comum em versões novas do yfinance)
                                if isinstance(proxy_data.columns, pd.MultiIndex):
                                    proxy_close = proxy_data['Close'].iloc[:, 0]
                                else:
                                    proxy_close = proxy_data['Close']
                                
                                proxy_close = proxy_close.sort_index()
                                
                                # O primeiro valor do proxy é a base (data de corte). Projetamos a variação a partir dele.
                                if not proxy_close.empty:
                                    base_price = proxy_close.iloc[0]
                                    last_bcb_value = df.iloc[-1, 0]
                                    
                                    # Projeção: Valor_Indice = Valor_BCB_Ultimo * (Preco_ETF / Preco_ETF_Base)
                                    # Removemos o primeiro dia (base) para não duplicar no concat
                                    extended_values = last_bcb_value * (proxy_close.iloc[1:] / base_price)
                                    
                                    df_extension = pd.DataFrame(extended_values)
                                    df_extension.columns = df.columns
                                    
                                    df = pd.concat([df, df_extension])
                                    df = df[~df.index.duplicated(keep='last')].sort_index()
                                    
                                    # Salva o cache estendido
                                    df.to_csv(cache_file)
                                    logger.info(f"Série BCB {codigo_bcb} estendida via proxy até {df.index[-1].date()}.")
                        except Exception as e:
                            logger.warning(f"Falha ao estender série {codigo_bcb} com proxy {proxy_ticker}: {e}")

        # Filtra pelo período solicitado antes de processar (para o cálculo de acumulado bater com o período)
        df = df.loc[start_date:end_date]
        if df.empty:
             return None
        
        # Garante ordenação e unicidade para o processamento subsequente
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

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
            # --- INTELIGÊNCIA: Verificar anos disponíveis (Dropdown + Texto) ---
            years_to_download = set(years)
            
            try:
                url = f'https://sistemaswebb3-listados.b3.com.br/indexStatisticsPage/daily-evolution/{index}?language=pt-br'
                logger.info(f"Acessando página do índice '{index}' para verificar histórico disponível...")
                driver.get(url)
                
                wait = WebDriverWait(driver, 30)
                
                # 1. Validação pelo Dropdown (O que é tecnicamente possível baixar)
                try:
                    select_element = wait.until(lambda x: x.find_element(By.ID, 'selectYear'))
                    year_select = Select(select_element)
                    dropdown_years = set()
                    for option in year_select.options:
                        val = option.get_attribute('value')
                        if val and val.isdigit():
                            dropdown_years.add(int(val))
                    
                    # Interseção: Só tentamos baixar o que está no dropdown
                    years_to_download = years_to_download.intersection(dropdown_years)
                except Exception as e:
                    logger.warning(f"Não foi possível ler o dropdown de anos para '{index}': {e}")

                # 2. Validação pelo Texto (O que realmente tem dados)
                try:
                    # Busca no texto visível da página inteira para ser mais robusto
                    body_element = driver.find_element(By.TAG_NAME, "body")
                    text_content = body_element.text
                    
                    # Regex para capturar o ano após "desde" (com ou sem mês)
                    match = re.search(r'desde\s+(?:[A-Za-zç]+\s+de\s+)?(\d{4})', text_content, re.IGNORECASE)
                    
                    if match:
                        start_year = int(match.group(1))
                        logger.info(f"Ano de início identificado via texto para '{index}': {start_year}")
                        years_to_download = {y for y in years_to_download if y >= start_year}
                    else:
                        logger.warning(f"Texto 'desde' não encontrado para refinar ano inicial.")
                        
                except Exception as e:
                    logger.warning(f"Erro ao tentar validar ano pelo texto para '{index}': {e}")

                # Log do que foi filtrado
                final_years = sorted(list(years_to_download))
                skipped = sorted(list(set(years) - set(final_years)))
                if skipped:
                    logger.warning(f"Anos ignorados para '{index}' (indisponíveis ou fora do período): {skipped}")

            except Exception as e:
                logger.error(f"Erro crítico ao verificar disponibilidade para '{index}': {e}")
                final_years = years # Fallback

            for year in final_years:
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
        
        # Verifica se o arquivo está vazio antes de ler
        if os.path.getsize(caminho_arquivo) == 0:
            logger.warning(f"Arquivo '{caminho_arquivo}' está vazio. Removendo e pulando ano {ano}.")
            try:
                os.remove(caminho_arquivo)
            except OSError:
                pass
            continue

        try:
            # Lê o CSV, pulando a primeira linha de título e tratando o formato brasileiro
            df_ano = pd.read_csv(caminho_arquivo, sep=';', decimal=',', skiprows=1, encoding='latin-1')
        except pd.errors.EmptyDataError:
            logger.warning(f"Arquivo '{caminho_arquivo}' não contém dados válidos (EmptyDataError). Removendo e pulando ano {ano}.")
            try:
                os.remove(caminho_arquivo)
            except OSError:
                pass
            continue
        except Exception as e:
            logger.error(f"Erro ao ler arquivo '{caminho_arquivo}': {e}")
            continue
        
        # Remove linhas de rodapé como 'MÍNIMO'
        df_ano = df_ano[pd.to_numeric(df_ano['Dia'], errors='coerce').notna()]
        df_ano['Dia'] = df_ano['Dia'].astype(int)

        # Unpivot: Transforma a tabela de meses em colunas para linhas de data e valor
        df_unpivoted = df_ano.melt(id_vars=['Dia'], var_name='Mes', value_name='Close')
        df_unpivoted.dropna(subset=['Close'], inplace=True)

        # Verifica se a coluna é do tipo object (string) antes de tentar limpar
        if df_unpivoted['Close'].dtype == 'object':
            # A coluna 'Close' é lida como string (ex: '3.320,47').
            # 1. Removemos o separador de milhar ('.').
            # 2. Substituímos o separador decimal (',') por um ponto ('.').
            # 3. Convertemos para float.
            df_unpivoted['Close'] = df_unpivoted['Close'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        else:
            # Se já for numérico, garante que é float
            df_unpivoted['Close'] = df_unpivoted['Close'].astype(float)

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
