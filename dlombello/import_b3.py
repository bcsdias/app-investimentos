from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.wait import WebDriverWait 
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os, requests, time, sys

# Define o diretório raiz do projeto 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Adiciona a raiz ao PYTHONPATH para permitir imports como "from utils.logger"
sys.path.append(BASE_DIR)
from utils.logger import setup_logger

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
    timeout_seconds = 30 # Aumentado para 30 segundos
    while 'Evolucao_Diaria.csv' not in os.listdir(download_folder):
        time.sleep(.1)
        if time.time() - start_time > timeout_seconds:
            logger.error(f"Tempo de espera excedido ({timeout_seconds}s). O arquivo 'Evolucao_Diaria.csv' não foi encontrado em '{download_folder}'.")
            raise TimeoutError("Download do arquivo demorou demais.")
        
    logger.info("Arquivo 'Evolucao_Diaria.csv' encontrado na pasta de downloads.")

    with open(f'{download_folder}/Evolucao_Diaria.csv', 'r', encoding='latin') as file:
        line = file.readline()
        index = line[:4]
        assert str(year) == line[-5:-1]

    # Define o caminho do arquivo de destino
    destination_file = f'{folder}/{index}-{year}.csv'
    # Se o arquivo de destino já existir, remove-o para evitar o FileExistsError
    if os.path.exists(destination_file):
        os.remove(destination_file)
        logger.warning(f"Arquivo de destino existente '{destination_file}' foi removido para ser substituído.")

    os.rename(f'{download_folder}/Evolucao_Diaria.csv', destination_file)
    logger.info(f"Arquivo renomeado e movido para: {destination_file}")

if __name__ == '__main__':
    logger = setup_logger(log_file='import_b3.log')
    logger.info("--- Iniciando script de download de dados da B3 ---")

    # Define os caminhos de forma dinâmica, baseados na localização do script
    project_folder = os.path.dirname(os.path.abspath(__file__))
    download_folder = os.path.join(project_folder, 'downloads')
    destination_folder = os.path.join(project_folder, 'dados')
    logger.debug(f"Pasta do projeto: {project_folder}")
    logger.debug(f"Pasta de downloads: {download_folder}")
    logger.debug(f"Pasta de destino: {destination_folder}")

    # Cria as pastas se elas não existirem
    os.makedirs(download_folder, exist_ok=True)
    os.makedirs(destination_folder, exist_ok=True)
    logger.info("Pastas de trabalho verificadas/criadas com sucesso.")

    # Configura o Chrome para usar a pasta de download personalizada
    chrome_options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_folder}
    chrome_options.add_experimental_option("prefs", prefs)

    driver = None
    try:
        logger.info("Inicializando driver do Chrome...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        index = 'IFIX'
        year = 2024
        
        download_b3_index_year(driver, index, year, download_folder, destination_folder, logger)
        logger.info("--- Script finalizado com sucesso! ---")

    except Exception as e:
        logger.exception("Ocorreu um erro durante a execução do script de download.")
    finally:
        if driver:
            driver.quit()
            logger.info("Driver do Chrome encerrado.")