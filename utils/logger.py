import logging
import os
from datetime import datetime

def setup_logger(hostname=None, level=None, debug=False, tipo=None, vendor=None):
    """
    Configura e retorna um logger configurado.
    O nível de log é determinado pelo parâmetro 'debug', a menos que 'level' seja especificado manualmente.
    
    Args:
        hostname (str): Nome do CMTS para criar arquivo de log específico.
        tipo (str): Tipo do equipamento (ex: 'cmts', 'olt') para criar subdiretório.
        vendor (str): Vendor do equipamento (ex: 'cisco', 'huawei') para criar subdiretório.
        debug (bool): Se True, define o nível de log para DEBUG.
        level (int): Nível de log para override manual (ex: logging.DEBUG).
    """
    logger_name = hostname if hostname else 'main'
    logger = logging.getLogger(logger_name)

    # Se o logger já estiver configurado (tem handlers), apenas o retorna.
    # Isso evita a duplicação de handlers em cenários de multithreading e
    # torna a função segura para ser chamada múltiplas vezes.
    if logger.hasHandlers():
        return logger

    # A configuração abaixo só será executada na primeira vez que o logger for solicitado.
    if level is None: # Se um nível de log não for passado manualmente, determina-o automaticamente
        if debug:
            level = logging.DEBUG
            # Apenas para feedback no console ao iniciar, não irá para o arquivo de log a menos que o logger principal já esteja configurado
            print(f"INFO: MODO DEBUG ATIVADO. Nível de log definido para DEBUG.")
        else:
            level = logging.INFO

    # Obter o diretório raiz do projeto (um nível acima do utils)
    root_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Define o diretório base para os logs
    base_log_dir = os.path.join(root_dir, 'log')

    if hostname:
        # Se tipo e vendor forem fornecidos, cria a estrutura de subpastas
        if tipo and vendor:
            # Garante que tipo e vendor estejam em minúsculas para os nomes das pastas
            log_dir = os.path.join(base_log_dir, tipo.lower(), vendor.lower())
        else:
            log_dir = base_log_dir
        
        data_atual = datetime.now().strftime('%d%m%Y')
        log_filename = f"{hostname} - {data_atual}.log"
    else:
        # Para o log principal, usa o diretório base
        log_dir = base_log_dir
        log_filename = "main.log"
    
    # Cria o diretório de log (incluindo subpastas) se não existir e define o caminho completo
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    # Impede propagação para o logger raiz (evita duplicação no console)
    logger.propagate = False

    # Habilita o log de depuração do Paramiko para um arquivo separado apenas se o modo debug estiver ativo
    # if debug:
    #     paramiko_log_path = os.path.join(base_log_dir, 'paramiko.log')
    #     paramiko_logger = logging.getLogger("paramiko")
    #     paramiko_logger.setLevel(logging.DEBUG)
    #     paramiko_logger.addHandler(logging.FileHandler(paramiko_log_path))
    #     logger.info("Log de depuração do Paramiko está ATIVADO (paramiko.log).")
    # Configura o nível de log (agora dinâmico)
    logger.setLevel(level)
    
    # Criar formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configurar o handler de arquivo
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Configurar handler de console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger