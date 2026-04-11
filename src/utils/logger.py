import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# --- Configuração de Caminhos ---
# Assume que este arquivo está em src/utils/logger.py
# O root do projeto é 2 níveis acima
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT_DIR, 'log')

# --- Detecção de Modo Debug ---
# Checa se '--debug' ou '-d' foi passado no terminal (através do Streamlit ou direto)
DEBUG_MODE = "--debug" in sys.argv or "-d" in sys.argv

def setup_logger(name="invest_app"):
    """
    Configura um logger único (Singleton) com rotação de arquivos
    e detecção automática de nível baseado nos argumentos de linha de comando.
    """
    logger = logging.getLogger(name)
    
    # Se o logger já tiver handlers, significa que já foi configurado (evita duplicidade)
    if logger.handlers:
        return logger

    # Nível de Log
    level = logging.DEBUG if DEBUG_MODE else logging.INFO
    logger.setLevel(level)
    
    # Garante que a pasta de logs existe
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'web_app.log')
    
    # Formato do Log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    
    # --- Handler: Rotação de Arquivo (10MB, 5 backups) ---
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=10 * 1024 * 1024, # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # --- Handler: Console ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Impede que os logs sejam propagados para o logger raiz (evita logs duplicados no Streamlit)
    logger.propagate = False
    
    if DEBUG_MODE:
        logger.debug("MODO DEBUG ATIVADO via flag de linha de comando.")
        
    return logger

# Expondo a instância global para fácil importação nas outras classes
logger = setup_logger()

def get_logger():
    """Retorna a instância global configurada."""
    return logger