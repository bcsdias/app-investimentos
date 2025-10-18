import requests
import os
from dotenv import load_dotenv
import pandas as pd
from requests.exceptions import RequestException

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém o token da variável de ambiente
token = os.getenv('DLP_TOKEN')

if not token:
    raise ValueError("A variável de ambiente 'DLP_TOKEN' não foi encontrada ou está vazia. Verifique seu arquivo .env.")

#def buscar_historico(token: str, ativo: str = None, classe: str = None, corretora: str = None, date_ini: str = None, date_fim: str = None) -> pd.DataFrame | None:
def buscar_historico(token: str, ativo: str = None, classe: str = None, corretora: str = None) -> pd.DataFrame | None:    
    """
    Busca o histórico de investimentos na API e retorna como um DataFrame do pandas.

    Args:
        token (str): Token de autorização para a API.
        ativo (str, optional): Filtra por um ativo específico. Defaults to None.
        classe (str, optional): Filtra por uma classe de ativo (ex: 'R.FIXA', 'ACOES'). Defaults to None.
        corretora (str, optional): Filtra por corretora. Defaults to None.
        date_ini (str, optional): Data de início no formato 'YYYY-MM-DD'. Defaults to None.
        date_fim (str, optional): Data de fim no formato 'YYYY-MM-DD'. Defaults to None.

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

# --- Exemplo de uso da função ---
# Busca o histórico do ativo 'TD:IPCA2045' para o ano de 2023
#df_historico = buscar_historico(token, ativo="TD:IPCA2045", date_ini="2023-01-01", date_fim="2023-12-31")
df_historico = buscar_historico(token, ativo="TD:IPCA2045")

if df_historico is not None:
    print("Dados capturados com sucesso! Exibindo as 5 primeiras linhas:")
    print(df_historico.head())

    # Exemplo de como "trabalhar" os dados: calcular o valor de mercado médio
    valor_medio_mercado = df_historico['vlr_mercado'].mean()
    print(f"\nO valor de mercado médio para o período foi: {valor_medio_mercado:.2f}")