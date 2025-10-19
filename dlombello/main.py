import requests
import os
from dotenv import load_dotenv
import pandas as pd
from requests.exceptions import RequestException
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém o token da variável de ambiente
token = os.getenv('DLP_TOKEN')

if not token:
    raise ValueError("A variável de ambiente 'DLP_TOKEN' não foi encontrada ou está vazia. Verifique seu arquivo .env.")

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

# --- Exemplo de uso da função ---
# Busca o histórico do ativo 'KLBN11'
df_historico = buscar_historico(token, ativo="KLBN11")

if df_historico is not None:
    print("Dados capturados com sucesso! Exibindo as 5 primeiras linhas:")
    print(df_historico.head())
    gerar_grafico_evolucao(df_historico, ativo="KLBN11")
    gerar_grafico_percentual(df_historico, ativo="KLBN11")