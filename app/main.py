import os
import sys
import argparse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Define o diretório raiz do projeto 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Adiciona a raiz ao PYTHONPATH para permitir imports como "from utils.logger"
sys.path.append(BASE_DIR)
from utils.logger import setup_logger
# Importa as funções de dados do novo módulo
from utils.market_data import (
    buscar_historico,
    buscar_dados_benchmark,
    buscar_dados_tesouro,
    buscar_dados_bcb,
    buscar_dados_b3,
    processar_benchmarks,
    _ensure_series
)

# Carrega as variáveis de ambiente do arquivo .env
#load_dotenv(os.path.join(BASE_DIR, '.env'))
load_dotenv()

# Obtém o token da variável de ambiente
token = os.getenv('DLP_TOKEN')

if not token:
    msg = "A variável de ambiente 'DLP_TOKEN' não foi encontrada ou está vazia. Verifique seu arquivo .env."
    # O logger ainda não foi configurado, então usamos print e raise para um erro crítico.
    print(f"CRITICAL: {msg}")
    raise ValueError(msg)

def get_report_path(subfolder: str) -> str:
    """Retorna o caminho absoluto para uma subpasta de relatórios, criando-a se necessário."""
    path = os.path.join(BASE_DIR, "reports", subfolder)
    os.makedirs(path, exist_ok=True)
    return path

def gerar_grafico_twr(df: pd.DataFrame, nome_grafico: str, logger) -> tuple[pd.DataFrame | None, plt.Figure | None]:
    """
    Calcula o Time-Weighted Return (TWR) e gera um gráfico da sua evolução.

    Args:
        df (pd.DataFrame): DataFrame com o histórico, contendo 'date', 'vlr_mercado' e 'vlr_investido'.
        nome_grafico (str): Nome para o título e arquivo do gráfico (ex: 'KLBN11' ou 'Ações').
    Returns:
        tuple: (DataFrame TWR, Figure Object)
    """
    colunas_necessarias = ['date', 'vlr_mercado', 'vlr_investido', 'proventos']
    if not all(col in df.columns for col in colunas_necessarias):
        logger.error(f"DataFrame não contém as colunas necessárias ({', '.join(colunas_necessarias)}) para calcular o TWR.")
        return None, None

    # Garante que a pasta de gráficos exista
    pasta_graficos = get_report_path("twr")

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
    fig = plt.figure(figsize=(12, 7))
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
    
    return df_twr, fig


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
    pasta_graficos = get_report_path("evolucao")

    # Prepara os dados para o gráfico
    df_grafico = df.copy()
    df_grafico['date'] = pd.to_datetime(df_grafico['date'])
    df_grafico = df_grafico.sort_values(by='date')

    # Calcula a evolução percentual, tratando divisão por zero
    df_grafico['evolucao_%'] = 0.0
    df_grafico.loc[df_grafico['vlr_investido'] != 0, 'evolucao_%'] = \
        ((df_grafico['vlr_mercado'] / df_grafico['vlr_investido']) - 1) * 100

    # Cria o gráfico
    fig = plt.figure(figsize=(12, 7))
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
    return fig

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
    pasta_graficos = get_report_path("evolucao")

    # Prepara os dados para o gráfico
    df_grafico = df.copy()
    df_grafico['date'] = pd.to_datetime(df_grafico['date'])
    df_grafico = df_grafico.sort_values(by='date')

    # Cria o gráfico
    fig = plt.figure(figsize=(12, 7))
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
    return fig

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
    # 'A' está sendo preterido em versões recentes do pandas; usar 'YE' (year end)
    anual_carteira = fator_carteira.resample('YE').last()
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
        # Usar 'YE' (year end) para compatibilidade futura
        anual_bench = fator_bench.resample('YE').last()
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
    # groupby with axis=1 is deprecated; transpose, groupby on columns, then transpose back
    df_resumo = df_resumo.T.groupby(level=0).first().T

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


def gerar_grafico_comparativo_twr(df_twr: pd.DataFrame, benchmarks_data: dict, nome_grafico: str, logger) -> plt.Figure:
    """
    Gera um gráfico comparando o TWR da carteira com outros benchmarks.
                logger.debug(f"Tipo de dados para o benchmark '{nome}': {type(retornos_anuais_bench)}")
    Args:
        df_twr (pd.DataFrame): DataFrame com a coluna 'twr_acc' e 'date'.
        benchmarks_data (dict): Dicionário onde a chave é o nome do benchmark (ex: 'IBOV')
                                e o valor é uma pd.Series com os dados de preço.
        nome_grafico (str): Nome do ativo principal para o título do gráfico.
    """
    pasta_graficos = get_report_path("twr")

    # --- VARIÁVEL DE CONTROLE ---
    # Define a cada quantos meses um rótulo de performance será exibido no gráfico.
    # Use 1 para exibir todos, 6 para exibir a cada semestre, 12 para anual, etc.
    intervalo_meses_rotulo = 6

    fig, ax = plt.subplots(figsize=(15, 10))

    # Lista para armazenar todos os objetos de texto que serão ajustados
    text_labels = []

    # Mapa para guardar a cor utilizada em cada linha (benchmark)
    color_map = {}

    # --- ORDENAR BENCHMARKS POR RENTABILIDADE TOTAL ---
    # Calcula o ranking dos benchmarks pela rentabilidade total (do maior para o menor)
    # Isso será usado para ordenar a tabela e a legenda
    def _parse_percent(val):
        # Aceita string tipo '23.4%' ou '23.4% (23.4%)' ou float
        if isinstance(val, str):
            v = val.split('%')[0].replace('(', '').replace(')', '').split()[0]
            try:
                return float(v.replace(',', '.')) / 100
            except Exception:
                return float('-inf')
        try:
            return float(val)
        except Exception:
            return float('-inf')

    # Garante que a função de resumo já foi chamada para obter a tabela
    df_resumo = calcular_rentabilidades_resumo(df_twr, benchmarks_data, nome_grafico, logger)
    if df_resumo is not None and 'Total' in df_resumo.columns:
        # Cria lista de (nome, total) e ordena
        total_ranking = sorted(
            [(idx, _parse_percent(df_resumo.at[idx, 'Total'])) for idx in df_resumo.index],
            key=lambda x: x[1], reverse=True)
        ordered_labels = [x[0] for x in total_ranking]
    else:
        ordered_labels = None

    # 1. Plotar o TWR da carteira (normalizado em base 100)
    # (twr_acc + 1) transforma o percentual de retorno em um fator de crescimento
    carteira_normalizada = (df_twr['twr_acc'] + 1) * 100
    # --- ORDEM DE PLOTAGEM ---
    # Monta lista de nomes para plotar, na ordem do ranking
    all_labels = [f'Carteira - {nome_grafico}'] + list(benchmarks_data.keys())
    if ordered_labels:
        # Garante que todos os labels estejam presentes
        plot_labels = [lbl for lbl in ordered_labels if lbl in all_labels]
        # Adiciona os que não estão no ranking (ex: benchmarks sem total)
        plot_labels += [lbl for lbl in all_labels if lbl not in plot_labels]
    else:
        plot_labels = all_labels

    # Plotagem na ordem do ranking
    for nome in plot_labels:
        if nome == f'Carteira - {nome_grafico}':
            linha_carteira, = ax.plot(df_twr['date'], carteira_normalizada, label=nome, color='red', linewidth=2.5)
            color_map[nome] = linha_carteira.get_color()
        elif nome in benchmarks_data:
            dados_benchmark = benchmarks_data[nome]
            if dados_benchmark is not None and not getattr(dados_benchmark, 'empty', False):
                benchmark_normalizado = (dados_benchmark / dados_benchmark.iloc[0]) * 100
                linha_bench, = ax.plot(benchmark_normalizado.index, benchmark_normalizado, label=nome, linestyle='--')
                color_map[nome] = linha_bench.get_color()

    # 2. Plotar cada benchmark (normalizado em base 100)
    # Adiciona rótulos nos pontos mensais correspondentes ao df_twr para cada benchmark (na ordem de plotagem)
    for nome in plot_labels:
        if nome == f'Carteira - {nome_grafico}':
            continue  # já rotulado abaixo
        if nome in benchmarks_data:
            dados_benchmark = benchmarks_data[nome]
            if dados_benchmark is not None and not getattr(dados_benchmark, 'empty', False):
                benchmark_normalizado = (dados_benchmark / dados_benchmark.iloc[0]) * 100
                benchmark_mensal = benchmark_normalizado.reindex(df_twr['date'], method='ffill')
                if isinstance(benchmark_mensal, pd.DataFrame):
                    benchmark_mensal = benchmark_mensal.iloc[:, 0]
                for i, (data, valor_ponto) in enumerate(benchmark_mensal.items()):
                    if pd.notna(valor_ponto) and (i % intervalo_meses_rotulo == 0 or i == 0):
                        label = ax.text(data, valor_ponto, f' {valor_ponto-100:.1f}%', va='top', ha='center', fontsize=8, alpha=0.7)
                        text_labels.append(label)
            else:
                logger.warning(f"Nenhum dado válido para o benchmark '{nome}'. Não será plotado.")

    # Adiciona os rótulos para a carteira a cada 6 meses
    for index, row in df_twr.iterrows():
        if index % intervalo_meses_rotulo == 0 or index == 0:
            valor_normalizado = (row['twr_acc'] + 1) * 100
            label = ax.text(row['date'], valor_normalizado, f' {valor_normalizado-100:.1f}%', va='bottom', ha='center', fontsize=8, color='red', weight='bold')
            text_labels.append(label)


    # 3. Customizar o gráfico
    ax.set_title(f'Comparativo de Rentabilidade: {nome_grafico} vs. Benchmarks', fontsize=16)
    ax.set_ylabel('Performance (Base 100)', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    # Ordena a legenda pela ordem de plotagem
    handles, labels = ax.get_legend_handles_labels()
    if ordered_labels:
        # Garante que todos os labels estejam presentes
        legend_labels = [lbl for lbl in plot_labels if lbl in labels]
        legend_handles = [handles[labels.index(lbl)] for lbl in legend_labels]
        ax.legend(legend_handles, legend_labels, fontsize=10, loc='upper left')
    else:
        ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.autofmt_xdate()

    # 4. Ajustar a posição dos rótulos para evitar sobreposição
    # A função adjust_text irá reposicionar os textos da lista 'text_labels'
    adjust_text(text_labels, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # 5. Adicionar a tabela de resumo de rentabilidade
    df_resumo = calcular_rentabilidades_resumo(df_twr, benchmarks_data, nome_grafico, logger)
    if df_resumo is not None:
        # Reordena a tabela pela ordem do ranking
        if ordered_labels:
            df_resumo = df_resumo.reindex(ordered_labels)
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

            def _find_color_for(name):
                # tentativa exata
                if name in color_map:
                    return color_map[name]
                # normaliza e tenta novamente
                name_norm = name.strip().lower()
                for k, v in color_map.items():
                    if k.strip().lower() == name_norm:
                        return v
                # tentativa por prefixo/sufixo
                for k, v in color_map.items():
                    kn = k.strip().lower()
                    if name_norm.startswith(kn) or kn.startswith(name_norm):
                        return v
                return None

            for (r, c), cell in celld.items():
                # coluna dos rótulos de linha no objeto table costuma ser -1
                if c == -1:
                    # nas células da tabela o header ocupa row 0, dados começam em row 1
                    if 1 <= r <= len(row_labels):
                        nome_linha = row_labels[r-1]
                        cor = _find_color_for(nome_linha)
                        if cor:
                            cell.get_text().set_color(cor)
                            cell.get_text().set_weight('bold')
        except Exception as e:
            logger.debug(f'Não foi possível colorir os rótulos das linhas na tabela: {e}')

        # Ajusta o layout para dar espaço para a tabela
        fig.subplots_adjust(bottom=0.3)

    # 6. Salvar o gráfico
    caminho_arquivo = os.path.join(pasta_graficos, f'comparativo_twr_{nome_grafico}.png')
    plt.savefig(caminho_arquivo, bbox_inches='tight')
    logger.info(f"Gráfico comparativo de TWR salvo com sucesso em: {caminho_arquivo}")
    return fig

def gerar_twr_historico(benchmarks_data: dict, years: int, nome_grafico: str, end_date: pd.Timestamp, logger) -> plt.Figure | None:
    """Gera um gráfico com o TWR histórico (normalizado em base 100) para cada benchmark
    no período de `years` anos até `end_date`. Se uma série não tiver histórico completo,
    ela é plotada a partir da sua primeira data disponível dentro do intervalo.

    Args:
        benchmarks_data (dict): dicionário nome -> pd.Series/df com índices de data.
        years (int): número de anos do histórico a plotar.
        nome_grafico (str): sufixo/nome para os arquivos gerados.
        end_date (pd.Timestamp): data final do período (usualmente df_twr['date'].max()).
        logger: logger para mensagens.
    """
    pasta_graficos = get_report_path("twr")

    start_candidate = end_date - pd.DateOffset(years=years) + pd.Timedelta(days=1)

    # Prepara um DataFrame para armazenar as séries normalizadas (base 100)
    normalized = []
    names = []

    for nome, série in benchmarks_data.items():
        s = _ensure_series(série)
        if s is None:
            logger.debug(f"Ignorando benchmark '{nome}': série inválida")
            continue

        # Ensina selecção do período: tenta usar start_candidate, senão usa o primeiro disponível
        try:
            s_period = s.loc[start_candidate:end_date]
        except Exception:
            # índices não compatíveis, tenta converter index para DatetimeIndex
            s = s.copy()
            s.index = pd.to_datetime(s.index, errors='coerce')
            s = s.dropna()
            if s.empty:
                logger.debug(f"Ignorando benchmark '{nome}': índice de datas inválido")
                continue
            s_period = s.loc[start_candidate:end_date]

        if s_period.empty:
            # plot a partir da primeira data disponível até end_date
            s_period = s.loc[:end_date]
            if s_period.empty:
                logger.debug(f"Ignorando benchmark '{nome}': sem dados até {end_date}")
                continue

        # Normaliza para base 100 a partir do primeiro ponto disponível no sub-período
        s_norm = (s_period / s_period.iloc[0]) * 100
        normalized.append(s_norm)
        names.append(nome)

    if not normalized:
        logger.info("Nenhum benchmark válido para gerar TWR histórico.")
        return None

    df_plot = pd.concat(normalized, axis=1)
    df_plot.columns = names

    # Calcula rentabilidade total para ordenação (do maior para o menor)
    # df_plot está em base 100. Usamos o último valor válido de cada série.
    total_returns = pd.Series(index=df_plot.columns, dtype=float)
    for col in df_plot.columns:
        last_valid = df_plot[col].dropna().iloc[-1]
        total_returns[col] = (last_valid / 100) - 1

    sorted_cols = total_returns.sort_values(ascending=False).index.tolist()
    df_plot = df_plot[sorted_cols]

    # Gera gráfico (usando fig/ax para permitir tabela abaixo)
    # Aumentado para melhorar a visualização da tabela e colunas (horizontal e vertical)
    fig, ax = plt.subplots(figsize=(18, 10))

    # Captura cores para colorir os rótulos da tabela
    color_map = {}
    for col in df_plot.columns:
        label_text = f"{col} ({total_returns[col]:.1%})" if pd.notna(total_returns[col]) else col
        line, = ax.plot(df_plot.index, df_plot[col], label=label_text)
        color_map[col] = line.get_color()

    ax.set_title(f'TWR Histórico ({years} anos) - {nome_grafico}', fontsize=14)
    ax.set_ylabel('Performance (Base 100)', fontsize=12)
    ax.set_xlabel('Data', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()

    # --- Monta tabela de rentabilidade anual semelhante ao comparativo ---
    # Calcula retornos anuais para cada série a partir dos dados originais (não normalizados)
    anos_dict = {}
    totais = {}
    for nome in sorted_cols:
        # recupera a série original a partir do df_plot col (reconstrói fator usando índice)
        s = df_plot[nome] / 100.0  # fator (pois df_plot é base 100)
        # recupera fator anual no final do ano
        try:
            anual = s.resample('YE').last()
            retornos = anual.pct_change().fillna(anual.iloc[0] - 1) if not anual.empty else pd.Series(dtype=float)
        except Exception:
            # tenta converter index para DatetimeIndex e repetir
            temp = s.copy()
            temp.index = pd.to_datetime(temp.index, errors='coerce')
            temp = temp.dropna()
            if temp.empty:
                retornos = pd.Series(dtype=float)
            else:
                anual = temp.resample('YE').last()
                retornos = anual.pct_change().fillna(anual.iloc[0] - 1) if not anual.empty else pd.Series(dtype=float)

        # Armazena retornos (index são timestamps de fim de ano)
        anos_dict[nome] = retornos
        # total acumulado: último fator - 1
        try:
            totais[nome] = float(s.iloc[-1] - 1)
        except Exception:
            totais[nome] = np.nan

    # Junta em um DataFrame: linhas = nomes, colunas = anos
    if anos_dict:
        df_anos = pd.DataFrame({k: v for k, v in anos_dict.items()}).T
        # df_anos index são nomes, columns are Timestamp; convert columns to year ints when possible
        col_years = []
        for c in df_anos.columns:
            try:
                col_years.append(int(pd.to_datetime(c).year))
            except Exception:
                col_years.append(str(c))
        df_anos.columns = col_years

        # Reorder columns (years ascending then Total will be computed below)
        year_cols = [c for c in df_anos.columns if str(c).lower() != 'total']
        try:
            year_cols_sorted = sorted(year_cols)
        except Exception:
            year_cols_sorted = year_cols
        ordered_cols = year_cols_sorted + (['Total'] if 'Total' in df_anos.columns else [])
        df_anos = df_anos.reindex(columns=ordered_cols)

        # Calcula acumulado por linha ao longo dos anos (preenche anos faltantes com 0)
        if year_cols_sorted:
            # Quando faltar algum ano, considera retorno anual 0 (fillna(0)) conforme solicitado
            acumulado_df = (1 + df_anos[year_cols_sorted].fillna(0)).cumprod(axis=1) - 1
        else:
            acumulado_df = pd.DataFrame(index=df_anos.index)

        # Calcula Total a partir dos anos (tratando anos ausentes como 0)
        if year_cols_sorted:
            totais_calc = (1 + df_anos[year_cols_sorted].fillna(0)).prod(axis=1) - 1
        else:
            # Se não há colunas de ano, considera Total como 0 para todas as séries
            totais_calc = pd.Series(0.0, index=df_anos.index)

        df_anos['Total'] = totais_calc

        # Formata as células: 'anual (acumulado)'
        def _format_cell(annual, acc):
            if pd.isna(annual):
                return '-'
            try:
                if pd.isna(acc):
                    return f'{float(annual):.1%}'
                return f'{float(annual):.1%} ({float(acc):.1%})'
            except Exception:
                return str(annual)

        df_formatted = pd.DataFrame(index=df_anos.index, columns=df_anos.columns, dtype=object)
        for c in year_cols_sorted:
            for idx in df_anos.index:
                annual = df_anos.at[idx, c]
                acc = acumulado_df.at[idx, c] if (idx in acumulado_df.index and c in acumulado_df.columns) else np.nan
                df_formatted.at[idx, c] = _format_cell(annual, acc)

        # Formata Total
        if 'Total' in df_anos.columns:
            for idx in df_anos.index:
                total_val = df_anos.at[idx, 'Total']
                df_formatted.at[idx, 'Total'] = f'{float(total_val):.1%}' if pd.notna(total_val) else '-'

        # Adiciona a tabela abaixo do gráfico
        cell_text = df_formatted.values
        row_labels = list(df_formatted.index)
        col_labels = list(df_formatted.columns)
        col_labels_display = [f"{c} (acc)" if str(c).lower() != 'total' else 'Total' for c in col_labels]

        tabela = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels_display,
                          loc='bottom', cellLoc='center', bbox=[0, -0.65, 1, 0.45])
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(9)
        # Aumenta a altura das linhas (1.6) para não espremer o texto
        tabela.scale(1, 1.6)

        # Colorir o rótulo das linhas para coincidir com as cores do gráfico
        try:
            celld = tabela.get_celld()

            def _find_color_for(name):
                if name in color_map:
                    return color_map[name]
                name_norm = name.strip().lower()
                for k, v in color_map.items():
                    if k.strip().lower() == name_norm:
                        return v
                for k, v in color_map.items():
                    kn = k.strip().lower()
                    if name_norm.startswith(kn) or kn.startswith(name_norm):
                        return v
                return None

            for (r, c), cell in celld.items():
                if c == -1:
                    if 1 <= r <= len(row_labels):
                        nome_linha = row_labels[r-1]
                        cor = _find_color_for(nome_linha)
                        if cor:
                            cell.get_text().set_color(cor)
                            cell.get_text().set_weight('bold')
        except Exception as e:
            logger.debug(f'Não foi possível colorir os rótulos da tabela histórica: {e}')

        # Ajusta o layout para dar espaço à tabela
        fig.subplots_adjust(bottom=0.45)

        # Salva CSV da tabela numérica (não formatada) e da tabela formatada para inspeção
        caminho_csv_table = os.path.join(pasta_graficos, f'twr_historico_{years}y_{nome_grafico}_table.csv')
        try:
            df_anos.to_csv(caminho_csv_table, sep=';', decimal=',')
        except Exception:
            logger.debug('Falha ao salvar CSV da tabela de rentabilidades históricas.')

    caminho_png = os.path.join(pasta_graficos, f'twr_historico_{years}y_{nome_grafico}.png')
    caminho_csv = os.path.join(pasta_graficos, f'twr_historico_{years}y_{nome_grafico}.csv')

    df_plot.to_csv(caminho_csv, sep=';', decimal=',')
    plt.savefig(caminho_png, bbox_inches='tight')
    logger.info(f'TWR histórico salvo em: {caminho_png} e dados em {caminho_csv}')
    return fig

def gerar_analise_risco(benchmarks_data: dict, risk_free_series: pd.Series | None, nome_analise: str, logger) -> plt.Figure | None:
    """
    Calcula métricas de risco (Volatilidade, Sharpe, Drawdown) e gera gráfico Risk x Return.
    """
    pasta_graficos = get_report_path("risco")
    
    metricas = []
    
    # Prepara série de retorno livre de risco (diário) para cálculo do Sharpe
    rf_returns = None
    if risk_free_series is not None:
        rf_returns = risk_free_series.pct_change().fillna(0)

    for nome, serie in benchmarks_data.items():
        s = _ensure_series(serie)
        if s is None or s.empty:
            continue
            
        # Retornos diários
        retornos = s.pct_change().fillna(0)
        
        # 1. Volatilidade Anualizada (Desvio Padrão * raiz(252))
        volatilidade = retornos.std() * np.sqrt(252)
        
        # 2. Retorno Anualizado (CAGR)
        days = (s.index[-1] - s.index[0]).days
        if days > 0:
            cagr = (s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1
        else:
            cagr = 0
            
        # 3. Sharpe Ratio = (Retorno Carteira - Retorno Livre de Risco) / Volatilidade
        if rf_returns is not None:
            # Alinha datas da SELIC com a carteira
            rf_aligned = rf_returns.reindex(retornos.index).fillna(0)
            excess_returns = retornos - rf_aligned
            if excess_returns.std() > 0:
                sharpe = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
            else:
                sharpe = 0
        else:
            # Fallback se sem SELIC (Sharpe simples)
            sharpe = cagr / volatilidade if volatilidade > 0 else 0
                
        # 4. Max Drawdown (Queda máxima do topo ao fundo)
        cummax = s.cummax()
        drawdown = (s / cummax) - 1
        max_drawdown = drawdown.min()
        
        metricas.append({
            'Ativo': nome,
            'Retorno Anualizado': cagr,
            'Volatilidade': volatilidade,
            'Sharpe': sharpe,
            'Max Drawdown': max_drawdown
        })
        
    if not metricas:
        logger.warning("Não foi possível calcular métricas de risco.")
        return None

    df_metricas = pd.DataFrame(metricas).set_index('Ativo')
    
    # Ordena por Sharpe (eficiência)
    df_metricas = df_metricas.sort_values('Sharpe', ascending=False)
    
    # Salva CSV
    caminho_csv = os.path.join(pasta_graficos, f'metricas_risco_{nome_analise}.csv')
    df_metricas.to_csv(caminho_csv, sep=';', decimal=',')
    logger.info(f"Tabela de métricas de risco salva em: {caminho_csv}")
    
    # Gera Gráfico Scatter (Risco x Retorno)
    fig = plt.figure(figsize=(14, 9))
    
    # Plota os pontos
    for ativo, row in df_metricas.iterrows():
        x = row['Volatilidade']
        y = row['Retorno Anualizado']
        
        # Destaque visual para a Carteira do usuário
        if str(ativo).startswith('Carteira'):
            plt.scatter(x, y, color='red', s=150, zorder=10, label=ativo, edgecolors='black')
            plt.text(x, y, f'  {ativo}', fontsize=10, fontweight='bold', color='red', va='bottom')
        else:
            plt.scatter(x, y, s=80, alpha=0.7, edgecolors='white')
            plt.text(x, y, f'  {ativo}', fontsize=8, alpha=0.8, va='bottom')

    plt.title(f'Risco (Volatilidade) x Retorno - {nome_analise}', fontsize=16)
    plt.xlabel('Risco (Volatilidade Anualizada)', fontsize=12)
    plt.ylabel('Retorno Anualizado (CAGR)', fontsize=12)
    
    # Formata eixos como porcentagem
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    
    caminho_png = os.path.join(pasta_graficos, f'grafico_risco_retorno_{nome_analise}.png')
    plt.savefig(caminho_png, bbox_inches='tight')
    logger.info(f"Gráfico de Risco x Retorno salvo em: {caminho_png}")
    return fig

def simular_evolucao_patrimonio(benchmarks_data: dict, carteiras_config: dict, aporte_mensal: float, meses_rebalanceamento: int, logger) -> dict:
    """
    Simula a evolução do patrimônio com aportes mensais e rebalanceamento periódico.
    Gera gráficos de área (Patrimônio vs Investido) e CSVs.
    Retorna um dicionário com os objetos Figure gerados.
    """
    pasta_graficos = get_report_path("simulacao")
    
    logger.info(f"Iniciando simulação: Aporte R${aporte_mensal:.2f}/mês, Rebalanceamento a cada {meses_rebalanceamento} meses.")

    resultados_consolidados = {}
    figuras_geradas = {}

    for nome_carteira, pesos in carteiras_config.items():
        # Verifica se todos os ativos da carteira estão disponíveis
        ativos_validos = True
        series_ativos = {}
        for ativo in pesos.keys():
            s = _ensure_series(benchmarks_data.get(ativo))
            if s is None or s.empty:
                logger.warning(f"Ativo '{ativo}' não encontrado ou vazio. Pulando simulação da carteira '{nome_carteira}'.")
                ativos_validos = False
                break
            series_ativos[ativo] = s
        
        if not ativos_validos:
            continue

        # Alinha as datas de todos os ativos (intersecção)
        df_precos = pd.concat(series_ativos, axis=1).dropna()
        if df_precos.empty:
            logger.warning(f"Sem dados coincidentes para a carteira '{nome_carteira}'.")
            continue

        # Inicialização da simulação
        # 'shares' armazena a quantidade "fictícia" de cada ativo (Valor / Preço)
        shares = pd.Series(0.0, index=pesos.keys())
        total_investido = 0.0
        
        historico_simulacao = []
        
        last_month = None
        months_since_start = 0
        
        # Itera dia a dia
        for data, row_precos in df_precos.iterrows():
            current_month = data.month
            
            # Verifica virada de mês para Aporte e Rebalanceamento
            if last_month is not None and current_month != last_month:
                months_since_start += 1
                
                # 1. Aporte Mensal (compra ativos mantendo proporção alvo ou apenas distribuindo o aporte)
                # Simplificação: Distribui o aporte conforme os pesos alvo
                cash_in = aporte_mensal
                total_investido += cash_in
                
                # Compra ativos
                shares += (cash_in * pd.Series(pesos)) / row_precos
                
                # 2. Rebalanceamento
                if months_since_start % meses_rebalanceamento == 0:
                    # Calcula valor atual total
                    valor_atual_total = (shares * row_precos).sum()
                    # Redefine shares para bater exatamente com os pesos alvo
                    # NOTA: Isso implica VENDA de ativos que excederam a meta e COMPRA dos que estão abaixo.
                    target_values = valor_atual_total * pd.Series(pesos)
                    shares = target_values / row_precos
            
            # No primeiro dia, faz o aporte inicial
            if last_month is None:
                total_investido += aporte_mensal
                shares += (aporte_mensal * pd.Series(pesos)) / row_precos

            last_month = current_month
            
            # Calcula valor diário do patrimônio
            valor_patrimonio = (shares * row_precos).sum()
            
            historico_simulacao.append({
                'Date': data,
                'Total Investido': total_investido,
                'Patrimônio Bruto': valor_patrimonio
            })
            
        # Cria DataFrame da simulação
        df_sim = pd.DataFrame(historico_simulacao)
        
        # Armazena para o gráfico consolidado
        resultados_consolidados[nome_carteira] = df_sim.set_index('Date')['Patrimônio Bruto']
        
        # Salva CSV
        safe_name = nome_carteira.replace('/', '_').replace('+', 'mais')
        caminho_csv = os.path.join(pasta_graficos, f'simulacao_{safe_name}.csv')
        df_sim.to_csv(caminho_csv, sep=';', decimal=',', index=False)
        
        # Gera Gráfico
        fig = plt.figure(figsize=(12, 7))
        plt.plot(df_sim['Date'], df_sim['Patrimônio Bruto'], label='Patrimônio Bruto', color='green', linewidth=2)
        plt.plot(df_sim['Date'], df_sim['Total Investido'], label='Total Investido', color='gray', linestyle='--', linewidth=1.5)
        
        # Preenche a área de lucro
        plt.fill_between(df_sim['Date'], df_sim['Total Investido'], df_sim['Patrimônio Bruto'], 
                         where=(df_sim['Patrimônio Bruto'] >= df_sim['Total Investido']),
                         color='green', alpha=0.1, interpolate=True)
        
        plt.title(f'Simulação: {nome_carteira}\n(Aporte: R${aporte_mensal} | Rebal: {meses_rebalanceamento} meses)', fontsize=14)
        plt.xlabel('Data')
        plt.ylabel('Valor (R$)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Formata eixo Y como moeda (aproximado)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('R${x:,.0f}'))
        
        caminho_png = os.path.join(pasta_graficos, f'simulacao_{safe_name}.png')
        plt.savefig(caminho_png, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Simulação '{nome_carteira}' concluída. Gráfico salvo em: {caminho_png}")
        figuras_geradas[nome_carteira] = fig

    # Gera Gráfico Consolidado com todas as simulações
    if resultados_consolidados:
        fig_consol = plt.figure(figsize=(16, 10))
        
        # Ordena pela rentabilidade final (maior para menor)
        sorted_items = sorted(resultados_consolidados.items(), key=lambda x: x[1].iloc[-1] if not x[1].empty else 0, reverse=True)
        
        for nome, serie in sorted_items:
            if serie.empty: continue
            final_val = serie.iloc[-1]
            plt.plot(serie.index, serie, label=f"{nome} (R${final_val:,.0f})", linewidth=2)
            
        plt.title(f'Consolidado: Simulações de Carteiras\n(Aporte: R${aporte_mensal} | Rebal: {meses_rebalanceamento} meses)', fontsize=16)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Patrimônio Bruto (R$)', fontsize=12)
        plt.legend(fontsize=9, loc='upper left')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('R${x:,.0f}'))
        plt.gcf().autofmt_xdate()
        
        caminho_consol = os.path.join(pasta_graficos, 'simulacao_consolidada_aportes.png')
        plt.savefig(caminho_consol, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico consolidado das simulações salvo em: {caminho_consol}")
        figuras_geradas['Consolidado'] = fig_consol

    return figuras_geradas

def main():
    """
    Função principal que orquestra a execução do script.
    """
    parser = argparse.ArgumentParser(description="Gera análises de carteira de investimentos a partir da API.")
    parser.add_argument('--debug', action='store_true', help='Ativa o modo de log detalhado (debug).')
    parser.add_argument('--historico', type=int, help='Número de anos para gerar TWR histórico de benchmarks (ex: 1, 5, 10).')
    
    parser.add_argument('--aporte', type=float, help='Valor do aporte mensal para simulação.')
    parser.add_argument('--rebalanceamento', type=int, help='Intervalo em meses para rebalanceamento.')

    # Grupo de argumentos mutuamente exclusivos: ou --ativo ou --classe deve ser fornecido.
    # Não forçamos aqui o required, pois quando --historico for usado sozinho não é necessário.
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--ativo', type=str, help='Código do ativo a ser analisado (ex: KLBN11).')
    group.add_argument('--classe', type=str, help='Classe de ativos a ser analisada (ex: AÇÃO).')

    args = parser.parse_args()

    # Se --historico não foi fornecido, então exige-se que o usuário passe --ativo ou --classe.
    if not args.historico and not (args.ativo or args.classe):
        parser.error("one of the arguments --ativo --classe is required unless --historico is provided")

    logger = setup_logger(debug=args.debug, log_file='main.log')
    if args.debug:
        logger.info("Modo de depuração ativado.")

    df_historico = None
    nome_analise = ""

    # Configuração centralizada de benchmarks (pode ser ajustada conforme necessidade)
    # Use esta lista para garantir consistência entre os modos
    benchmarks_yf_config = {
        'S&P 500': 'SPY',
        #'S&P 500 BRL': 'SPY.BA',
        'IVVB11': 'IVVB11.SA', # Disponível em ambos agora
        'IMID': 'IMID.L',
        'Bitcoin': 'BTC-USD'
    }
    benchmarks_b3_config = {
        #'IBSD': 'IBSD', # Disponível em ambos agora
        #'IDIV': 'IDIV',
        #'IBLV': 'IBLV'  # Disponível em ambos agora
    }
    benchmarks_bcb_config = {
        'SELIC': 11,
        'IPCA': 433
    }
    benchmarks_td_config = {
        # Nome do título deve ser exato como no CSV do Tesouro (Tesouro IPCA+)
        'TD IPCA 2035': {'titulo': 'Tesouro IPCA+', 'vencimento': '15/05/2035'},
        'TD IPCA 2045': {'titulo': 'Tesouro IPCA+', 'vencimento': '15/05/2045'}
    }
    
    # Definição das Carteiras Sintéticas para geração de índices e simulação
    carteiras_sinteticas_config = {
        'IMID BRL 50 + (IPCA+6%) 50': {'IMID BRL': 0.5, 'IPCA + 6%': 0.5},
        'IMID BRL 25 + (IPCA+6%) 75': {'IMID BRL': 0.25, 'IPCA + 6%': 0.75},
        'IMID BRL 75 + (IPCA+6%) 25': {'IMID BRL': 0.75, 'IPCA + 6%': 0.25},
        'IMID BRL 60 + (IPCA+6%) 40': {'IMID BRL': 0.60, 'IPCA + 6%': 0.40},
        'IMID BRL 40 + (IPCA+6%) 60': {'IMID BRL': 0.40, 'IPCA + 6%': 0.60},
        
        'IMID BRL 47.5 + (IPCA+6%) 47.5 + BTC 5': {'IMID BRL': 0.475, 'IPCA + 6%': 0.475, 'Bitcoin BRL': 0.05},
        'IMID BRL 23.75 + (IPCA+6%) 71.25 + BTC 5': {'IMID BRL': 0.2375, 'IPCA + 6%': 0.7125, 'Bitcoin BRL': 0.05},
        'IMID BRL 71.25 + (IPCA+6%) 23.75 + BTC 5': {'IMID BRL': 0.7125, 'IPCA + 6%': 0.2375, 'Bitcoin BRL': 0.05},
        'IMID BRL 70 + (IPCA+6%) 25 + BTC 5': {'IMID BRL': 0.7, 'IPCA + 6%': 0.25, 'Bitcoin BRL': 0.05},
        'IMID BRL 57 + (IPCA+6%) 38 + BTC 5': {'IMID BRL': 0.57, 'IPCA + 6%': 0.38, 'Bitcoin BRL': 0.05},
        'IMID BRL 38 + (IPCA+6%) 57 + BTC 5': {'IMID BRL': 0.38, 'IPCA + 6%': 0.57, 'Bitcoin BRL': 0.05},

        'IMID BRL 47.5 + TD 2035 47.5 + BTC 5': {'IMID BRL': 0.475, 'TD IPCA 2035': 0.475, 'Bitcoin BRL': 0.05},
        'IMID BRL 23.75 + TD 2035 71.25 + BTC 5': {'IMID BRL': 0.2375, 'TD IPCA 2035': 0.7125, 'Bitcoin BRL': 0.05},
        'IMID BRL 71.25 + TD 2035 23.75 + BTC 5': {'IMID BRL': 0.7125, 'TD IPCA 2035': 0.2375, 'Bitcoin BRL': 0.05},
        'IMID BRL 70 + TD 2035 25 + BTC 5': {'IMID BRL': 0.7, 'TD IPCA 2035': 0.25, 'Bitcoin BRL': 0.05},
        'IMID BRL 57 + TD 2035 38 + BTC 5': {'IMID BRL': 0.57, 'TD IPCA 2035': 0.38, 'Bitcoin BRL': 0.05},
        'IMID BRL 38 + TD 2035 57 + BTC 5': {'IMID BRL': 0.38, 'TD IPCA 2035': 0.57, 'Bitcoin BRL': 0.05},

        'IMID BRL 47.5 + TD 2045 47.5 + BTC 5': {'IMID BRL': 0.475, 'TD IPCA 2045': 0.475, 'Bitcoin BRL': 0.05},
        'IMID BRL 23.75 + TD 2045 71.25 + BTC 5': {'IMID BRL': 0.2375, 'TD IPCA 2045': 0.7125, 'Bitcoin BRL': 0.05},
        'IMID BRL 71.25 + TD 2045 23.75 + BTC 5': {'IMID BRL': 0.7125, 'TD IPCA 2045': 0.2375, 'Bitcoin BRL': 0.05},
        'IMID BRL 70 + TD 2045 25 + BTC 5': {'IMID BRL': 0.7, 'TD IPCA 2045': 0.25, 'Bitcoin BRL': 0.05},
        'IMID BRL 57 + TD 2045 38 + BTC 5': {'IMID BRL': 0.57, 'TD IPCA 2045': 0.38, 'Bitcoin BRL': 0.05},
        'IMID BRL 38 + TD 2045 57 + BTC 5': {'IMID BRL': 0.38, 'TD IPCA 2045': 0.57, 'Bitcoin BRL': 0.05},
        
        'IMID BRL/(IPCA+6%)': {'IMID BRL': 0.50, 'IDIV': 0.25, 'IPCA + 6%': 0.25},
    }

    # Lista de benchmarks que serão EXIBIDOS nos gráficos e tabelas.
    # O script calcula todos (para compor carteiras), mas só mostra estes.
    benchmarks_exibir = [
        #'S&P 500',
        #'IVVB11',
        #'IMID',
        #'IDIV',
        #'IPCA',
        #'SELIC',
        'IMID BRL',
        'TD IPCA 2035',
        'S&P 500 BRL',
        'IPCA + 6%',
        #'IMID BRL 50 + (IPCA+6%) 50',
        #'IMID BRL 25 + (IPCA+6%) 75',
        #'IMID BRL 75 + (IPCA+6%) 25',
        #'IMID BRL 60 + (IPCA+6%) 40',
        #'IMID BRL 40 + (IPCA+6%) 60',
        #'IMID BRL 47.5 + (IPCA+6%) 47.5 + BTC 5',
        #'IMID BRL 23.75 + (IPCA+6%) 71.25 + BTC 5',
        #'IMID BRL 71.25 + (IPCA+6%) 23.75 + BTC 5',
        #'IMID BRL 70 + (IPCA+6%) 25 + BTC 5',
        #'IMID BRL 57 + (IPCA+6%) 38 + BTC 5',
        #'IMID BRL 38 + (IPCA+6%) 57 + BTC 5',
        'IMID BRL 47.5 + TD 2035 47.5 + BTC 5',
        'IMID BRL 23.75 + TD 2035 71.25 + BTC 5',
        'IMID BRL 71.25 + TD 2035 23.75 + BTC 5',
        'IMID BRL 70 + TD 2035 25 + BTC 5',
        'IMID BRL 57 + TD 2035 38 + BTC 5',
        'IMID BRL 38 + TD 2035 57 + BTC 5',
        'IMID BRL 47.5 + TD 2045 47.5 + BTC 5',
        'IMID BRL 23.75 + TD 2045 71.25 + BTC 5',
        'IMID BRL 71.25 + TD 2045 23.75 + BTC 5',
        'IMID BRL 70 + TD 2045 25 + BTC 5',
        'IMID BRL 57 + TD 2045 38 + BTC 5',
        'IMID BRL 38 + TD 2045 57 + BTC 5',
        #'IDIV/IMID BRL/(IPCA+6%)'
    ]

    # Caso especial: quando o usuário solicita apenas '--historico N' (sem --ativo/--classe),
    # geramos somente o TWR histórico dos benchmarks e encerramos o script.
    if args.historico and not (args.ativo or args.classe):
        years = args.historico
        end_dt = pd.Timestamp.today()
        start_dt = (end_dt - pd.DateOffset(years=years) + pd.Timedelta(days=1))
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')

        logger.info(f"Modo histórico standalone: gerando TWR dos benchmarks para {years} anos ({start_date} a {end_date}).")

        # Define os benchmarks para o modo Histórico (visão macro)
        # Usa a configuração centralizada
        benchmarks_data = processar_benchmarks(start_date, end_date, benchmarks_yf_config, benchmarks_b3_config, benchmarks_bcb_config, benchmarks_td_config, carteiras_sinteticas_config, logger)
        
        # Captura SELIC para cálculo de Sharpe antes de filtrar
        selic_series = benchmarks_data.get('SELIC')

        # Filtra apenas os benchmarks que devem ser exibidos
        benchmarks_data_exibir = {k: v for k, v in benchmarks_data.items() if k in benchmarks_exibir}


        # Gera o TWR histórico e encerra
        try:
            gerar_twr_historico(benchmarks_data_exibir, years, f'historico_{years}y', end_dt, logger)
            gerar_analise_risco(benchmarks_data_exibir, selic_series, f'historico_{years}y', logger)
            
            # Se parâmetros de simulação foram passados, executa a simulação
            if args.aporte and args.rebalanceamento:
                # Filtra carteiras para simular apenas as que estão na lista de exibição (para não demorar demais)
                carteiras_para_simular = {k: v for k, v in carteiras_sinteticas_config.items() if k in benchmarks_exibir}
                simular_evolucao_patrimonio(benchmarks_data, carteiras_para_simular, args.aporte, args.rebalanceamento, logger)

        except Exception as e:
            logger.exception(f"Erro ao gerar TWR histórico standalone: {e}")
        return

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
        pasta_saida = get_report_path("dados_brutos")
        caminho_csv = os.path.join(pasta_saida, f'historico_completo_{nome_analise}.csv')
        df_historico.to_csv(caminho_csv, sep=';', decimal=',', index=False)
        logger.info(f"Histórico completo salvo em: {caminho_csv}")

        logger.info("Gerando gráficos...")
        gerar_grafico_evolucao(df_historico, nome_grafico=nome_analise, logger=logger)
        
        # Calcula o TWR e obtém o dataframe com os resultados
        df_twr, _ = gerar_grafico_twr(df_historico, nome_grafico=nome_analise, logger=logger)

        if df_twr is not None:
            # Define os benchmarks para o modo Comparativo (Ativo vs Mercado)
            # Usa a configuração centralizada
            
            # Define o período para a busca dos benchmarks
            start_date = df_twr['date'].min().strftime('%Y-%m-%d')
            end_date = df_twr['date'].max().strftime('%Y-%m-%d')
            logger.info(f"Período da análise: {start_date} a {end_date}")

            # Processa benchmarks usando a função centralizada
            benchmarks_data = processar_benchmarks(start_date, end_date, benchmarks_yf_config, benchmarks_b3_config, benchmarks_bcb_config, benchmarks_td_config, carteiras_sinteticas_config, logger)
            
            # Captura SELIC para cálculo de Sharpe antes de filtrar
            selic_series = benchmarks_data.get('SELIC')
            # Filtra apenas os benchmarks que devem ser exibidos
            benchmarks_data_exibir = {k: v for k, v in benchmarks_data.items() if k in benchmarks_exibir}

            # Se foi solicitado, gera o TWR histórico de benchmarks para o período em anos
            if getattr(args, 'historico', None):
                try:
                    # Cria uma cópia dos benchmarks e adiciona a carteira atual para comparação histórica
                    dados_historico = benchmarks_data_exibir.copy()
                    # Converte TWR acumulado (0.x) para fator (1.x) para ser comparável com preços
                    dados_historico[f'Carteira - {nome_analise}'] = df_twr.set_index('date')['twr_acc'] + 1
                    
                    gerar_twr_historico(dados_historico, args.historico, nome_analise, df_twr['date'].max(), logger)
                    # Gera análise de risco histórica
                    gerar_analise_risco(dados_historico, selic_series, f'{nome_analise}_historico_{args.historico}y', logger)
                    
                    # Se parâmetros de simulação foram passados, executa a simulação
                    if args.aporte and args.rebalanceamento:
                        carteiras_para_simular = {k: v for k, v in carteiras_sinteticas_config.items() if k in benchmarks_exibir}
                        simular_evolucao_patrimonio(benchmarks_data, carteiras_para_simular, args.aporte, args.rebalanceamento, logger)
                except Exception as e:
                    logger.debug(f"Erro ao gerar TWR histórico: {e}")

            gerar_grafico_comparativo_twr(df_twr, benchmarks_data_exibir, nome_grafico=nome_analise, logger=logger)
            
            # Gera análise de risco para o período comparativo (carteira vs benchmarks no período da carteira)
            dados_comparativo = benchmarks_data_exibir.copy()
            dados_comparativo[f'Carteira - {nome_analise}'] = df_twr.set_index('date')['twr_acc'] + 1
            gerar_analise_risco(dados_comparativo, selic_series, f'{nome_analise}_comparativo', logger)

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