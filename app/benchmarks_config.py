# app/benchmarks_config.py
"""
Arquivo de configuração para Benchmarks e Carteiras Sintéticas (V2).
Aqui definimos os catálogos de dados disponíveis e quais serão efetivamente usados.
"""

# --- CATÁLOGOS DE DADOS (Fontes Disponíveis) ---
# Definições técnicas de onde buscar os dados. Não edite a menos que queira adicionar novas fontes.

CATALOGO_YF = {
    'S&P 500': 'SPY',
    'S&P 500 BRL': 'SPY.BA', # Fallback se cálculo sintético falhar
    'IVVB11': 'IVVB11.SA',
    'BOVA11': 'BOVA11.SA',
    'IMID': 'IMID.L',
    'Bitcoin': 'BTC-USD',
    'Ibovespa (YF)': '^BVSP'
}

CATALOGO_B3 = {
    'IBOV': 'IBOV',
    'IFIX': 'IFIX',
    'SMLL': 'SMLL',
    'IDIV': 'IDIV',
}

CATALOGO_BCB = {
    'SELIC': 11,
    'IPCA': 433,
    'CDI': 12,
    'IMA-B': 12466,
    'IMA-B 5': 12467,
    'IMA-B 5+': 12468,
    'IRF-M': 12461,
    'IRF-M 1': 12463,
    'IMA-S': 12469,
    'IMA-Geral': 12462
}

CATALOGO_TD = {
    'TD IPCA 2035': {'titulo': 'Tesouro IPCA+', 'vencimento': '15/05/2035'},
    'TD IPCA 2045': {'titulo': 'Tesouro IPCA+', 'vencimento': '15/05/2045'}
}

# --- CONFIGURAÇÃO ATIVA ---
# Lista unificada do que será calculado e exibido nos gráficos.
# Para ocultar um item, basta comentar a linha.
#
# Formatos aceitos:
# 1. String: Nome de um ativo presente nos catálogos acima ou derivado (ex: 'S&P 500 BRL').
# 2. Dicionário: Definição de carteira sintética {'nome': '...', 'composicao': {...}}.

BENCHMARKS_ATIVOS = [
    # --- Índices de Mercado ---
    'CDI',
    # 'S&P 500 BRL',
    # 'IBOV',
    # 'IPCA + 6%',
    # 'Bitcoin BRL',
    # 'IMA-B 5+',
    
    # --- Carteiras Sintéticas (IMID BRL + IPCA+6%) ---
    # {'nome': 'IMID BRL 50 + (IPCA+6%) 50', 'composicao': {'IMID BRL': 0.5, 'IPCA + 6%': 0.5}},
    # {'nome': 'IMID BRL 25 + (IPCA+6%) 75', 'composicao': {'IMID BRL': 0.25, 'IPCA + 6%': 0.75}},
    # {'nome': 'IMID BRL 75 + (IPCA+6%) 25', 'composicao': {'IMID BRL': 0.75, 'IPCA + 6%': 0.25}},
    # {'nome': 'IMID BRL 60 + (IPCA+6%) 40', 'composicao': {'IMID BRL': 0.60, 'IPCA + 6%': 0.40}},
    # {'nome': 'IMID BRL 40 + (IPCA+6%) 60', 'composicao': {'IMID BRL': 0.40, 'IPCA + 6%': 0.60}},
    
    # --- Carteiras Sintéticas (IMID BRL + IPCA+6% + BTC) ---
    # {'nome': 'IMID BRL 47.5 + (IPCA+6%) 47.5 + BTC 5', 'composicao': {'IMID BRL': 0.475, 'IPCA + 6%': 0.475, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 23.75 + (IPCA+6%) 71.25 + BTC 5', 'composicao': {'IMID BRL': 0.2375, 'IPCA + 6%': 0.7125, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 71.25 + (IPCA+6%) 23.75 + BTC 5', 'composicao': {'IMID BRL': 0.7125, 'IPCA + 6%': 0.2375, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 70 + (IPCA+6%) 25 + BTC 5', 'composicao': {'IMID BRL': 0.7, 'IPCA + 6%': 0.25, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 57 + (IPCA+6%) 38 + BTC 5', 'composicao': {'IMID BRL': 0.57, 'IPCA + 6%': 0.38, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 38 + (IPCA+6%) 57 + BTC 5', 'composicao': {'IMID BRL': 0.38, 'IPCA + 6%': 0.57, 'Bitcoin BRL': 0.05}},

    # --- Carteiras Sintéticas (IMID BRL + TD 2035 + BTC) ---
    # {'nome': 'IMID BRL 47.5 + TD 2035 47.5 + BTC 5', 'composicao': {'IMID BRL': 0.475, 'TD IPCA 2035': 0.475, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 23.75 + TD 2035 71.25 + BTC 5', 'composicao': {'IMID BRL': 0.2375, 'TD IPCA 2035': 0.7125, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 71.25 + TD 2035 23.75 + BTC 5', 'composicao': {'IMID BRL': 0.7125, 'TD IPCA 2035': 0.2375, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 70 + TD 2035 25 + BTC 5', 'composicao': {'IMID BRL': 0.7, 'TD IPCA 2035': 0.25, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 57 + TD 2035 38 + BTC 5', 'composicao': {'IMID BRL': 0.57, 'TD IPCA 2035': 0.38, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 38 + TD 2035 57 + BTC 5', 'composicao': {'IMID BRL': 0.38, 'TD IPCA 2035': 0.57, 'Bitcoin BRL': 0.05}},

    # --- Carteiras Sintéticas (IMID BRL + TD 2045 + BTC) ---
    # {'nome': 'IMID BRL 47.5 + TD 2045 47.5 + BTC 5', 'composicao': {'IMID BRL': 0.475, 'TD IPCA 2045': 0.475, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 23.75 + TD 2045 71.25 + BTC 5', 'composicao': {'IMID BRL': 0.2375, 'TD IPCA 2045': 0.7125, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 71.25 + TD 2045 23.75 + BTC 5', 'composicao': {'IMID BRL': 0.7125, 'TD IPCA 2045': 0.2375, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 70 + TD 2045 25 + BTC 5', 'composicao': {'IMID BRL': 0.7, 'TD IPCA 2045': 0.25, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 57 + TD 2045 38 + BTC 5', 'composicao': {'IMID BRL': 0.57, 'TD IPCA 2045': 0.38, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 38 + TD 2045 57 + BTC 5', 'composicao': {'IMID BRL': 0.38, 'TD IPCA 2045': 0.57, 'Bitcoin BRL': 0.05}},

    # --- Carteiras Sintéticas (IMID BRL + IMA-B + BTC) ---
    # {'nome': 'IMID BRL 47.5 + IMA-B 47.5 + BTC 5', 'composicao': {'IMID BRL': 0.475, 'IMA-B': 0.475, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 23.75 + IMA-B 71.25 + BTC 5', 'composicao': {'IMID BRL': 0.2375, 'IMA-B': 0.7125, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 71.25 + IMA-B 23.75 + BTC 5', 'composicao': {'IMID BRL': 0.7125, 'IMA-B': 0.2375, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 70 + IMA-B 25 + BTC 5', 'composicao': {'IMID BRL': 0.7, 'IMA-B': 0.25, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 57 + IMA-B 38 + BTC 5', 'composicao': {'IMID BRL': 0.57, 'IMA-B': 0.38, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 38 + IMA-B 57 + BTC 5', 'composicao': {'IMID BRL': 0.38, 'IMA-B': 0.57, 'Bitcoin BRL': 0.05}},

    # --- Carteiras Sintéticas (IMID BRL + IMA-B 5 + BTC) ---
    {'nome': 'IMID BRL 47.5 + IMA-B5 47.5 + BTC 5', 'composicao': {'IMID BRL': 0.475, 'IMA-B 5': 0.475, 'Bitcoin BRL': 0.05}},
    {'nome': 'IMID BRL 23.75 + IMA-B5 71.25 + BTC 5', 'composicao': {'IMID BRL': 0.2375, 'IMA-B 5': 0.7125, 'Bitcoin BRL': 0.05}},
    {'nome': 'IMID BRL 71.25 + IMA-B5 23.75 + BTC 5', 'composicao': {'IMID BRL': 0.7125, 'IMA-B 5': 0.2375, 'Bitcoin BRL': 0.05}},
    {'nome': 'IMID BRL 70 + IMA-B5 25 + BTC 5', 'composicao': {'IMID BRL': 0.7, 'IMA-B 5': 0.25, 'Bitcoin BRL': 0.05}},
    {'nome': 'IMID BRL 57 + IMA-B5 38 + BTC 5', 'composicao': {'IMID BRL': 0.57, 'IMA-B 5': 0.38, 'Bitcoin BRL': 0.05}},
    {'nome': 'IMID BRL 38 + IMA-B5 57 + BTC 5', 'composicao': {'IMID BRL': 0.38, 'IMA-B 5': 0.57, 'Bitcoin BRL': 0.05}},

    # --- Carteiras Sintéticas (IMID BRL + 'IMA-B 5+' + BTC) ---
    # {'nome': 'IMID BRL 47.5 + IMA-B5+ 47.5 + BTC 5', 'composicao': {'IMID BRL': 0.475, 'IMA-B 5+': 0.475, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 23.75 + IMA-B5+ 71.25 + BTC 5', 'composicao': {'IMID BRL': 0.2375, 'IMA-B 5+': 0.7125, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 71.25 + IMA-B5+ 23.75 + BTC 5', 'composicao': {'IMID BRL': 0.7125, 'IMA-B 5+': 0.2375, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 70 + IMA-B5+ 25 + BTC 5', 'composicao': {'IMID BRL': 0.7, 'IMA-B 5+': 0.25, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 57 + IMA-B5+ 38 + BTC 5', 'composicao': {'IMID BRL': 0.57, 'IMA-B 5+': 0.38, 'Bitcoin BRL': 0.05}},
    # {'nome': 'IMID BRL 38 + IMA-B5+ 57 + BTC 5', 'composicao': {'IMID BRL': 0.38, 'IMA-B 5+': 0.57, 'Bitcoin BRL': 0.05}},

    # --- Carteiras Sintéticas (IMID BRL + IMA-B 5) ---
    # {'nome': 'IMID BRL 50 + IMA-B5 50', 'composicao': {'IMID BRL': 0.5, 'IMA-B 5': 0.5}},
    # {'nome': 'IMID BRL 25 + IMA-B5 75', 'composicao': {'IMID BRL': 0.25, 'IMA-B 5': 0.75}},
    # {'nome': 'IMID BRL 75 + IMA-B5 25', 'composicao': {'IMID BRL': 0.75, 'IMA-B 5': 0.25}},
    # {'nome': 'IMID BRL 60 + IMA-B5 40', 'composicao': {'IMID BRL': 0.6, 'IMA-B 5': 0.4}},
    # {'nome': 'IMID BRL 40 + IMA-B5 60', 'composicao': {'IMID BRL': 0.4, 'IMA-B 5': 0.6}},

    # --- Outros ---
    # {'nome': 'IMID BRL/(IPCA+6%)', 'composicao': {'IMID BRL': 0.50, 'IDIV': 0.25, 'IPCA + 6%': 0.25}},
    # {'nome': 'IDIV', 'composicao': {'IDIV': 1.0}},
    {'nome': 'IMA-B', 'composicao': {'IMA-B': 1.0}},
    {'nome': 'IMA-B 5', 'composicao': {'IMA-B 5': 1.0}},
    {'nome': 'IMA-B 5+', 'composicao': {'IMA-B 5+': 1.0}},
]
