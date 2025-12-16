# Configuração centralizada de benchmarks e carteiras

BENCHMARKS_YF = {
    'S&P 500': 'SPY',
    'S&P 500 BRL': 'SPY.BA',
    'IVVB11': 'IVVB11.SA', # Disponível em ambos agora
    'IMID': 'IMID.L',
    'Bitcoin': 'BTC-USD',
    'Ibovespa (YF)': '^BVSP' # Índice teórico via Yahoo (histórico mais longo que ETFs)
}

BENCHMARKS_B3 = {
    #'IBSD': 'IBSD', # Índice Brasil Setorial Dividendos
    'IDIV': 'IDIV', # Índice Dividendos
    'IBLV': 'IBLV', # Índice Baixa Volatilidade
    'IBOV': 'IBOV', # Ibovespa (Benchmark Principal)
    'SMLL': 'SMLL', # Small Caps (Empresas de menor capitalização)
    'IFIX': 'IFIX', # Índice de Fundos Imobiliários
    'UTIL': 'UTIL', # Índice de Utilidade Pública (Defensivo/Elétricas)
}

BENCHMARKS_BCB = {
    # --- Taxas Básicas ---
    'SELIC': 11,           # Taxa Selic (% a.d.)
    'IPCA': 433,           # IPCA Mensal (% a.m.)
    'CDI': 12,             # Taxa CDI (% a.d.)

    # --- Indexados à Inflação (NTN-B / IPCA+) ---
    'IMA-B': 12466,        # Carteira completa de NTN-B
    'IMA-B 5': 12467,      # NTN-B com vencimento até 5 anos (Curto Prazo)
    'IMA-B 5+': 12468,     # NTN-B com vencimento acima de 5 anos (Longo Prazo)

    # --- Prefixados (LTN e NTN-F) ---
    'IRF-M': 12461,        # Carteira completa de Prefixados
    'IRF-M 1': 12463,      # Prefixados até 1 ano (Curto Prazo)
    'IRF-M 1+': 12464,     # Prefixados acima de 1 ano (Longo Prazo)

    # --- Pós-Fixados (LFT / Tesouro Selic) ---
    'IMA-S': 12469,        # Carteira de LFTs (Tesouro Selic)

    # --- Índice Geral ---
    'IMA-Geral': 12462     # Média de todos os títulos públicos
}

BENCHMARKS_TD = {
    # Nome do título deve ser exato como no CSV do Tesouro (Tesouro IPCA+)
    'TD IPCA 2035': {'titulo': 'Tesouro IPCA+', 'vencimento': '15/05/2035'},
    'TD IPCA 2045': {'titulo': 'Tesouro IPCA+', 'vencimento': '15/05/2045'}
}

# Definição das Carteiras Sintéticas para geração de índices e simulação
CARTEIRAS_SINTETICAS = {
    #'IMID BRL 50 + (IPCA+6%) 50': {'IMID BRL': 0.5, 'IPCA + 6%': 0.5},
    #'IMID BRL 25 + (IPCA+6%) 75': {'IMID BRL': 0.25, 'IPCA + 6%': 0.75},
    #'IMID BRL 75 + (IPCA+6%) 25': {'IMID BRL': 0.75, 'IPCA + 6%': 0.25},
    #'IMID BRL 60 + (IPCA+6%) 40': {'IMID BRL': 0.60, 'IPCA + 6%': 0.40},
    #'IMID BRL 40 + (IPCA+6%) 60': {'IMID BRL': 0.40, 'IPCA + 6%': 0.60},
    
    #'IMID BRL 47.5 + (IPCA+6%) 47.5 + BTC 5': {'IMID BRL': 0.475, 'IPCA + 6%': 0.475, 'Bitcoin BRL': 0.05},
    #'IMID BRL 23.75 + (IPCA+6%) 71.25 + BTC 5': {'IMID BRL': 0.2375, 'IPCA + 6%': 0.7125, 'Bitcoin BRL': 0.05},
    #'IMID BRL 71.25 + (IPCA+6%) 23.75 + BTC 5': {'IMID BRL': 0.7125, 'IPCA + 6%': 0.2375, 'Bitcoin BRL': 0.05},
    #'IMID BRL 70 + (IPCA+6%) 25 + BTC 5': {'IMID BRL': 0.7, 'IPCA + 6%': 0.25, 'Bitcoin BRL': 0.05},
    #'IMID BRL 57 + (IPCA+6%) 38 + BTC 5': {'IMID BRL': 0.57, 'IPCA + 6%': 0.38, 'Bitcoin BRL': 0.05},
    #'IMID BRL 38 + (IPCA+6%) 57 + BTC 5': {'IMID BRL': 0.38, 'IPCA + 6%': 0.57, 'Bitcoin BRL': 0.05},

    #'IMID BRL 47.5 + TD 2035 47.5 + BTC 5': {'IMID BRL': 0.475, 'TD IPCA 2035': 0.475, 'Bitcoin BRL': 0.05},
    #'IMID BRL 23.75 + TD 2035 71.25 + BTC 5': {'IMID BRL': 0.2375, 'TD IPCA 2035': 0.7125, 'Bitcoin BRL': 0.05},
    #'IMID BRL 71.25 + TD 2035 23.75 + BTC 5': {'IMID BRL': 0.7125, 'TD IPCA 2035': 0.2375, 'Bitcoin BRL': 0.05},
    #'IMID BRL 70 + TD 2035 25 + BTC 5': {'IMID BRL': 0.7, 'TD IPCA 2035': 0.25, 'Bitcoin BRL': 0.05},
    #'IMID BRL 57 + TD 2035 38 + BTC 5': {'IMID BRL': 0.57, 'TD IPCA 2035': 0.38, 'Bitcoin BRL': 0.05},
    #'IMID BRL 38 + TD 2035 57 + BTC 5': {'IMID BRL': 0.38, 'TD IPCA 2035': 0.57, 'Bitcoin BRL': 0.05},

    #'IMID BRL 47.5 + TD 2045 47.5 + BTC 5': {'IMID BRL': 0.475, 'TD IPCA 2045': 0.475, 'Bitcoin BRL': 0.05},
    #'IMID BRL 23.75 + TD 2045 71.25 + BTC 5': {'IMID BRL': 0.2375, 'TD IPCA 2045': 0.7125, 'Bitcoin BRL': 0.05},
    #'IMID BRL 71.25 + TD 2045 23.75 + BTC 5': {'IMID BRL': 0.7125, 'TD IPCA 2045': 0.2375, 'Bitcoin BRL': 0.05},
    #'IMID BRL 70 + TD 2045 25 + BTC 5': {'IMID BRL': 0.7, 'TD IPCA 2045': 0.25, 'Bitcoin BRL': 0.05},
    #'IMID BRL 57 + TD 2045 38 + BTC 5': {'IMID BRL': 0.57, 'TD IPCA 2045': 0.38, 'Bitcoin BRL': 0.05},
    #'IMID BRL 38 + TD 2045 57 + BTC 5': {'IMID BRL': 0.38, 'TD IPCA 2045': 0.57, 'Bitcoin BRL': 0.05},

    #'IMID BRL 47.5 + IMA-B 47.5 + BTC 5': {'IMID BRL': 0.475, 'IMA-B': 0.475, 'Bitcoin BRL': 0.05},
    #'IMID BRL 23.75 + IMA-B 71.25 + BTC 5': {'IMID BRL': 0.2375, 'IMA-B': 0.7125, 'Bitcoin BRL': 0.05},
    #'IMID BRL 71.25 + IMA-B 23.75 + BTC 5': {'IMID BRL': 0.7125, 'IMA-B': 0.2375, 'Bitcoin BRL': 0.05},
    #'IMID BRL 70 + IMA-B 25 + BTC 5': {'IMID BRL': 0.7, 'IMA-B': 0.25, 'Bitcoin BRL': 0.05},
    #'IMID BRL 57 + IMA-B 38 + BTC 5': {'IMID BRL': 0.57, 'IMA-B': 0.38, 'Bitcoin BRL': 0.05},
    #'IMID BRL 38 + IMA-B 57 + BTC 5': {'IMID BRL': 0.38, 'IMA-B': 0.57, 'Bitcoin BRL': 0.05},

    'IMID BRL 47.5 + IMA-B5 47.5 + BTC 5': {'IMID BRL': 0.475, 'IMA-B 5': 0.475, 'Bitcoin BRL': 0.05},
    'IMID BRL 23.75 + IMA-B5 71.25 + BTC 5': {'IMID BRL': 0.2375, 'IMA-B 5': 0.7125, 'Bitcoin BRL': 0.05},
    'IMID BRL 71.25 + IMA-B5 23.75 + BTC 5': {'IMID BRL': 0.7125, 'IMA-B 5': 0.2375, 'Bitcoin BRL': 0.05},
    'IMID BRL 70 + IMA-B5 25 + BTC 5': {'IMID BRL': 0.7, 'IMA-B 5': 0.25, 'Bitcoin BRL': 0.05},
    'IMID BRL 57 + IMA-B5 38 + BTC 5': {'IMID BRL': 0.57, 'IMA-B 5': 0.38, 'Bitcoin BRL': 0.05},
    'IMID BRL 38 + IMA-B5 57 + BTC 5': {'IMID BRL': 0.38, 'IMA-B 5': 0.57, 'Bitcoin BRL': 0.05},
    
    'IMID BRL 50 + IMA-B5 50': {'IMID BRL': 0.5, 'IMA-B 5': 0.5},
    'IMID BRL 25 + IMA-B5 75': {'IMID BRL': 0.25, 'IMA-B 5': 0.75},
    'IMID BRL 75 + IMA-B5 25': {'IMID BRL': 0.75, 'IMA-B 5': 0.25},
    'IMID BRL 60 + IMA-B5 40': {'IMID BRL': 0.6, 'IMA-B 5': 0.4},
    'IMID BRL 40 + IMA-B5 60': {'IMID BRL': 0.4, 'IMA-B 5': 0.6},
    
    #'IMID BRL/(IPCA+6%)': {'IMID BRL': 0.50, 'IDIV': 0.25, 'IPCA + 6%': 0.25},
}

# Lista de benchmarks que serão EXIBIDOS nos gráficos e tabelas.
BENCHMARKS_EXIBIR = [
    'IBOV',
    #'S&P 500',
    #'IVVB11',
    #'IMID',
    #'IDIV',
    #'IPCA',
    #'SELIC',
    #'IMID BRL',
    #'TD IPCA 2035',
    #'S&P 500 BRL',
    #'IPCA + 6%',
    #'IMA-B',
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
    'IMID BRL 47.5 + IMA-B 47.5 + BTC 5',
    'IMID BRL 23.75 + IMA-B 71.25 + BTC 5',
    'IMID BRL 71.25 + IMA-B 23.75 + BTC 5',
    'IMID BRL 70 + IMA-B 25 + BTC 5',
    'IMID BRL 57 + IMA-B 38 + BTC 5',
    'IMID BRL 38 + IMA-B 57 + BTC 5',
    'IMID BRL 47.5 + IMA-B5 47.5 + BTC 5',
    'IMID BRL 23.75 + IMA-B5 71.25 + BTC 5',
    'IMID BRL 71.25 + IMA-B5 23.75 + BTC 5',
    'IMID BRL 70 + IMA-B5 25 + BTC 5',
    'IMID BRL 57 + IMA-B5 38 + BTC 5',
    'IMID BRL 38 + IMA-B5 57 + BTC 5',
    'IMID BRL 50 + IMA-B5 50',
    'IMID BRL 25 + IMA-B5 75',
    'IMID BRL 75 + IMA-B5 25',
    'IMID BRL 60 + IMA-B5 40',
    'IMID BRL 40 + IMA-B5 60',
]
