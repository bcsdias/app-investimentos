# App de Investimentos e Análise de Portfólio

Este projeto é uma ferramenta completa para análise de rentabilidade de carteiras de investimentos, comparação com benchmarks de mercado, cálculo de risco e simulação de estratégias de alocação.

O sistema consome dados de uma API proprietária de histórico de investimentos e cruza com dados de mercado de diversas fontes (Yahoo Finance, B3, Banco Central, Tesouro Direto) para gerar relatórios detalhados.

## 🚀 Funcionalidades Principais

*   **Cálculo de Rentabilidade Real (TWR):** Utiliza a metodologia *Time-Weighted Return* para calcular o retorno da carteira, isolando o efeito dos aportes e retiradas.
*   **Comparativo com Benchmarks:** Compara a performance da carteira (ou ativo/classe) contra diversos índices de mercado (IBOV, S&P 500, CDI, IPCA+, IMA-B, Bitcoin, etc.).
*   **Análise de Risco x Retorno:** Gera gráficos de dispersão (Scatter Plot) correlacionando Volatilidade (Risco) e Retorno Anualizado (CAGR) para avaliar a eficiência dos ativos (Índice de Sharpe).
*   **Simulação de Carteiras:** Permite simular a evolução patrimonial de carteiras teóricas (ex: 50% Renda Fixa + 50% Renda Variável) com aportes mensais e rebalanceamento periódico.
*   **Benchmarks Sintéticos:** Cria índices personalizados, como "S&P 500 em Reais", "IPCA + 6%", ou carteiras mistas.
*   **Extensão de Dados:** Utiliza ETFs como proxies para estender séries históricas de índices que pararam de ser divulgados publicamente (ex: IMA-B via IMAB11).

## 📊 Fontes de Dados

O sistema integra dados de múltiplas fontes automaticamente:

1.  **API do Usuário:** Histórico de transações e posição da carteira.
2.  **Yahoo Finance:** Cotações de ativos globais, ETFs e Criptomoedas.
3.  **B3 (Web Scraping):** Índices oficiais da bolsa brasileira (IDIV, IBOV, SMLL, IFIX, etc.) via Selenium.
4.  **Banco Central do Brasil (SGS/Olinda):** Taxas econômicas (SELIC, CDI), Inflação (IPCA), Câmbio (PTAX) e Índices de Títulos Públicos (IMA-B, IRF-M).
5.  **Tesouro Transparente:** Preços históricos de títulos públicos específicos (ex: Tesouro IPCA+ 2035).

## 🛠️ Instalação e Configuração

### Pré-requisitos
*   Python 3.10+
*   Google Chrome instalado (para o scraper da B3)

### Passo a Passo

1.  **Clone o repositório e entre na pasta:**
    ```bash
    git clone <url-do-repositorio>
    cd app_investimentos
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as Variáveis de Ambiente:**
    Crie um arquivo `.env` na raiz do projeto e adicione seu token de acesso à API de histórico:
    ```env
    DLP_TOKEN=seu_token_aqui
    ```

## 🖥️ Como Usar

O ponto de entrada é o script `app/main.py`. Você pode executá-lo de diferentes formas dependendo do objetivo.

### Argumentos Disponíveis

| Argumento | Descrição |
| :--- | :--- |
| `--ativo <COD>` | Analisa um ativo específico da sua carteira (ex: `KLBN11`). |
| `--classe <NOME>` | Analisa uma classe de ativos da sua carteira (ex: `AÇÃO`, `FII`, `R.FIXA`). |
| `--historico <ANOS>` | Define a janela de tempo para análise (ex: `5` para os últimos 5 anos). Se usado sem `--ativo` ou `--classe`, gera apenas um panorama de mercado. |
| `--aporte <VALOR>` | Valor do aporte mensal para simulações (ex: `1000`). |
| `--rebalanceamento <MESES>` | Intervalo em meses para rebalanceamento nas simulações (ex: `6`). |
| `--debug` | Ativa logs detalhados no terminal. |

### Exemplos de Uso

**1. Analisar uma classe de ativos específica (ex: Ações):**
Gera gráficos de TWR, Evolução Patrimonial e Comparativo com Benchmarks para suas ações.
```bash
python app/main.py --classe "AÇÃO"
```

**2. Analisar um ativo específico com recorte de tempo:**
Analisa apenas o ativo `PETR4` nos últimos 2 anos.
```bash
python app/main.py --ativo PETR4 --historico 2
```

**3. Panorama de Mercado (Modo Standalone):**
Gera gráficos comparativos de todos os benchmarks configurados e carteiras sintéticas para os últimos 10 anos, sem ler dados da sua carteira pessoal.
```bash
python app/main.py --historico 10
```

**4. Simulação de Investimentos:**
Simula como teriam performado diversas carteiras teóricas (definidas em `config.py`) nos últimos 10 anos, considerando aportes de R$ 2.000,00 e rebalanceamento semestral.
```bash
python app/main.py --historico 10 --aporte 2000 --rebalanceamento 6
```

## 📂 Estrutura do Projeto

*   **`app/`**: Código fonte principal.
    *   `main.py`: Orquestrador e gerador de gráficos.
    *   `config.py`: Configuração de benchmarks, carteiras sintéticas e listas de exibição.
*   **`utils/`**: Módulos utilitários.
    *   `market_data.py`: Lógica de download, cache e processamento de dados de mercado (YF, B3, BCB).
    *   `logger.py`: Configuração de logs.
*   **`data/`**: Armazenamento local.
    *   `raw/`: Cache de arquivos CSV baixados e processados.
    *   `downloads/`: Pasta temporária para downloads do Selenium.
*   **`reports/`**: Saída do sistema.
    *   `twr/`: Gráficos e CSVs de Rentabilidade (Time-Weighted Return).
    *   `evolucao/`: Gráficos de Evolução Patrimonial e Percentual.
    *   `risco/`: Gráficos de Risco x Retorno e métricas.
    *   `simulacao/`: Resultados das simulações de aportes.

## ⚙️ Personalização

Você pode adicionar novos benchmarks ou criar novas carteiras teóricas editando o arquivo `app/config.py`:

*   **`BENCHMARKS_YF`**: Adicione tickers do Yahoo Finance.
*   **`BENCHMARKS_BCB`**: Adicione códigos de séries do Banco Central.
*   **`CARTEIRAS_SINTETICAS`**: Defina combinações de ativos e pesos para simulação.
*   **`BENCHMARKS_EXIBIR`**: Controle quais índices aparecem nos gráficos finais.