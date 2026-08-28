O sistema consome dados de uma API proprietária de histórico de investimentos e cruza com dados de mercado de diversas fontes (Yahoo Finance, B3, Banco Central, Tesouro Direto) para gerar relatórios detalhados.

## 🚧 Estado do projeto — migração em andamento

O repositório está migrando de **Streamlit + serviços em nuvem** para
**Django + React no Docker LAB**. As duas bases coexistem:

| | Onde | Estado |
|---|---|---|
| **App atual (Streamlit)** | `src/` | Em uso. Única interface de usuário. Instruções abaixo. |
| **Backend novo (Django)** | `backend/` | Fundação pronta — Fases 0 + 1. Ver [`backend/README.md`](backend/README.md). |

Arquitetura, decisões e roteiro: [`docs/`](docs/README.md) — o
[documento mestre](docs/consolidacao_arquitetura_e_migracao.md) e os
[ADRs](docs/adr/README.md). Desenvolvimento na branch `migracao-django`.

## 📝 Changelog (Recente)

### v3.5.0 (Interface Premium e Experiência do Usuário)
- ✨ **Reformulação Visual Premium:** Interface renovada com estética moderna, *glassmorphism* e cards de informação.
- 🌓 **Theme Engine:** Suporte nativo a **Modo Escuro** (Padrão) e **Modo Claro** com alternância em tempo real.
- 🚀 **Dashboard Executivo:** Tela inicial transformada em um resumo de performance com KPIs e atalhos.
- 📊 **Headers Padronizados:** Todas as páginas agora exibem métricas críticas (TWR, Drawdown, Volatilidade) no topo.
- 💄 **Componentes UI:** Implementação de `theme.py` e `headers.py` para garantir consistência visual.
- 🐛 **Estabilidade:** Correção de IDs duplicados e chamadas redundantes na ferramenta de migração.

### v3.4.0 (Redis como Single Source of Truth)
- ♻️ **Arquitetura Modular:** Separação total entre Interface (UI), Lógica Financeira (Engine) e Acesso a Dados (Data).
- ☁️ **Redis SSOT:** Redis como fonte única de verdade para todos os dados de mercado (B3, YF, BCB, TD).
- 🚀 **Script de Sincronização Total:** `scripts/sync_to_redis.py` com `--download` forçado, pré-scan de itens e relatórios detalhados.
- 🚀 **B3 Crawler 2.0:** Novo extrator paralelo (3 workers) com atualização incremental inteligente (sem Selenium no runtime).
- 📱 **Nova Interface Streamlit:** Dashboard migrado de script único para aplicação multipáginas (`src/ui/app.py`).
- 📂 **Estrutura Profissional:** Organização seguindo padrões modernos de projetos Python (`src/`, `tests/`, `scripts/`).
- ⚡ **Performance:** Integração total do Redis na camada de dados, priorizando o cache em nuvem sobre arquivos locais.
- 🛠️ **Resiliência:** Cache com tratamento automático de MultiIndex e armazenamento permanente de metadados.
- 🔒 **Logging Global:** Rastreabilidade completa via `st.session_state` salvando em `log/main.log`.
- 🧹 **Limpeza de Débito Técnico:** Remoção de pastas legadas, dados pessoais e dependências pesadas.

### v3.2.0 (Autenticação e Cache em Nuvem - Fase 2)
- 🔐 **Autenticação Multi-Usuário:** Integração com Google OAuth2 (`st.login`) e proteção de rotas.
- 🛡️ **Segurança Avançada:** Armazenamento de tokens DLP com criptografia AES-256 (Fernet) em repouso.
- ☁️ **Persistência Cloud:** Uso de Supabase para perfis de usuário e Upstash Redis para cache global de mercado.
- 🔧 **Página de Configurações:** Interface para o usuário gerenciar seu próprio token DLP de forma segura.

### v3.3.0 (Planejamento de Migração e Eficiência Fiscal)
- ✨ **Planejador de Migração:** Implementa ferramenta para transição gradual de carteira respeitando o limite de isenção de IR (R$ 20k/mês).
- 🐛 **Correção de API:** Corrige erro de variável indefinida (`NameError`) ao buscar resumo da carteira no módulo `market_data`.
- 💄 **Melhorias de UI:** Ajuste de contraste e cores nos componentes de métrica para garantir legibilidade em diferentes temas.

### v3.2.1 (Ajustes de Telemetria e Estabilidade)
- 🐛 **Correção de Proxy:** Corrigido erro de variável indefinida ao buscar índices via Yahoo Finance.
- 🔍 **Telemetria de Mercado:** Inclusão de nomes amigáveis nos logs de busca de benchmarks (BCB e YF).
- ⚙️ **Robustez:** Padronização de assinaturas de funções na camada de dados de mercado.

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

## 🛡️ Arquitetura e Segurança (Fase 2)

O sistema utiliza uma pilha moderna de serviços em nuvem para garantir escalabilidade e segurança:

*   **Google OAuth2:** Utilizado para autenticação de identidade. O usuário deve estar logado em sua conta Google para acessar o sistema. Utilizamos a integração nativa do Streamlit (`st.login()`).
*   **Supabase (PostgreSQL + RLS):** Atua como base de dados persistente. Armazena as configurações de perfil e as chaves de API dos usuários. Utilizamos **Row Level Security (RLS)** para garantir que um usuário nunca consiga acessar os dados de outro.
*   **Upstash Redis:** Camada de cache distribuído compatível com instâncias *Serverless*. Armazena resultados de cotações do Yahoo Finance e séries do Banco Central para evitar limites de taxa (rate-limits) e acelerar o dashboard.
*   **Fernet (Criptografia AES-256):** Implementa criptografia simétrica no nível da aplicação. O token DLP do usuário é criptografado localmente antes de ser enviado ao Supabase, garantindo que mesmo em caso de vazamento da base de dados, as chaves de acesso permaneçam protegidas.

## 🛠️ Instalação e Configuração

### Pré-requisitos
*   Python 3.12 (o repo padronizou nesta versão para a migração; o app Streamlit roda em 3.10+)
*   Google Chrome (apenas para execução local do script de cache da B3)

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

4.  **Configure os Segredos (Streamlit Secrets):**
    No desenvolvimento local, crie o arquivo `.streamlit/secrets.toml` com base no arquivo de exemplo fornecido:
    ```bash
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    ```
    Preencha os campos com suas credenciais do **Google Cloud, Supabase e Upstash Redis**.

5.  **Gere sua chave de criptografia:**
    Execute o comando abaixo no terminal para gerar uma chave Fernet e insira no campo `fernet_key` do seu `secrets.toml`:
    ```bash
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    ```

## 🖥️ Como Usar

O sistema agora é uma aplicação **Streamlit**. O ponto de entrada principal é o `src/ui/app.py`.

### Executando o Web App

Para iniciar o dashboard interativo:
```bash
streamlit run src/ui/app.py
```

### Atualizando Cache da B3 (Local-only)

Para baixar novos dados da B3 de forma otimizada e paralela:
```bash
python scripts/update_b3_cache.py
```

## 📂 Estrutura do Projeto

*   **`src/engine/`**: Lógica de cálculo (TWR, IRR, Métricas de Risco).
*   **`src/data/`**: Camada de dados, configurações de benchmarks e fontes (DLP, YF, BCB).
*   **`src/ui/`**: Interface Streamlit (Páginas, Componentes e Temas).
*   **`data/static/`**: Histórico consolidado de índices B3 (mantido via script/CI).
*   **`scripts/`**: Utilitários de manutenção (Update B3 Cache).
*   **`tests/`**: Testes automatizados (pytest).
*   **`log/`**: Registros de execução.

## ⚙️ Personalização

Benchmarks e carteiras sintéticas são configurados em `src/data/benchmarks_config.py`:

*   **`CATALOGO_YF` / `CATALOGO_B3` / `CATALOGO_BCB` / `CATALOGO_TD` / `CATALOGO_CRYPTO`**: catálogos de fontes disponíveis (nome → ticker YF / código de série BCB / etc.). Edite para adicionar novas fontes.
*   **`BENCHMARKS_ATIVOS`**: lista unificada do que é calculado e exibido nos gráficos. Cada item é uma **string** (nome presente nos catálogos ou derivado, ex.: `'S&P 500 BRL'`, `'IPCA + 6%'`) ou um **dicionário** de carteira sintética (`{'nome': ..., 'composicao': {...}}`). Comente a linha para ocultar o item.

## 📝 Créditos

Desenvolvido por **Bruno Dias**.