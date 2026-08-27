# Documento Mestre de Consolidação Arquitetural e Roteiro de Migração
## Projeto: `app-investimentos` | Ambiente: Docker LAB | Branch: `lab-dev`

> **Data de Consolidação:** 2026-08-27  
> **Status:** Aprovado para Execução  
> **Objetivo:** Documento único e definitivo que consolida o funcionamento da aplicação, histórico de dependências externas, decisões de arquitetura e o roteiro passo a passo de migração para **PostgreSQL Local + Redis Local + Backend Django Ninja + Frontend React SPA**.

---

## 1. O que é a Aplicação `app-investimentos`

O `app-investimentos` é uma plataforma de inteligência analítica para consolidação e acompanhamento de carteiras de investimentos, comparação de performance contra múltiplos benchmarks de mercado e avaliação de risco financeiro.

### 1.1. Principais Funcionalidades de Negócio:
- **Consolidação de Carteira Pessoal:** Integração com a API DLP (`dlombelloplanilhas.com`) para obter histórico de movimentações, ativos em custódia, valores investidos e proventos recebidos.
- **Cálculo de Rentabilidade TWR (Time-Weighted Return):** Cálculo de retorno ponderado pelo tempo, normalizado em base 100 para comparação justa e isenta de distorções causadas por aportes/resgates.
- **Cálculo de TIR / XIRR (Taxa Interna de Retorno):** Algoritmo numérico (Newton-Raphson) para apurar a rentabilidade real do dinheiro investido.
- **Simulação de Aportes (*Shadow Portfolio*):** Simulação comparativa calculando quanto o investidor teria de patrimônio se todos os aportes históricos tivessem sido alocados nos benchmarks (CDI, IBOV, S&P 500, etc.).
- **Métricas Avançadas de Risco e Volatilidade:**
  - Volatilidade histórica e móvel anualizada (janela de 252 dias úteis).
  - Sharpe Ratio histórico e móvel (utilizando a série da SELIC diária do BCB como taxa livre de risco).
  - Drawdown acumulado e perdas máximas (*peak-to-trough*).
  - Gráfico de dispersão Risco (Volatilidade) x Retorno (CAGR).
  - Tabela resumo de rentabilidade ano a ano.

### 1.2. Provedores e Fontes de Dados de Mercado:
1. **Banco Central do Brasil (`python-bcb` SGS):**
   - CDI Diário (Código 12)
   - SELIC Diária (Código 11)
   - IPCA Mensal (Código 433)
   - Séries sintéticas calculadas: "IPCA + X%", "X% do CDI", "CDI + X%".
2. **Tesouro Direto (Tesouro Transparente):**
   - Histórico oficial diário de Preços Unitários (PU Base Manhã) e taxas de títulos públicos federais (Tesouro IPCA+, Selic e Prefixados).
3. **Yahoo Finance (`yfinance`):**
   - Índices globais e locais (`^BVSP`, `^GSPC`, `^IXIC`), Ações B3 (`.SA`), ETFs Internacionais (`.L`, `.US`), REITs e Criptoativos.
4. **Catálogo de Ativos (`ativos.csv`):**
   - Mapeamento de tickers, siglas, classes de ativos (`ACAO`, `FII`, `FI_INFRA`, `ETF`, `STOCK`, `BDR`) e bolsas de negociação (`BVMF`, `LON`).

---

## 2. Diagnóstico do Estado Anterior e Dependências Externas

Na versão anterior (baseada na branch `dev` e interface Streamlit), identificamos os seguintes gargalos e dependências que precisavam ser eliminadas para permitir uma operação 100% local, autônoma e de alto desempenho:

| Dependência Anterior | Papel no Projeto | Situação Anterior | Decisão / Nova Arquitetura no LAB |
|---|---|---|---|
| **Supabase Cloud** | Armazenamento de tokens DLP dos usuários | Nuvem (REST API / Service Role) | **Migrado 100%** para o PostgreSQL 16 local (`lab-postgres`). |
| **Upstash Redis** | Cache de cotações e séries históricas | Nuvem (HTTP REST Upstash) | **Migrado 100%** para o Redis 7 local (`lab-redis` via TCP padrão). |
| **Google OAuth2** | Autenticação de usuários | Google Cloud Console OAuth | **Autenticação Local / Django Auth** nativa no LAB. |
| **Streamlit UI** | Camada de visualização gráfica | Acoplada, gerando PNGs estáticos em Matplotlib | **Desacoplado 100%**: Backend em Django Ninja (API REST JSON) + Frontend React SPA (Apache ECharts). |
| **Ingestão Sob Demanda** | Busca de dados BCB, Tesouro e Yahoo | Bloqueava a tela na primeira carga | **Rotina de Ingestão Agendada** (`manage.py ingest_market_data`) em background. |

---

## 3. Decisões Arquiteturais Consolidadas

```mermaid
flowchart TD
    subgraph Client_Browser [Navegador do Usuário]
        React_UI["Frontend SPA (React 18 + Vite + Tailwind + ECharts)<br/><i>Interface rica, responsiva e com gráficos interativos</i>"]
    end

    subgraph Reverse_Proxy [Roteamento Central]
        Nginx["lab-nginx Reverse Proxy (:80 / :8080)"]
    end

    subgraph Django_Backend [Backend Django 5 + Django Ninja (:8000)]
        API["Django Ninja Router (/api/v1/...)"]
        Swagger["Swagger UI Docs (/api/docs)"]
        Admin["Django Admin (/admin)"]
        Engine["Engine Financeiro Python (TWR, TIR, Sharpe)"]
        Ingestion["Management Command (ingest_market_data)"]
    end

    subgraph Local_Storage [Persistência Docker LAB]
        Postgres[("lab-postgres (PostgreSQL 16)")]
        Redis[("lab-redis (Redis 7 Alpine)")]
    end

    subgraph External_Sources [Fontes Externas Obrigatórias]
        DLP["DLP API (Carteira Pessoal)"]
        BCB["Banco Central SGS"]
        TD["Tesouro Transparente"]
        YF["Yahoo Finance"]
    end

    React_UI -->|"HTTP / REST (JSON)"| Nginx
    Nginx -->|"/api/* e /admin/*"| Django_Backend
    Nginx -->|"/*"| React_UI

    API --> Engine
    Engine --> Postgres
    Engine --> Redis
    Engine --> DLP

    Ingestion --> BCB
    Ingestion --> TD
    Ingestion --> YF
    Ingestion --> Postgres
    Ingestion --> Redis
```

### 3.1. Por que Django + Django Ninja no Backend?
- **Reaproveitamento de 100% do código Python:** Os módulos de cálculo financeiro (`financial_report.py`, `twr.py`, `irr.py`, `metrics.py`, `market_data.py`) são mantidos e importados diretamente.
- **ORM e Migrações Robustas:** Criação e manutenção automática do schema no PostgreSQL (`lab-postgres`).
- **Painel Administrativo Nativo (`/admin`):** Gestão de usuários, tokens criptografados e inspeção de dados sem esforço extra.
- **API Moderna com Django Ninja:** Sintaxe enxuta, validação de tipos com Pydantic, respostas rápidas em JSON e documentação Swagger interativa automática em `/api/docs`.

### 3.2. Por que React (com Vite, Tailwind e Apache ECharts) no Frontend?
- **Interatividade Total em Gráficos Financeiros:** O **Apache ECharts** renderiza gráficos no cliente via Canvas/SVG com zoom pelo scroll do mouse, slider temporal (*dataZoom*), tooltips ricos e sincronizados, e opção de ligar/desligar ativos na legenda sem recarregar a página.
- **Desacoplamento Total:** O frontend é uma SPA autônoma que consome a API REST via JSON.

### 3.3. Por que Rotina de Ingestão Agendada?
- Transforma a experiência do usuário em instantânea. Os dados do BCB (CDI, SELIC, IPCA), Tesouro Direto e Yahoo Finance são atualizados em segundo plano (ex: às 06:00 e às 19:00 de dias úteis) e pré-cacheados no Redis e PostgreSQL.

---

## 4. Estrutura Completa de Diretórios e Arquivos

Todas as alterações serão estruturadas dentro do repositório `/data/projetos/app-investimentos` na branch **`lab-dev`**:

```
/data/projetos/app-investimentos/
│
├── docker-compose.yml                    # Orquestração do Backend, Redis, Postgres e Nginx
│
├── backend/                              # [NOVO] Backend Django 5 + Django Ninja
│   ├── Dockerfile                        # Container Docker Python para o Backend
│   ├── requirements.txt                  # Dependências (Django, Ninja, psycopg2, redis, pandas, etc.)
│   ├── manage.py                         # Entrypoint CLI do Django
│   │
│   ├── config/                           # Configurações do Projeto Django
│   │   ├── __init__.py
│   │   ├── settings.py                   # Banco Postgres, Cache Redis, CORS, Chaves Fernet
│   │   ├── urls.py                       # Roteamento principal (/api/, /admin/)
│   │   └── wsgi.py
│   │
│   ├── apps/
│   │   ├── core/                         # Modelos base, Criptografia e Admin
│   │   │   ├── models.py                 # UserToken (criptografado), MarketSeries
│   │   │   ├── admin.py                  # Configurações do Django Admin
│   │   │   └── security.py               # Algoritmo AES-256 Fernet para tokens
│   │   │
│   │   ├── api/                          # Endpoints REST Django Ninja
│   │   │   ├── router.py                 # Rotas da API (/api/v1/...)
│   │   │   └── schemas.py                # Schemas Pydantic de entrada e saída
│   │   │
│   │   └── market/                       # Ingestão e Integração de Mercado
│   │       ├── services.py               # Coleta de dados (BCB, Tesouro, Yahoo, DLP)
│   │       └── management/commands/
│   │           └── ingest_market_data.py # Comando: python manage.py ingest_market_data
│   │
│   └── engine/                           # [MIGRADO] Núcleo de Cálculo Financeiro
│       ├── financial_report.py           # Construtor de relatórios e datasets unificados
│       ├── twr.py                        # Cálculo de TWR Base 100
│       ├── irr.py                        # Cálculo de TIR (XIRR via Newton-Raphson)
│       └── metrics.py                    # Volatilidade, Sharpe Ratio, CAGR, Drawdown
│
└── frontend/                             # [NOVO] Frontend SPA em React 18
    ├── Dockerfile                        # Multi-stage build Nginx para produção
    ├── package.json                      # React 18, Vite, ECharts, Tailwind, Lucide Icons, Axios
    ├── vite.config.js                    # Configuração Vite com proxy para o Backend
    ├── tailwind.config.js                # Design System com tema escuro (Dark Theme)
    ├── postcss.config.js
    │
    └── src/
        ├── main.jsx                      # Ponto de entrada da aplicação React
        ├── App.jsx                       # Layout principal do Dashboard
        ├── index.css                     # Estilos globais e Tailwind
        │
        ├── api/
        │   └── client.js                 # Cliente HTTP Axios centralizado
        │
        └── components/                   # Componentes Modulares de UI
            ├── Header.jsx                # Cabeçalho com status, perfil e token DLP
            ├── SummaryCards.jsx          # Cards: Rentabilidade, CAGR, Volatilidade, Sharpe
            ├── FiltersBar.jsx            # Filtros de período (1A, 2A, 5A, Tudo) e benchmarks
            ├── YearlyTable.jsx           # Tabela de rentabilidade anual consolidada
            │
            └── charts/                   # Componentes Interativos com Apache ECharts
                ├── TwrChart.jsx          # Gráfico de evolução da rentabilidade (Base 100)
                ├── DrawdownChart.jsx     # Gráfico de queda máxima acumulada (Drawdown)
                ├── RollingSharpeChart.jsx# Gráfico de Sharpe Ratio móvel
                ├── ShadowChart.jsx       # Simulação de Aportes (Shadow Portfolio)
                └── RiskReturnChart.jsx   # Gráfico de dispersão Risco x Retorno
```

---

## 5. Roteiro Passo a Passo de Implementação (Roadmap)

### Fase 1: Infraestrutura de Banco e Backend Django (Imediata)
1. **Configuração de Dependências e Ambiente:**
   - Criar `backend/requirements.txt` com todas as bibliotecas necessárias.
   - Criar `backend/config/settings.py` configurando:
     - PostgreSQL Local (`POSTGRES_HOST=lab-postgres`, porta `5432`, banco `banco_lab`).
     - Redis Local (`django-redis` em `lab-redis:6379`).
     - CORS liberado para o Frontend React.
     - Chave de segurança `FERNET_KEY`.
2. **Modelos de Dados e Migrações:**
   - Implementar `apps/core/models.py` (`UserToken` com criptografia e `MarketSeries` para histórico de cotações).
   - Executar `python manage.py makemigrations` e `python manage.py migrate`.
   - Registrar modelos no `apps/core/admin.py` e criar superusuário para o Django Admin.
3. **Migração do Engine Financeiro:**
   - Adaptar os módulos de cálculo existentes para `backend/engine/`, garantindo compatibilidade com o formato de saída JSON.

### Fase 2: Serviços de Mercado e Rotina de Ingestão
1. **Implementação do `apps/market/services.py`:**
   - Módulo para busca da carteira na API DLP usando token descriptografado.
   - Módulo para download e parse das séries do BCB SGS (CDI, SELIC, IPCA).
   - Módulo para download do arquivo mestre do Tesouro Direto (`PrecoTaxaTesouroDireto.csv`).
   - Módulo para busca de cotações de fechamento no Yahoo Finance (`yfinance`).
2. **Comando de Ingestão (`manage.py ingest_market_data`):**
   - Criar o comando CLI para atualização em lote e aquecimento do cache Redis.
   - Testar execução do comando e validar persistência no Postgres.

### Fase 3: Endpoints da API REST (Django Ninja)
1. **Construção das Rotas da API em `apps/api/router.py`:**
   - `POST /api/v1/auth/token`: Salvar e criptografar token DLP do usuário.
   - `GET /api/v1/auth/token-status`: Verificar se o usuário possui token configurado.
   - `GET /api/v1/portfolio/summary`: Retornar métricas consolidadas (Patrimônio, TWR, CAGR, Sharpe, Volatilidade).
   - `GET /api/v1/portfolio/twr-evolution`: Retornar séries temporais formatadas para o ECharts.
   - `GET /api/v1/portfolio/drawdown`: Retornar série histórica de Drawdown.
   - `GET /api/v1/portfolio/rolling-metrics`: Retornar Volatilidade e Sharpe móveis.
   - `GET /api/v1/portfolio/shadow-simulation`: Retornar simulação de aportes comparativa.
   - `GET /api/v1/portfolio/yearly-summary`: Retornar rentabilidade ano a ano em formato tabular.
   - `GET /api/v1/benchmarks/catalog`: Retornar lista de benchmarks disponíveis para seleção.
2. **Validação no Swagger:**
   - Testar todas as rotas interativamente em `http://localhost:8000/api/docs`.

### Fase 4: Frontend React SPA (Vite + Tailwind + Apache ECharts)
1. **Inicialização do Projeto React:**
   - Configurar `package.json`, `vite.config.js`, `tailwind.config.js`.
   - Criar cliente HTTP `api/client.js`.
2. **Construção dos Componentes de UI:**
   - `Header.jsx`: Input para configurar o Token DLP e status de conexão.
   - `FiltersBar.jsx`: Botões de período rápido (1A, 2A, 5A, Tudo, Customizado) e multiselect de Benchmarks.
   - `SummaryCards.jsx`: Cards com destaques visuais (KPIs financeiros).
   - `TwrChart.jsx`, `DrawdownChart.jsx`, `ShadowChart.jsx`, `RiskReturnChart.jsx`, `RollingSharpeChart.jsx`: Gráficos interativos em Apache ECharts.
   - `YearlyTable.jsx`: Tabela estilizada de rentabilidade ano a ano.
3. **Integração e Testes de UX:**
   - Conectar o estado do React à API Django Ninja, validando filtros dinâmicos e zoom nos gráficos.

### Fase 5: Dockerização e Unificação no Docker LAB
1. **Dockerfiles e Docker Compose:**
   - Criar `backend/Dockerfile` e `frontend/Dockerfile`.
   - Atualizar `docker-compose.yml` consolidando:
     - `lab-postgres`
     - `lab-redis`
     - `lab-backend` (Django Gunicorn)
     - `lab-frontend` (Nginx servindo build estático)
     - `lab-nginx` (Proxy reverso unificado na porta 80/8080)
2. **Validação Final de Ponta a Ponta.**

---

## 6. Plano de Verificação e Testes

### 6.1. Testes Automatizados e Comandos CLI:
```bash
# 1. Validação de Migrações do Django Backend
cd /data/projetos/app-investimentos/backend
python manage.py check
python manage.py migrate

# 2. Teste de Conexão com Postgres e Redis
python manage.py shell -c "from apps.core.models import UserToken; print('Postgres Conectado'); from django.core.cache import cache; cache.set('teste', 123, 10); print('Redis Conectado:', cache.get('teste'))"

# 3. Teste da Rotina de Ingestão
python manage.py ingest_market_data

# 4. Validação de Build do Frontend React
cd /data/projetos/app-investimentos/frontend
npm install
npm run build
```

### 6.2. Testes Manuais de Usuário:
- [ ] Acessar `http://localhost:8000/admin` e efetuar login administrativo.
- [ ] Acessar `http://localhost:8000/api/docs` e validar execução de endpoints com resposta JSON 200 OK.
- [ ] Acessar a interface React no navegador, informar token DLP e visualizar os gráficos carregando instantaneamente.
- [ ] Testar controles de zoom e seleção de benchmarks no gráfico ECharts.
- [ ] Confirmar que nenhum dado ou token vaza em logs ou respostas não autorizadas.
