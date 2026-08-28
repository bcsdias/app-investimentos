# Documento Mestre de Consolidação Arquitetural e Roteiro de Migração
## Projeto: `app-investimentos` | Ambiente: Docker LAB | Branch: `migracao-django`

> **Data de Consolidação:** 2026-08-27
> **Status:** Aprovado — base para os ciclos design → plano → implementação por fase
> **Histórico:** revisões e drafts anteriores em [`historico/`](historico/README.md)
> **Objetivo:** Documento único e definitivo que consolida o funcionamento da aplicação,
> histórico de dependências externas, decisões de arquitetura e o roteiro passo a passo de
> migração para **PostgreSQL Local + Redis Local + Backend Django Ninja + Frontend React SPA**.

> **O que esta revisão corrigiu em relação aos drafts anteriores ([`historico/`](historico/README.md)):**
> 1. Diagnóstico do estado atual reescrito com base no código real (o repo hoje é 100% Streamlit;
>    não existe nada de Django/Docker/LAB ainda).
> 2. `market_data.py` reclassificado: é **camada de dados** (`src/data/sources/`, 796 linhas),
>    não módulo do engine — vai para `apps/market/`, e precisa ser **quebrado**, não copiado.
> 3. Refatoração do `financial_report.py` promovida a **fase própria e explícita** (2.5):
>    hoje mistura cálculo + Matplotlib nos 7 métodos `plot_*`.
> 4. Corrigida a premissa de "tudo instantâneo": a carteira do usuário (DLP) é buscada **ao vivo**;
>    a ingestão agendada cobre apenas BCB, Tesouro e Yahoo.
> 5. Incluída a **Ferramenta de Migração/Isenção de IR** (`3_migracao_ir.py`, 735 linhas + PDF),
>    ausente na v1, agora no escopo (endpoints + tela React + PDF server-side).
> 6. Definido **multi-usuário**: migração da tabela `user_tokens`, reuso da chave Fernet,
>    autenticação da SPA por sessão Django.
> 7. Definido **Selenium mantido** para índices B3 (Chrome no container de ingestão, nunca no request).
> 8. Adicionadas seções faltantes: auth SPA→API, migração de dados, contrato de benchmarks
>    sintéticos, serialização, testes de paridade, logs em container, CORS.
> 9. Novas fases: **0** (Postgres/Redis primeiro), **2.5** (refatorar engine), **6** (validação de paridade).

---

## 1. O que é a Aplicação `app-investimentos`

O `app-investimentos` é uma plataforma de inteligência analítica para consolidação e
acompanhamento de carteiras de investimentos, comparação de performance contra múltiplos
benchmarks de mercado e avaliação de risco financeiro.

### 1.1. Principais Funcionalidades de Negócio

- **Consolidação de Carteira Pessoal:** Integração com a API DLP
  (`users.dlombelloplanilhas.com`) para obter histórico de movimentações, ativos em custódia,
  valores investidos e proventos recebidos.
- **Cálculo de Rentabilidade TWR (Time-Weighted Return):** Retorno ponderado pelo tempo,
  normalizado em base 100, para comparação justa e isenta de distorções causadas por
  aportes/resgates.
- **Cálculo de TIR / XIRR (Taxa Interna de Retorno):** Algoritmo numérico (Newton-Raphson,
  até 50 iterações) para apurar a rentabilidade real do dinheiro investido.
- **Simulação de Aportes (*Shadow Portfolio*):** Simulação comparativa calculando quanto o
  investidor teria de patrimônio se todos os aportes históricos tivessem sido alocados nos
  benchmarks (CDI, IBOV, S&P 500, etc.).
- **Métricas Avançadas de Risco e Volatilidade:**
  - Volatilidade histórica e móvel anualizada (janela de 252 dias úteis).
  - Sharpe Ratio histórico e móvel (série da SELIC diária do BCB como taxa livre de risco;
    hoje há também um `rf_constant=0.10` como fallback no código).
  - Drawdown acumulado e perdas máximas (*peak-to-trough*).
  - Gráfico de dispersão Risco (Volatilidade) x Retorno (CAGR).
  - Tabela resumo de rentabilidade ano a ano.
- **Planejador de Migração de Carteira com Isenção de IR:** Ferramenta que lê a carteira atual
  (endpoint `/resumo` da DLP), recebe uma alocação-alvo (hoje IMAB11 / VWRA11 / BITH11),
  e calcula um plano de **compras mensais rebalanceadas** respeitando o limite de
  **R$ 20.000/mês** de vendas isentas de IR, com **exportação em PDF** (biblioteca `fpdf2`).
  *(Feature mais recente do repositório — últimos commits de `origin/dev`.)*

### 1.2. Provedores e Fontes de Dados de Mercado

1. **Banco Central do Brasil (`python-bcb`, módulo `sgs`):**
   - CDI Diário (código SGS **12**), SELIC Diária (código **11**), IPCA Mensal (código **433**).
   - Família ANBIMA IMA-B / IRF-M / IMA-S (códigos 12461–12469), com **fallback** para ETFs
     equivalentes via Yahoo quando a série é descontinuada
     (12466→`IMAB11.SA`, 12461→`IRFM11.SA`, 12467→`B5P211.SA`, 12468→`IB5M11.SA`, 12469→`LFTS11.SA`).
   - PTAX USD/BRL via API Olinda do BCB (`buscar_dolar_bcb`), usada para converter ativos
     internacionais em BRL.
   - Séries sintéticas calculadas por regex: `"IPCA + X%"`, `"X% do CDI"`, `"CDI + X%"`.
2. **Tesouro Direto (Tesouro Transparente):**
   - Download do CSV nacional `PrecoTaxaTesouroDireto.csv`, filtrado por tipo de título e
     data de vencimento; série `PU Base Manhã`.
   - *Observação:* `CATALOGO_TD` está **vazio** hoje (`{}`) — caminho dormente, mas funcional.
3. **Yahoo Finance (`yfinance`):**
   - Índices globais e locais (`^BVSP`, `^GSPC`, `^IXIC`), ações B3 (`.SA`), ETFs
     internacionais (`.L`, `.US`), REITs e criptoativos.
   - `yf.download(ticker, start='1995-01-01', auto_adjust=True, progress=False)`.
4. **B3 (scraping via Selenium + headless Chrome):**
   - Índices de evolução diária (IFIX, SMLL, IDIV, etc.) raspados do site
     `sistemaswebb3-listados.b3.com.br`.
   - Fonte alternativa: CSVs estáticos em `data/static/<TICKER>_all.csv`, gerados **mensalmente**
     pela GitHub Action `.github/workflows/update_b3.yml` e lidos por
     `src/data/sources/b3_source.py::get_b3_index` (função **já existente, hoje não utilizada**).
5. **Catálogo de Ativos (`ativos.csv`):**
   - Arquivo `;`-separado, UTF-8 com BOM. Colunas: `sigla`, `market_cod` (`EXCHANGE:TICKER`),
     `classe`, `razao_social`. ~7.500 linhas.
   - Classes presentes: `STOCK`, `CRIPTO`, `ETF_US`, `BDR`, `ACAO`, `FII`, `REIT`, `ETF_GB`,
     `ETF`, `ETF RF`, `FI_INFRA`, `INDICE`, `FIP_IE`, entre outras.
   - **Existe duplicado** (`/ativos.csv` na raiz e `/data/ativos.csv`); os scripts leem `data/ativos.csv`.

---

## 2. Diagnóstico do Estado Atual (auditado em 2026-08-27)

> A v1 descrevia um "estado anterior" baseado na branch `dev` + Streamlit. Esta seção descreve
> o **estado real do repositório hoje**, na branch `migracao-django` (publicada em
> `origin/migracao-django`; `main` e `dev` intactas).
>
> **Fases 0 + 1 concluídas em 2026-08-28** — ver §2.0. As decisões travadas estão em
> [`adr/`](adr/README.md); a orientação de trabalho no backend em [`../backend/CLAUDE.md`](../backend/CLAUDE.md).

### 2.0. Já implementado — Fases 0 + 1 (2026-08-28)

| Camada | Local | Situação |
|---|---|---|
| **Infra LAB** | `/data/projetos/lab-postgres/`, `/data/projetos/lab-redis/` (fora do repo) | `lab-postgres` + `lab-redis` compartilhados, *healthy*, na rede externa `lab-net`. `lab-nginx` de teste removido. [ADR 0001](adr/0001-infra-lab-compartilhada.md). |
| **Database do app** | `lab-postgres` → database + role `appinvest` (`CREATEDB`) | Criado por `backend/scripts/bootstrap_db.sh` (idempotente, versionado). Nenhuma tabela do app no `banco_lab`. [ADR 0006](adr/0006-prefixo-lab-containers-appinvest-codigo.md). |
| **Projeto Django** | `backend/config/` | `settings.py` dual host/container: DB `appinvest`, cache `django-redis`, Fernet, logging stdout ([ADR 0008](adr/0008-logging-stdout-apenas.md)), TZ `America/Sao_Paulo`. `urls.py` só `/admin/` + stub `/api/` ([ADR 0009](adr/0009-django-ninja-stub-desde-fase-1.md)). |
| **Usuário** | `backend/apps/accounts/` | `User(AbstractUser)` customizado; `AUTH_USER_MODEL = "accounts.User"`. [ADR 0002](adr/0002-modelo-user-customizado-dia-1.md). |
| **Cripto** | `backend/apps/core/security.py` | `encrypt()` / `decrypt()` Fernet sobre `settings.FERNET_KEY` — **mesma chave do app legado**, compat. testada nos dois sentidos. [ADR 0003](adr/0003-reuso-da-chave-fernet-do-legado.md). |
| **Modelos** | `backend/apps/core/models.py` | `UserToken` (OneToOne, token cifrado em repouso, `set_token`/`get_token`) e `MarketSeries` (`UniqueConstraint(series_key, reference_date)`). Migration `core.0001`. [ADR 0007](adr/0007-modelagem-usertoken-marketseries.md). |
| **Admin** | `backend/apps/core/admin.py` | `UserToken` sem "add", `encrypted_token` read-only; `MarketSeries` com filtro por `source` e navegação por data. |
| **Testes** | `backend/apps/*/tests/` + `backend/pytest.ini` | `pytest-django`, 15 testes verdes (usuário, security, modelos, smoke de DB/cache). `--reuse-db`. |
| **Scripts** | `backend/scripts/` | `bootstrap_db.sh` (cria role/database) e `rotate_db_password.sh` (rotaciona a senha). |
| **Docs** | `backend/README.md`, `backend/CLAUDE.md`, `docs/adr/` | Setup/rodar, convenções do agente, e ADRs. |

> Driver Postgres: `psycopg[binary]` 3.x ([ADR 0004](adr/0004-psycopg-binary-3.md)). Sem
> `django-cors-headers` ([ADR 0005](adr/0005-sem-django-cors-headers.md)). Env via
> `python-dotenv` puro ([ADR 0010](adr/0010-python-dotenv-puro.md)).

### 2.1. O que existe — legado Streamlit (coexiste até a Fase 6)

| Camada | Local | Situação |
|---|---|---|
| **UI** | `src/ui/app.py` + `src/ui/pages/1..4` + `src/ui/components/{theme,headers,sidebar,charts}.py` | App **Streamlit multipágina** (v3.5.0). Única interface **de usuário** ainda. |
| **Engine de cálculo** | `src/engine/{twr,irr,metrics}.py` | **NumPy/pandas puros, zero acoplamento a framework, com testes unitários** (`tests/`, ~150 linhas). Reuso direto. |
| **Orquestrador** | `src/engine/financial_report.py` (417 linhas) | Classe `FinancialReport`. **Importa `matplotlib` no topo**; 7 métodos `plot_*` **misturam cálculo + plotagem** e retornam `(fig, DataFrame)`. |
| **Camada de dados** | `src/data/sources/market_data.py` (**796 linhas**) | Um único arquivo com: cliente DLP, `yfinance`, BCB (`python-bcb`), Tesouro (CSV), PTAX Olinda, **e scraping B3 via Selenium**. |
| **Cache** | `src/data/cache.py` (130 linhas) | 2 camadas (dict em memória + **Upstash Redis REST**), envelope de serialização pandas próprio, fallback silencioso para memória. Já lê `UPSTASH_REDIS_REST_URL/TOKEN` via `load_dotenv()`. L3 adicional de CSVs em `data/raw/`. |
| **Aquecedor de cache** | `scripts/sync_to_redis.py` (295 linhas) | Já empurra séries de mercado para o Redis com retry/backoff. Embrião da ingestão agendada. |
| **Tokens DLP** | `src/data/user_store.py` (85 linhas) | **Fernet (AES-256)** → **Supabase**, tabela `user_tokens` (PK = e-mail), usando a **service_role key** (RLS ignorada de propósito). |
| **Auth** | `src/ui/app.py` | **OIDC nativo do Streamlit** (`st.login()` Google) + **bypass "Modo Dev"** que dá acesso total sem credencial (grava token sob o e-mail literal `"Usuário"`). |
| **Planejador de IR** | `src/ui/pages/3_migracao_ir.py` (**735 linhas**) | `calcular_compras_rebalanceadas`, leitura de carteira via `buscar_resumo_carteira`, `obter_preco_atual` (yfinance), **`gerar_pdf_plano` (fpdf2)**. É praticamente uma sub-app, com input de token próprio. |
| **CI** | `.github/workflows/update_b3.yml` | Único workflow. Scraper B3 mensal (`cron: '0 0 1 * *'`) que commita CSVs em `data/static/`. |
| **Config** | `.streamlit/secrets.toml` (git-ignored; só `.example` commitado) | Blocos `[auth]` (Google OIDC), `[supabase]`, `[upstash_redis]`, `[security].fernet_key`. |

### 2.2. O que **não** existe ainda

- **Sem serviços de mercado** (`apps/market/`): a ingestão agendada (`ingest_market_data`),
  o port de `src/data/sources/market_data.py` e a gravação em `MarketSeries` são a Fase 2.
- **Sem API HTTP**: `django-ninja` está instalado mas é só stub em `config/urls.py` (Fase 3).
- **Sem `frontend/`, sem `package.json`** — zero JS/Node no repositório (Fase 4).
- **Sem Dockerfiles do app** nem compose final/nginx único — o backend roda no host (Fase 5).
  Só os containers de **infra** (`lab-postgres`, `lab-redis`) existem.
- **Dados ainda no legado**: os tokens continuam no Supabase e o cache no Upstash; a
  migração das linhas (`user_tokens` → `UserToken`) e a troca do `cache.py` não foram feitas.
- `.devcontainer/devcontainer.json` aponta para `app/web_app_v2.py` (**caminho stale**, não existe mais).

### 2.3. Dívida técnica / lixo a resolver **durante** a migração (não carregar adiante)

- `financial_report.py` referencia `self.logger` (inexistente) no branch de dados vazios — **quebra**.
- `redis.py` na raiz (script solto de 12 linhas) faz *shadow* do pacote `redis`.
- `ativos.csv` duplicado (raiz + `data/`).
- Arquivos de scratch do Gemini/Antigravity commitados (`.gemini/antigravity/.../scratch/*.py`).
- `requirements.txt`: **16 libs sem nenhum pin de versão**; `selenium` e `webdriver-manager`
  são usados mas **não constam** do arquivo (instalados ad-hoc na Action).
- Páginas 2 e 3 importam `from src...` sem patch de `sys.path` → causa do `ModuleNotFoundError`
  registrado em `erros.txt`. (Some com a nova arquitetura, mas vale registrar.)
- `deployment_guide.md` contém caminhos locais Windows stale.

### 2.4. Dependências externas a eliminar

| Dependência | Papel | Situação atual | Nova arquitetura no LAB |
|---|---|---|---|
| **Supabase Cloud** | Armazenamento de tokens DLP (Fernet) | Nuvem (REST, service_role) | **Migrado** para PostgreSQL 16 local (`lab-postgres`), via modelo `UserToken`. **Requer script de migração das linhas existentes** (ver §7.2). |
| **Upstash Redis** | Cache de cotações e séries | Nuvem (HTTP REST) | **Migrado** para Redis 7 local (`lab-redis`, TCP). `cache.py` precisa trocar `upstash_redis` por `redis`/`django-redis`. |
| **Google OAuth2** | Autenticação de usuários | `st.login()` nativo do Streamlit | **Autenticação Django** (login/sessão). SSO Google é **perdido** — ver §7.1 para o trade-off e a alternativa opcional (`django-allauth`). |
| **Streamlit UI** | Visualização | Acoplada; Altair (TWR) + Matplotlib (`st.pyplot`) no resto | **Desacoplado**: Django Ninja (API REST JSON) + React SPA (Apache ECharts). |
| **Ingestão sob demanda** | BCB, Tesouro, Yahoo, **B3 (Selenium)** | Bloqueia a 1ª carga | **Rotina agendada** (`manage.py ingest_market_data`) em background. **A carteira DLP continua sob demanda** (ver §3.4). |

---

## 3. Decisões Arquiteturais Consolidadas

```mermaid
flowchart TD
    subgraph Client_Browser [Navegador do Usuário]
        React_UI["Frontend SPA (React 18 + Vite + Tailwind + ECharts)"]
    end

    subgraph Reverse_Proxy [Roteamento]
        Nginx["lab-nginx (:80 / :8080) — serve build estático + proxy /api e /admin"]
    end

    subgraph Django_Backend [Backend Django 5 + Django Ninja (:8000)]
        Auth["Django Auth (login/sessão via cookie)"]
        API["Django Ninja Router (/api/v1/...)"]
        Swagger["Swagger UI (/api/docs)"]
        Admin["Django Admin (/admin)"]
        Compute["Engine de Cálculo (compute_* puros) — TWR, TIR, Sharpe, Drawdown, Shadow, IR"]
        Services["apps/market/services/* — clientes DLP, BCB, Tesouro, Yahoo, B3"]
        Ingestion["Management Command: ingest_market_data (agendado)"]
    end

    subgraph Local_Storage [Persistência Docker LAB]
        Postgres[("lab-postgres (PostgreSQL 16) — UserToken, MarketSeries, User")]
        Redis[("lab-redis (Redis 7) — cache de séries e respostas")]
    end

    subgraph External_Sources [Fontes Externas]
        DLP["DLP API (Carteira Pessoal — sob demanda)"]
        BCB["Banco Central SGS + Olinda PTAX"]
        TD["Tesouro Transparente"]
        YF["Yahoo Finance"]
        B3["B3 (Selenium — só na ingestão)"]
    end

    React_UI -->|"HTTP / REST (JSON) + cookie de sessão"| Nginx
    Nginx -->|"/api/* e /admin/*"| Django_Backend
    Nginx -->|"/*"| React_UI

    API --> Auth
    API --> Compute
    API --> Services
    Compute --> Postgres
    Compute --> Redis
    Services --> DLP
    Services --> Redis

    Ingestion --> BCB
    Ingestion --> TD
    Ingestion --> YF
    Ingestion --> B3
    Ingestion --> Postgres
    Ingestion --> Redis
```

### 3.1. Por que Django + Django Ninja no Backend

- **Reuso do engine de cálculo puro:** `twr.py`, `irr.py`, `metrics.py` são importados
  diretamente, sem alteração. `financial_report.py` é **refatorado** (fase 2.5) para separar
  cálculo de plotagem.
- **ORM e migrações:** schema versionado no PostgreSQL (`lab-postgres`), incluindo a tabela de
  tokens e o histórico de séries de mercado.
- **Django Admin (`/admin`):** gestão de usuários, tokens criptografados e inspeção das séries
  sem esforço extra. Substitui o bypass "Modo Dev" por usuários reais.
- **Django Ninja:** validação com Pydantic, respostas JSON rápidas, Swagger automático em `/api/docs`.
- **Nota de dimensionamento:** Ninja/FastAPI-style seria suficiente sozinho, mas Admin + ORM +
  auth + migrations do Django agregam valor real para o cenário **multi-usuário** — escolha mantida.

### 3.2. Por que React (Vite + Tailwind + Apache ECharts) no Frontend

- **Interatividade nos gráficos:** ECharts renderiza no cliente com zoom por scroll, slider
  temporal (`dataZoom`), tooltips sincronizados e toggle de séries na legenda sem recarregar.
  Substitui a mistura atual Altair + Matplotlib (`st.pyplot`).
- **Desacoplamento:** SPA autônoma consumindo a API REST.
- **Dimensionamento honesto:** o frontend é a **maior tarefa isolada** do plano — 7 gráficos +
  tabela anual + barra de filtros + UI de token + **tela do planejador de IR**. Fase 4 deve ser
  planejada como o maior bloco, não "uma fase entre cinco".

### 3.3. Por que Rotina de Ingestão Agendada

- Torna instantânea a **parte de mercado**: BCB (CDI, SELIC, IPCA, IMA-B), PTAX, Tesouro,
  Yahoo e índices B3 são atualizados em segundo plano (ex.: 06:00 e 19:00 em dias úteis) e
  gravados em `MarketSeries` (Postgres) + cache (Redis).
- Selenium roda **apenas aqui**, nunca no caminho de request (ver §3.5).

### 3.4. O que a ingestão **não** cobre: a carteira do usuário

- `FinancialReport.fetch_user_portfolio()` → `buscar_historico(token)` chama a **API DLP ao
  vivo, de forma síncrona**, com cache Redis de **TTL 600s**.
- Como o token é **por-usuário**, a ingestão agendada **não tem como pré-aquecer** as carteiras
  de todos os usuários. O primeiro acesso de um usuário (ou após 10 min) **sempre** faz a
  chamada DLP.
- **Implicação de UX:** a promessa de "tudo instantâneo" da v1 vale para os benchmarks, não
  para a carteira. A tela deve exibir estado de carregamento na primeira análise e, se
  desejado, um job assíncrono pode pré-buscar a carteira logo após o login do usuário.

### 3.5. Índices B3: Selenium mantido

- `buscar_dados_b3` (headless Chrome via `selenium` + `webdriver-manager`) é **mantido**.
- O container que executa `ingest_market_data` precisa de **Chrome + chromedriver** (imagem
  base não pode ser `python:slim` puro — usar imagem com Chrome, ou instalar Chrome + deixar
  o `webdriver-manager` resolver o driver em runtime).
- **Selenium só na ingestão.** Nenhum endpoint da API dispara scraping.
- **Fallback de leitura:** se a ingestão falhar, a API lê os CSVs estáticos via
  `b3_source.get_b3_index` (`data/static/*_all.csv`), mantidos pela GitHub Action mensal.

### 3.6. Autenticação da SPA → API (multi-usuário)

- **Login do app:** `django.contrib.auth` (usuário/senha), telas de login/logout servidas pelo
  backend ou pela SPA via endpoint `POST /api/v1/auth/login`.
- **Sessão:** cookie de sessão do Django (`SessionMiddleware`), com o nginx servindo SPA e API
  **na mesma origem** → sem CORS, cookie `SameSite=Lax`, `CSRF` habilitado para métodos de escrita.
- **`django-ninja`**: usar `django_auth` (autenticador de sessão embutido) nas rotas protegidas.
- **Opcional (SSO Google):** se o SSO for requisito, adicionar `django-allauth` numa fase
  posterior — não bloqueia o MVP.
- O **token DLP** é um segredo **por-usuário** guardado no backend, **não** é o mecanismo de
  login. Endpoints `POST /api/v1/dlp/token` / `GET /api/v1/dlp/token-status`.

---

## 4. Estrutura de Diretórios Alvo

> Tudo dentro de `/data/projetos/app-investimentos`, branch `migracao-django`. O `src/` atual
> (Streamlit) permanece durante a transição para **validação de paridade** (§Fase 6) e é
> removido apenas ao final.

```
/data/projetos/app-investimentos/
│
├── docker-compose.yml                    # lab-postgres, lab-redis, lab-backend, lab-ingestion, lab-frontend, lab-nginx
│
├── backend/                              # [NOVO] Django 5 + Django Ninja
│   ├── Dockerfile                        # imagem do backend (Gunicorn) — SEM Chrome
│   ├── Dockerfile.ingestion              # imagem da ingestão — COM Chrome + chromedriver
│   ├── requirements.txt                  # dependências PINADAS (inclui selenium, webdriver-manager)
│   ├── manage.py
│   │
│   ├── config/
│   │   ├── settings.py                   # Postgres, Redis (django-redis), sessão/CSRF, FERNET_KEY (REUSAR a atual)
│   │   ├── urls.py                       # /api/, /admin/
│   │   ├── wsgi.py
│   │   └── asgi.py
│   │
│   ├── apps/
│   │   ├── core/
│   │   │   ├── models.py                 # UserToken (Fernet), MarketSeries
│   │   │   ├── admin.py
│   │   │   └── security.py               # wrapper Fernet (porta de src/data/user_store.py)
│   │   │
│   │   ├── accounts/
│   │   │   ├── api.py                    # login/logout/whoami
│   │   │   └── migrations/
│   │   │       └── 0002_import_supabase_tokens.py   # data migration (ver §7.2)
│   │   │
│   │   ├── api/
│   │   │   ├── router.py                 # rotas /api/v1/...
│   │   │   ├── schemas.py                # schemas Pydantic in/out
│   │   │   └── serializers.py            # pd.Series/DataFrame -> JSON (porta do envelope de cache.py)
│   │   │
│   │   ├── market/
│   │   │   ├── services/
│   │   │   │   ├── dlp.py                # buscar_historico, buscar_resumo_carteira
│   │   │   │   ├── bcb.py                # SGS + PTAX Olinda
│   │   │   │   ├── tesouro.py
│   │   │   │   ├── yahoo.py
│   │   │   │   ├── b3.py                 # Selenium + fallback b3_source
│   │   │   │   └── benchmarks.py         # processar_benchmarks + séries sintéticas (regex)
│   │   │   ├── cache.py                  # porta de src/data/cache.py -> redis-py/django-redis
│   │   │   ├── catalog.py               # porta de benchmarks_config.py + ativos.csv
│   │   │   └── management/commands/
│   │   │       └── ingest_market_data.py
│   │   │
│   │   └── migration_ir/                 # [NOVO] Planejador de Isenção de IR
│   │       ├── services.py               # calcular_compras_rebalanceadas, obter_preco_atual
│   │       ├── pdf.py                    # gerar_pdf_plano (fpdf2) — porta de 3_migracao_ir.py
│   │       └── api.py                    # endpoints do planejador
│   │
│   └── engine/                           # [MIGRADO] Núcleo de cálculo
│       ├── twr.py                        # sem alteração
│       ├── irr.py                        # sem alteração
│       ├── metrics.py                    # sem alteração
│       ├── compute.py                    # [NOVO] compute_twr_evolution, compute_drawdown,
│       │                                 #        compute_rolling_metrics, compute_risk_return,
│       │                                 #        compute_shadow, compute_yearly_summary,
│       │                                 #        compute_irr_evolution  (lógica extraída dos plot_*)
│       └── report_legacy.py              # os plot_* Matplotlib preservados p/ export CLI
│
└── frontend/                             # [NOVO] React 18 SPA
    ├── Dockerfile                        # multi-stage: build Vite -> assets estáticos
    ├── package.json                      # React 18, Vite, ECharts, Tailwind, Lucide, Axios
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    │
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        ├── api/client.js                 # Axios (withCredentials: true)
        ├── auth/                         # login form, guard de rota, contexto de sessão
        └── components/
            ├── Header.jsx
            ├── SummaryCards.jsx
            ├── FiltersBar.jsx
            ├── YearlyTable.jsx
            ├── charts/
            │   ├── TwrChart.jsx
            │   ├── DrawdownChart.jsx
            │   ├── RollingSharpeChart.jsx
            │   ├── ShadowChart.jsx
            │   └── RiskReturnChart.jsx
            └── migration/               # [NOVO] Planejador de Isenção de IR
                ├── TargetAllocationForm.jsx
                ├── MonthlyPurchasesTable.jsx
                └── ExportPdfButton.jsx
```

---

## 5. Roteiro Passo a Passo (Roadmap)

### Fase 0 — Infra mínima em container (pré-requisito)
1. Criar `docker-compose.yml` inicial com **apenas** `lab-postgres` (PostgreSQL 16) e
   `lab-redis` (Redis 7), com volumes nomeados e healthchecks.
2. Subir e validar conectividade (`psql`, `redis-cli ping`).
   *Motivo:* a Fase 1 roda `manage.py migrate` contra o Postgres — ele precisa existir antes.

### Fase 1 — Backend Django + modelos
1. `backend/requirements.txt` **com versões pinadas** (Django 5.x, django-ninja, psycopg[binary],
   django-redis, redis, pandas, numpy, yfinance, python-bcb, cryptography, fpdf2, gunicorn,
   **selenium**, **webdriver-manager**, requests, python-dotenv).
2. `config/settings.py`:
   - Postgres (`POSTGRES_HOST=lab-postgres:5432`, banco a definir — ver §7.4).
   - Cache Redis via `django-redis` (`redis://lab-redis:6379/1`).
   - `SESSION_ENGINE` padrão, `CSRF_TRUSTED_ORIGINS`, `SECURE_*` conforme o proxy.
   - **`FERNET_KEY` = a chave Fernet ATUAL** (copiada de `.streamlit/secrets.toml`) — ver §7.3.
   - Logging para **stdout** (não `RotatingFileHandler`).
3. `apps/core/models.py`: `UserToken` (FK para `User`, campo criptografado, `updated_at`),
   `MarketSeries` (nome/código da série, data, valor, fonte; índice único por série+data).
4. `apps/core/security.py`: portar `_get_cipher` / `encrypt` / `decrypt` de `src/data/user_store.py`.
5. `makemigrations` + `migrate`; registrar modelos no Admin; criar superusuário.

### Fase 2 — Serviços de mercado (`apps/market/services/`)
1. Quebrar `src/data/sources/market_data.py` (796 linhas) em módulos por fonte:
   `dlp.py`, `bcb.py`, `tesouro.py`, `yahoo.py`, `b3.py`, `benchmarks.py`.
2. Portar `src/data/cache.py` → `apps/market/cache.py` usando `redis-py` (ou o cache do Django),
   **mantendo o envelope de serialização pandas** (`{"_type": "...", "payload": to_json(orient="table")}`).
3. Portar `benchmarks_config.py` + o loader de `ativos.csv` → `apps/market/catalog.py`.
4. Definir o **contrato dos benchmarks sintéticos** (`"IPCA + X%"`, `"X% do CDI"`, `"CDI + X%"`):
   manter como string livre parseada, **ou** endpoint recebendo `{tipo, indexador, percentual}`.
5. `management/commands/ingest_market_data.py`: atualização em lote de BCB/PTAX/Tesouro/Yahoo/B3,
   gravando em `MarketSeries` + aquecendo o Redis. Reaproveitar a lógica de
   `scripts/sync_to_redis.py`.
6. Testar o comando; validar persistência no Postgres e chaves no Redis.

### Fase 2.5 — Refatorar o engine (extrair cálculo da plotagem)
1. Criar `backend/engine/compute.py` com funções **puras** que retornam DataFrame/dict
   JSON-serializável, extraindo a lógica de dentro de cada `plot_*` de `financial_report.py`:
   `compute_twr_evolution`, `compute_drawdown`, `compute_rolling_metrics`,
   `compute_risk_return`, `compute_shadow`, `compute_yearly_summary`, `compute_irr_evolution`.
2. Mover os `plot_*` Matplotlib para `backend/engine/report_legacy.py` (mantidos para export CLI;
   **não** chamados pela API).
3. Corrigir o bug do `self.logger` inexistente no caminho de dados vazios.
4. Adaptar `fetch_user_portfolio` / `build_dataset` para consumir `MarketSeries`/serviços novos.
5. Testes unitários das novas `compute_*` (reaproveitar fixtures de `tests/`).

### Fase 3 — Endpoints da API (Django Ninja)
1. `apps/accounts/api.py`: `POST /api/v1/auth/login`, `POST /api/v1/auth/logout`, `GET /api/v1/auth/me`.
2. `apps/api/router.py`:
   - `POST /api/v1/dlp/token` — salvar/criptografar token DLP do usuário logado.
   - `GET  /api/v1/dlp/token-status` — se o usuário tem token configurado.
   - `GET  /api/v1/portfolio/summary` — KPIs (patrimônio, TWR, CAGR, Sharpe, Volatilidade).
   - `GET  /api/v1/portfolio/twr-evolution` — séries base 100 para ECharts.
   - `GET  /api/v1/portfolio/drawdown`
   - `GET  /api/v1/portfolio/rolling-metrics` — volatilidade e Sharpe móveis.
   - `GET  /api/v1/portfolio/shadow-simulation`
   - `GET  /api/v1/portfolio/yearly-summary`
   - `GET  /api/v1/portfolio/irr-evolution`
   - `GET  /api/v1/benchmarks/catalog`
3. `apps/migration_ir/api.py` (**planejador de IR**):
   - `GET  /api/v1/migration/wallet` — carteira atual (via `/resumo` da DLP).
   - `POST /api/v1/migration/plan` — recebe alocação-alvo, retorna plano de compras mensais
     (respeitando o limite de R$ 20.000/mês).
   - `POST /api/v1/migration/plan.pdf` — mesmo cálculo, retorna os **bytes do PDF** (`fpdf2`);
     o React apenas dispara o download.
4. `apps/api/serializers.py`: conversão `pd.Series`/`DataFrame` → JSON (o retorno de
   `processar_benchmarks` é `dict[str, pd.Series]`, **não** serializável direto).
5. Validar tudo no Swagger (`/api/docs`).

### Fase 4 — Frontend React SPA (maior bloco)
1. Bootstrap: `package.json`, `vite.config.js` (proxy `/api` → backend em dev),
   `tailwind.config.js` (tema escuro), `api/client.js` (`withCredentials: true`).
2. Auth: formulário de login, contexto de sessão, guard de rota (redireciona sem sessão).
3. Componentes:
   - `Header.jsx` — status de conexão + configuração do token DLP.
   - `FiltersBar.jsx` — período (1A, 2A, 5A, Tudo, custom) + multiselect de benchmarks.
   - `SummaryCards.jsx` — KPIs.
   - `charts/*` — 5 gráficos ECharts (TWR, Drawdown, RollingSharpe, Shadow, RiskReturn).
   - `YearlyTable.jsx` — rentabilidade ano a ano.
   - `migration/*` — formulário de alocação-alvo, tabela de compras mensais, botão de export PDF.
4. Integração: ligar estado do React à API; validar filtros dinâmicos, zoom nos gráficos e o
   fluxo de "primeira análise" (loading enquanto a DLP responde — ver §3.4).

### Fase 5 — Dockerização e unificação
1. `backend/Dockerfile` (Gunicorn, sem Chrome) e `backend/Dockerfile.ingestion` (com Chrome).
2. `frontend/Dockerfile` (multi-stage: build Vite → assets estáticos).
3. `docker-compose.yml` final: `lab-postgres`, `lab-redis`, `lab-backend`, `lab-ingestion`
   (comando: `manage.py ingest_market_data`, agendado por cron do host ou `ofelia`/`cron` no container),
   `lab-frontend` (assets), `lab-nginx` (**um só** proxy: serve os assets do frontend e roteia
   `/api` e `/admin` para o backend).
4. Migração de dados: rodar o import dos tokens do Supabase (§7.2) e o `ingest_market_data` inicial.

### Fase 6 — Validação de paridade (obrigatória — app financeiro)
1. Selecionar 1–2 tokens DLP reais e 2–3 janelas de período.
2. Rodar o Streamlit atual (`src/ui`) e a API nova para os mesmos inputs.
3. Diferenciar numericamente: TWR base 100, Drawdown, Volatilidade/Sharpe móveis, XIRR,
   Shadow Portfolio, tabela anual. Tolerância explícita (ex.: 1e-6 relativo).
4. Só então remover `src/` (Streamlit) e as dependências `streamlit`, `altair`, `supabase`,
   `upstash-redis` do projeto.

---

## 6. Seções que faltavam na v1

### 6.1. Autenticação SPA → API
Ver §3.6. Resumo: login Django + cookie de sessão, mesma origem via nginx (sem CORS), CSRF
habilitado, `django_auth` do Ninja nas rotas protegidas. SSO Google fica para fase posterior
opcional (`django-allauth`).

### 6.2. Migração dos tokens do Supabase
- Script/`data migration` (`apps/accounts/migrations/0002_import_supabase_tokens.py` **ou**
  um `manage.py import_supabase_tokens`).
- Passos: ler `user_tokens` do Supabase (URL + service_role atuais) → para cada `user_email`,
  `get_or_create` do `User` Django → gravar o `encrypted_token` **como está** (já está cifrado
  com a mesma chave Fernet) no `UserToken`.
- **Não descriptografar/recriptografar** — basta reusar a chave (ver §6.3).

### 6.3. Continuidade da chave Fernet
- **Copiar a `fernet_key` atual** de `.streamlit/secrets.toml` para `FERNET_KEY` no
  `settings.py` (via env/secret do compose).
- Uma chave nova **inutiliza todos os tokens já cadastrados** — obrigaria todo usuário a
  recadastrar o token DLP. Evitar.
- Registrar a chave num gerenciador de segredos do LAB (não commitar).

### 6.4. Contrato dos benchmarks sintéticos
- Hoje: strings `"IPCA + 6%"`, `"110% do CDI"`, `"CDI + 2%"` parseadas por regex dentro de
  `build_dataset`.
- Decisão a tomar na Fase 2: manter string livre no request **ou** estruturar
  (`{"base": "IPCA", "operador": "+", "valor": 6}`). Recomendação: estruturar na API, aceitar a
  string só como açúcar no frontend.

### 6.5. Serialização pandas → JSON
- Reusar o envelope de `src/data/cache.py`:
  `{"_type": "pd.Series"|"pd.DataFrame", "payload": obj.to_json(orient="table")}`.
- `processar_benchmarks` devolve `dict[str, pd.Series]` — o serializer precisa iterar e
  converter cada série; alinhar o formato de saída ao que o ECharts espera
  (`[[timestamp, valor], ...]` ou colunas paralelas).

### 6.6. Testes
- Portar `tests/test_twr.py`, `test_irr.py`, `test_metrics.py` (engine — continuam válidos).
- Novos: testes das `compute_*` (Fase 2.5), testes de contrato dos endpoints (Django Ninja /
  `pytest-django`), teste do cálculo do plano de IR (limite R$ 20k/mês), e o **harness de
  paridade** da Fase 6.
- Adicionar `pytest.ini`/`pyproject.toml` com config do `pytest-django` e um banco de teste.

### 6.7. Logs em container
- Trocar `RotatingFileHandler` (`src/utils/logger.py`) por handler de **stdout**; deixar a
  coleta a cargo do Docker. Nível via env (`LOG_LEVEL`).

### 6.8. CORS
- Com nginx servindo SPA + API na mesma origem, **CORS é desnecessário**. Não instalar
  `django-cors-headers` salvo se o frontend passar a ser servido de outra origem em dev
  (nesse caso, liberar só `http://localhost:5173`).

---

## 7. Decisões pendentes / pontos de atenção

### 7.1. SSO Google
Perde-se com Django auth. Aceito para o MVP (LAB). Se voltar a ser requisito, `django-allauth`
numa fase posterior.

### 7.2. Volume de usuários
Multi-usuário confirmado. Isso torna obrigatórios: migração de `user_tokens`, isolamento de
carteira por usuário, e a auth de sessão da SPA. `MarketSeries` permanece **global** (dado de
mercado é comum a todos).

### 7.3. Nome do banco / prefixo `lab-`
A v1 mistura `banco_lab` e nomes genéricos. **Decidir:** se a stack LAB vai virar a mainline
(merge de `migracao-django` em `dev`/`main`), remover o prefixo `lab-` de containers e banco antes de
consolidar. Se for fork paralelo permanente, manter.

### 7.4. Agendador da ingestão
Cron do host chamando `docker compose run --rm lab-ingestion`, ou um scheduler no container
(`ofelia`, `supercronic`). Recomendação: `ofelia` no compose (simples, declarativo).

### 7.5. Selenium em container
Fonte de fragilidade. Mitigações: imagem dedicada (`Dockerfile.ingestion`), timeout curto,
retries, e **fallback automático** para `data/static/*_all.csv` quando o scraping falhar.

---

## 8. Plano de Verificação e Testes

### 8.1. CLI
```bash
# Fase 0 — infra
docker compose up -d lab-postgres lab-redis
docker compose exec lab-postgres pg_isready
docker compose exec lab-redis redis-cli ping

# Fase 1 — backend
cd backend
python manage.py check
python manage.py migrate
python manage.py createsuperuser

# Conexão Postgres + Redis pela app
python manage.py shell -c "from apps.core.models import UserToken; print('PG OK'); from django.core.cache import cache; cache.set('t',123,10); print('Redis OK:', cache.get('t'))"

# Fase 2 — ingestão
python manage.py ingest_market_data
python manage.py shell -c "from apps.core.models import MarketSeries; print(MarketSeries.objects.count(), 'pontos de série')"

# Fase 2.5 — engine
pytest backend/engine backend/apps -q

# Fase 4 — frontend
cd frontend && npm install && npm run build

# Fase 6 — paridade
python scripts/parity_check.py --token <TOKEN> --start 2020-01-01 --end 2024-12-31
```

### 8.2. Testes manuais
- [ ] `/admin` — login administrativo; inspecionar `UserToken` e `MarketSeries`.
- [ ] `/api/docs` — executar cada endpoint, resposta `200` + JSON coerente.
- [ ] SPA — login, cadastrar token DLP, ver os 5 gráficos + tabela anual carregando.
- [ ] SPA — zoom (`dataZoom`), toggle de benchmarks na legenda, troca de período.
- [ ] Planejador de IR — definir alocação-alvo, conferir tabela mensal e o limite de R$ 20k,
      baixar o PDF.
- [ ] Primeira análise de um usuário novo — estado de loading enquanto a DLP responde.
- [ ] Confirmar que token e dados não vazam em logs nem em respostas não autenticadas.
- [ ] Paridade: números do Streamlit antigo == números da API nova (dentro da tolerância).

---

## 9. Ordem de execução recomendada (checklist)

1. [ ] **Fase 0** — `docker-compose.yml` com Postgres + Redis; validar.
2. [ ] **Fase 1** — Django + `settings.py` (Fernet reusada) + `UserToken`/`MarketSeries` + Admin.
3. [ ] **Fase 2** — quebrar `market_data.py` em `apps/market/services/`; portar cache; `ingest_market_data`.
4. [ ] **Fase 2.5** — `engine/compute.py` (extrair dos `plot_*`); `report_legacy.py`; corrigir `self.logger`.
5. [ ] **Fase 3** — endpoints Ninja (auth, portfolio, benchmarks, migration_ir + PDF); Swagger.
6. [ ] **Fase 4** — SPA React (auth, filtros, 5 gráficos ECharts, tabela anual, planejador de IR).
7. [ ] **Fase 5** — Dockerfiles (backend / ingestion-com-Chrome / frontend), compose final, nginx único.
8. [ ] **Migração de dados** — import dos tokens do Supabase; `ingest_market_data` inicial.
9. [ ] **Fase 6** — validação de paridade; só então remover `src/` e libs legadas.
10. [ ] Decidir prefixo `lab-` / merge de `migracao-django` (§7.3).
