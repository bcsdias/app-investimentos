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
> **Fases 0 + 1 concluídas em 2026-08-28; Fases 2 + 2.5 em 2026-08-31; Fase 3 em 2026-10-26** — ver §2.0. As
> decisões travadas estão em [`adr/`](adr/README.md); a orientação de trabalho no backend em
> [`../backend/CLAUDE.md`](../backend/CLAUDE.md).

### 2.0. Já implementado — Fases 0 + 1 (2026-08-28), Fases 2 + 2.5 (2026-08-31) e Fase 3 (2026-10-26)

| Camada | Local | Situação |
|---|---|---|
| **Infra LAB** | `/data/projetos/lab-postgres/`, `/data/projetos/lab-redis/` (fora do repo) | `lab-postgres` + `lab-redis` compartilhados, *healthy*, na rede externa `lab-net`. `lab-nginx` de teste removido. [ADR 0001](adr/0001-infra-lab-compartilhada.md). |
| **Database do app** | `lab-postgres` → database + role `appinvest` (`CREATEDB`) | Criado por `backend/scripts/bootstrap_db.sh` (idempotente, versionado). Nenhuma tabela do app no `banco_lab`. [ADR 0006](adr/0006-prefixo-lab-containers-appinvest-codigo.md). |
| **Projeto Django** | `backend/config/` | `settings.py` dual host/container: DB `appinvest`, cache `django-redis`, Fernet, logging stdout ([ADR 0008](adr/0008-logging-stdout-apenas.md)), TZ `America/Sao_Paulo`. `urls.py` expõe `/admin/` e `/api/v1/` (`config/api.py`, Swagger em `/api/v1/docs`). |
| **Usuário** | `backend/apps/accounts/` | `User(AbstractUser)` customizado; `AUTH_USER_MODEL = "accounts.User"`. Endpoints de sessão `/api/v1/auth/{csrf,login,logout,me}` em `apps/accounts/api.py`. [ADR 0002](adr/0002-modelo-user-customizado-dia-1.md). |
| **Cripto** | `backend/apps/core/security.py` | `encrypt()` / `decrypt()` Fernet sobre `settings.FERNET_KEY` — **mesma chave do app legado**, compat. testada nos dois sentidos. [ADR 0003](adr/0003-reuso-da-chave-fernet-do-legado.md). |
| **Modelos** | `backend/apps/core/models.py` | `UserToken` (OneToOne, token cifrado em repouso, `set_token`/`get_token`) e `MarketSeries` (`UniqueConstraint(series_key, reference_date)`). Migration `core.0001`. [ADR 0007](adr/0007-modelagem-usertoken-marketseries.md). |
| **Admin** | `backend/apps/core/admin.py` | `UserToken` sem "add", `encrypted_token` read-only; `MarketSeries` com filtro por `source` e navegação por data. |
| **Testes** | `backend/{apps,engine}/*/tests/` + `backend/pytest.ini` | `pytest-django`, `--reuse-db`, marker `integration` desmarcado por padrão. **Total atual: 238 testes unitários/contrato (+ 3 `integration`)** verdes. |
| **Scripts** | `backend/scripts/` | `bootstrap_db.sh` (cria role/database) e `rotate_db_password.sh` (rotaciona a senha). |
| **Docs** | `backend/README.md`, `backend/CLAUDE.md`, `docs/adr/` | Setup/rodar, convenções do agente, e ADRs 0001–0014. |
| **Serviços de Mercado (Fase 2 + Fase 3)** | `backend/apps/market/` | App `apps.market`: `services/sources/{base,bcb,ptax,yahoo,tesouro,b3,dlp}.py` (erros DLP tipados `DLPAuthError`/`DLPUnavailableError`, invalidação por geração e single-flight via `cache.add`), `repository.py`, `cache.py` (`single_flight`), `catalog.py` (`SeriesDef`), `assets.py` + `data/ativos.csv` (universo de ~7.500 tickers + variante BRL), `synthetics.py` (`SyntheticSpec` + `display_name`), `transforms.py`, `series.py` (catálogo + fetch-on-read de tickers sob demanda). Migration `core.0002` (`kind`), comandos `ingest_market_data` (inclui tickers sob demanda) / `backfill_market_data`. [ADR 0011](adr/0011-marketseries-valor-cru-kind.md), [ADR 0014](adr/0014-tickers-sob-demanda.md). |
| **Engine de Cálculo (Fase 2.5)** | `backend/engine/` | Pacote puro `engine` (irmão de `apps/`, sem `models.py`): `twr.py` / `irr.py` cópia literal de `src/engine/`; `metrics.py` idem + `pct_change(fill_method=None)`; `compute.py` (7 funções puras extraídas dos `plot_*`); `dataset.py` (`fetch_user_portfolio` + `build_dataset`, consome `apps.market.series`); `report_legacy.py` (export CLI Matplotlib, nunca importado pela API). Bug `self.logger` corrigido. `pytest.ini` → `testpaths = apps engine`. [ADR 0012](adr/0012-engine-copia-literal-fase-2-5.md). |
| **Correções de revisão (Fases 2 / 2.5)** | `backend/apps/market/`, `backend/engine/` | Revisão pós-implementação (`code-review` + manual, 2026-08-31): 6 defeitos corrigidos com TDD — ver §2.3. |
| **API da Carteira (Fase 3)** | `backend/config/api.py`, `backend/apps/api/`, `backend/apps/accounts/api.py` | `NinjaAPI` em `/api/v1/` (Swagger em `/api/v1/docs`), `ApiSessionAuth` (sessão + CSRF no envelope `ApiError`), `/auth/{csrf,login,logout,me}`, `/dlp/{token,token-status}`, `/portfolio/{filters,refresh,summary,twr-evolution,drawdown,rolling-metrics,risk-return,shadow,yearly-summary,irr-evolution}`, `/benchmarks/catalog`, `/assets/search`. Integração ao vivo validada contra DLP (HTTP 401/403 em token inválido → `422 dlp_token_rejected`) e Yahoo Finance (`AAPL` sob demanda). **238 testes verdes** (`.venv/bin/pytest -q`), `manage.py check` limpo, sem migrations pendentes. [ADR 0013](adr/0013-contrato-da-api-v1.md), [ADR 0014](adr/0014-tickers-sob-demanda.md). |

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

- **Sem endpoints do Planejador de IR (`apps/migration_ir/`)**: separados na **Fase 3b** (spec própria).
- **Sem `frontend/`, sem `package.json`** — zero JS/Node no repositório (Fase 4).
- **Sem Dockerfiles do app** nem compose final/nginx único — o backend roda no host (Fase 5).
  Só os containers de **infra** (`lab-postgres`, `lab-redis`) existem.
- **Dados ainda no legado**: os tokens continuam no Supabase (a migração `user_tokens` → `UserToken` ocorre na fase correspondente); o cache já está integrado no `lab-redis`.
- ~~`.devcontainer/devcontainer.json` aponta para `app/web_app_v2.py` (**caminho stale**, não existe mais).~~ **Removido em 2026-10-26**: template Codespaces do app Streamlit (Python 3.11), incompatível com o fluxo atual (backend no host, Python 3.12; Docker só para `lab-postgres`/`lab-redis`).

### 2.3. Dívida técnica / lixo a resolver **durante** a migração (não carregar adiante)

- `financial_report.py` referenciava `self.logger` (inexistente) no branch de dados vazios — **corrigido** em `engine/dataset.py` na Fase 2.5 (mantido o registro histórico).
- Algoritmos numéricos do engine (`irr.py`, `twr.py`, `metrics.py`) copiados literalmente na Fase 2.5 para manter validade de paridade da Fase 6; revisão profunda agendada para **pós-Fase 6** ([ADR 0012](adr/0012-engine-copia-literal-fase-2-5.md)).
- **Revisão pós-Fases 2/2.5 (2026-08-31, `code-review` + manual) — 6 defeitos corrigidos com TDD** (commits `a15fcf1` market, `e1214a4` engine):
  1. `transforms.synthetic_from_spec` anualizava o spread do operador `+` por `1/252` fixo; numa série mensal (`IPCA + 6%`, benchmark default) isso rendia ~0.28% a.a. em vez de 6%. Passou a detectar a cadência da base (12 passos/ano p/ mensal, 252 p/ diária) — casa com o `processar_benchmarks` legado (`1.06 ** (1/12)`).
  2. `services/sources/yahoo.fetch_yahoo`: `yf.download` engole erro de rede e devolve DataFrame vazio; retorno vazio virou `FetchError` (o `ingest` marca `failed_keys` e sai 1) em vez de `[]` silencioso.
  3. `repository.upsert_points`: dedup de `reference_date` repetido no lote — `bulk_create(update_conflicts=True)` levantava *"cannot affect row a second time"* com CSV/PTAX de dia duplicado.
  4. `ingest_market_data`: `upsert_points` movido para dentro do `try` (um `DatabaseError` numa série não aborta mais o lote); `_warm_cache` usa `hoje - 5*365 dias` em vez de `date.replace(year=…)`, que levantava `ValueError` em 29/fev.
  5. `engine/compute.compute_risk_return`: série de volatilidade zero (benchmark achatado por `ffill`) dava Sharpe `±inf` que o `.dropna()` não removia — agora a linha cai da tabela.
  6. Homóglifo cirílico (`о`) numa string de log de `backfill_market_data.py`.
- **Pendências de baixo impacto assumidas de propósito** (candidatas a limpeza futura, não bloqueiam as próximas fases):
  - `apps.market.cache.make_key` não sanitiza a chave: nomes com espaço/`&`/`/` (`"S&P 500"`, `"AAPL BRL"`) geram `CacheKeyWarning`. Tolerado pelo Redis (`django-redis`); quebraria com memcached.
  - Ativos de Londres (`.L`) na variante `"BRL"` são convertidos pela PTAX USD/BRL (igual ao legado, para preservar paridade na Fase 6); corrigir câmbio GBP/GBp pós-Fase 6 ([ADR 0014](adr/0014-tickers-sob-demanda.md)).
  - `CATALOGO_TD` **vazio de propósito** — `services/sources/tesouro.py` + ramos `td:` de `catalog`/`backfill` são caminho dormente, testados só com fixture. Renda fixa hoje é modelada pelos índices ANBIMA (IMA-B/IRF-M/IMA-S via BCB SGS + emenda com ETF), não por PU de título individual.
  - `test_sources_b3.py::test_fetch_b3_selenium_*` fica verde porque `ChromeDriverManager().install()` falha antes do mock — cobre o `raise FetchError`, mas pelo motivo errado.
  - Alguns testes de `engine/tests/test_dataset.py` usam `@pytest.mark.django_db` sem tocar o banco (monkeypatch total).
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
>
> **Nota:** os blocos `config/`, `apps/accounts/`, `apps/market/`, `apps/api/` e `engine/` abaixo refletem o que foi **efetivamente
> construído** nas Fases 1, 2, 2.5 e 3.
> `apps/migration_ir/` (Fase 3b) e `frontend/` (Fase 4) seguem sendo **alvo**.
> Estrutura viva e detalhada do backend em [`../backend/README.md`](../backend/README.md).

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
│   │   ├── api.py                        # NinjaAPI (/api/v1/, docs em /api/v1/docs) + handler ApiError
│   │   ├── settings.py                   # Postgres, Redis (django-redis), sessão/CSRF, FERNET_KEY (REUSAR a atual)
│   │   ├── urls.py                       # /api/v1/, /admin/
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
│   │   │   ├── api.py                    # /api/v1/auth/{csrf,login,logout,me}
│   │   │   └── migrations/
│   │   │       └── 0002_import_supabase_tokens.py   # data migration (Fase 5 — ver §7.2)
│   │   │
│   │   ├── market/                       # [FASES 2 + 3] serviços de mercado e universo de tickers
│   │   │   ├── services/sources/         # cada fonte fala só com a origem externa; levanta FetchError
│   │   │   │   ├── base.py               # RawPoint, FetchError, DLPError, DLPAuthError, DLPUnavailableError
│   │   │   │   ├── bcb.py                # SGS (chunking de 5 anos, valor cru)
│   │   │   │   ├── ptax.py               # PTAX venda via Olinda
│   │   │   │   ├── yahoo.py              # yf.download (Close ajustado); vazio -> FetchError
│   │   │   │   ├── tesouro.py            # PU Base Manhã (caminho dormente — CATALOGO_TD vazio)
│   │   │   │   ├── b3.py                 # read_b3_static (host) + fetch_b3_selenium (só ingestão)
│   │   │   │   └── dlp.py                # buscar_historico / buscar_resumo_carteira (geração + single-flight)
│   │   │   ├── data/ativos.csv           # cópia empacotada do universo (~7.500 linhas)
│   │   │   ├── assets.py                 # universo de tickers Yahoo + variante BRL + search()
│   │   │   ├── repository.py             # único acesso a core.MarketSeries (upsert em lote + dedup, load_series)
│   │   │   ├── cache.py                  # envelope pandas sobre django-redis + single_flight
│   │   │   ├── catalog.py                # SeriesDef em Python puro (cruas + derivadas) + DEFAULT_BENCHMARKS
│   │   │   ├── synthetics.py             # SyntheticSpec{base,operador,valor} + display_name + parse_legacy_string
│   │   │   ├── transforms.py             # derivações puras: accumulate, splice_proxy, to_brl, synthetic_from_spec, composite, rebase_100
│   │   │   ├── series.py                 # fachada de leitura: Redis -> repository + transforms (catálogo + fetch-on-read) -> Redis
│   │   │   └── management/commands/
│   │   │       ├── ingest_market_data.py   # ingestão incremental ao vivo (catálogo + tickers sob demanda)
│   │   │       └── backfill_market_data.py # bootstrap offline dos CSVs do repo legado (descartável)
│   │   │
│   │   ├── api/                          # [FASE 3] API da Carteira (Django Ninja)
│   │   │   ├── errors.py                 # ApiError(status, code, message, details)
│   │   │   ├── csrf.py                   # ApiSessionAuth (CSRF devolvendo ApiError 403)
│   │   │   ├── schemas.py                # schemas Ninja (request/response)
│   │   │   ├── serializers.py            # pd.Series/DataFrame -> JSON no formato do ECharts
│   │   │   ├── analysis.py               # validate_request, get_token, dlp_errors, run_analysis
│   │   │   └── routers/
│   │   │       ├── dlp.py                # POST/DELETE /dlp/token, GET /dlp/token-status
│   │   │       ├── portfolio.py          # filters, refresh, summary + 7 endpoints de gráfico
│   │   │       ├── benchmarks.py         # GET /benchmarks/catalog
│   │   │       └── assets.py             # GET /assets/search
│   │   │
│   │   └── migration_ir/                 # [FASE 3b — alvo] Planejador de Isenção de IR
│   │       ├── services.py               # calcular_compras_rebalanceadas, obter_preco_atual
│   │       ├── pdf.py                    # gerar_pdf_plano (fpdf2) — porta de 3_migracao_ir.py
│   │       └── api.py                    # endpoints do planejador
│   │
│   └── engine/                           # [FASE 2.5] pacote puro, irmão de apps/ (sem models.py)
│       ├── twr.py                        # cópia literal de src/engine/twr.py
│       ├── irr.py                        # cópia literal de src/engine/irr.py
│       ├── metrics.py                    # cópia literal + pct_change(fill_method=None) (spec §5.4)
│       ├── compute.py                    # 7 funções puras extraídas dos plot_*: compute_twr_evolution,
│       │                                 #   compute_drawdown, compute_rolling_metrics, compute_risk_return,
│       │                                 #   compute_shadow, compute_yearly_summary, compute_irr_evolution
│       ├── dataset.py                    # fetch_user_portfolio (DLP -> TWR) + build_dataset (consome apps.market.series)
│       └── report_legacy.py              # plot_* Matplotlib preservados p/ export CLI — a API NÃO importa
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

### Fase 3 — API da Carteira (Django Ninja)
1. `apps/accounts/api.py`: `GET /api/v1/auth/csrf`, `POST /api/v1/auth/login`, `POST /api/v1/auth/logout`, `GET /api/v1/auth/me`.
2. `apps/api/routers/{dlp,portfolio,benchmarks,assets}.py` + `analysis.py` + `serializers.py`:
   - `POST /api/v1/dlp/token`, `DELETE /api/v1/dlp/token`, `GET /api/v1/dlp/token-status`.
   - `GET  /api/v1/portfolio/filters` e `POST /api/v1/portfolio/refresh`.
   - `POST /api/v1/portfolio/{summary,twr-evolution,drawdown,rolling-metrics,risk-return,shadow,yearly-summary,irr-evolution}` (corpo `AnalysisRequest` / `RollingRequest`, envelope `{data, meta}`).
   - `GET  /api/v1/benchmarks/catalog` e `GET /api/v1/assets/search`.
3. `apps/market/assets.py` + `series.py`: universo de tickers do `ativos.csv`, variante `"BRL"`, fetch-on-read no Yahoo e atualização no `ingest_market_data`.
4. Validar no Swagger (`/api/v1/docs`).

### Fase 3b — Planejador de Migração de IR (`apps/migration_ir/`)
1. `apps/migration_ir/api.py`:
   - `GET  /api/v1/migration/wallet` — carteira atual (via `/resumo` da DLP).
   - `POST /api/v1/migration/plan` — recebe alocação-alvo, retorna plano de compras mensais
     (respeitando o limite de R$ 20.000/mês).
   - `POST /api/v1/migration/plan.pdf` — mesmo cálculo, retorna os **bytes do PDF** (`fpdf2`);
     o React apenas dispara o download.

### Fase 3c — Otimizações DLP sem impacto no engine (mapeado em 2026-10-26; escopo reduzido em 2026-10-26)
> **Referência técnica:** auditoria realizada sobre [`referencia-api-dlp-invest.md`](referencia-api-dlp-invest.md) cruzada com `apps/market/services/sources/dlp.py`, `engine/dataset.py`, `engine/twr.py` e `apps/api/routers/portfolio.py`.
> **Escopo:** só o que **não muda números nem toca `engine/`** — preserva a paridade da Fase 6 ([ADR 0012](adr/0012-engine-copia-literal-fase-2-5.md)). Entra **antes da Fase 4** (a SPA troca filtros o tempo todo). Os itens que tocam o engine ou mudam números foram movidos para a **Fase 7**.

1. **Cache único de `/historico` + filtragem em memória (`dlp.buscar_historico`):** como `GET /historico` sem filtros já traz obrigatoriamente `ativo`, `classe` e `corretora` em todas as linhas (`HistoricModel`) e já é chamado no carregamento inicial por `GET /api/v1/portfolio/filters`, cachear apenas o DataFrame completo por `(token_hash, generation)` no Redis e aplicar os filtros `ativo`/`classe`/`corretora` em memória no Pandas. Elimina chamadas HTTP extras (até 60s) à DLP ao trocar filtros na UI e reduz o armazenamento no Redis para 1 DataFrame por usuário.
2. **Extração rápida de filtros (`portfolio._extract_distinct`):** substituir `df[col].dropna().astype(str).map(str.strip)` na coluna inteira do histórico diário por `pd.unique(df[col].dropna())` antes da normalização e ordenação.
3. **Domínio oficial DLP (`dlp._BASE`):** atualizar de `https://users.dlombelloplanilhas.com` (legado) para `https://users.dlpinvest.com.br` (ou parametrizar via `settings.py`).
4. **Verificação:** teste `integration` com token real confirmando que o filtro em memória produz o mesmo DataFrame que `GET /historico?ativo=…&classe=…` (premissa do item 1).

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

### Fase 7 — Correções e otimizações do engine pós-paridade (movido da 3c em 2026-10-26)
> Só depois da Fase 6: estes itens alteram `engine/` (cópia literal do legado) e/ou mudam números, o que invalidaria a comparação de paridade. Entra junto com a revisão profunda do engine já prevista na [ADR 0012](adr/0012-engine-copia-literal-fase-2-5.md).

1. **Otimizações (sem mudar números):**
   - **Pré-processamento de `date` e remoção de `groupby` duplicado (`dlp.py`, `dataset.py`, `twr.py`):** converter `df["date"] = pd.to_datetime(df["date"])` uma única vez antes de `mc.set_df` em `dlp.buscar_historico` e evitar que `calculate_twr` refaça `pd.to_datetime`, `groupby("date").sum()` e cálculo de `fluxo` quando `fetch_user_portfolio` já entrega o DataFrame agrupado por data.
2. **Correções de recorte de período (`start_date` / `end_date`) em `engine/dataset.py` (mudam números de `/summary`, `/shadow`, `/irr-evolution`):**
   - **Corte por `end_date` em `portfolio_df` (`grp`):** hoje `fetch_user_portfolio` filtra `grp` apenas por `start_date`, deixando `portfolio_df` ir até a data atual mesmo quando o usuário informa `end_date` no passado (apenas `twr` era cortado por `end_date` dentro de `calculate_twr`). Filtrar `grp[grp["date"] <= pd.to_datetime(end_date)]` para que `/summary` (`pat_fim`, `aportes`, `ganho`, `xirr`), `/shadow` e `/irr-evolution` respeitem `end_date`.
   - **Recálculo de `fluxo` após ajuste de `gap` no `start_date`:** quando `start_date` recorta o histórico no meio da carteira, `filt["vlr_investido"] += gap` ajusta o valor investido para igualar `vlr_mercado.iloc[0]`, mas `filt["fluxo"].iloc[0]` permanecia com o diff anterior ao ajuste, distorcendo o fluxo inicial usado por `compute_irr_evolution` (`all_flows = -1 * df["fluxo"].values`). Recalcular `filt["fluxo"]` (ou ajustar `filt["fluxo"].iloc[0] = filt["vlr_investido"].iloc[0] - filt["proventos"].iloc[0]`) após somar o `gap`.

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

1. [x] **Fase 0** — `docker-compose.yml` com Postgres + Redis; validar.
2. [x] **Fase 1** — Django + `settings.py` (Fernet reusada) + `UserToken`/`MarketSeries` + Admin.
3. [x] **Fase 2** — quebrar `market_data.py` em `apps/market/services/`; portar cache; `ingest_market_data` + `backfill_market_data`.
4. [x] **Fase 2.5** — `engine/compute.py` (extrair dos `plot_*`); `report_legacy.py`; corrigir `self.logger`.
5. [x] **Fase 3** — endpoints Ninja da carteira (auth, token DLP, 8 análises + filters/refresh, benchmarks, busca e fetch-on-read de tickers); Swagger em `/api/v1/docs`.
6. [ ] **Fase 3b** — Planejador de Isenção de IR (`/migration/*` + PDF `fpdf2`).
7. [ ] **Fase 3c** — Otimizações DLP sem impacto no engine (cache único de `/historico` com filtro em memória, `unique()` em `/filters`, domínio `dlpinvest.com.br` parametrizado) (ver §Fase 3c).
8. [ ] **Fase 4** — SPA React (auth, filtros, gráficos ECharts, tabela anual, planejador de IR).
9. [ ] **Fase 5** — Dockerfiles (backend / ingestion-com-Chrome / frontend), compose final, nginx único.
10. [ ] **Migração de dados** — import dos tokens do Supabase; `ingest_market_data` inicial.
11. [ ] **Fase 6** — validação de paridade; só então remover `src/` e libs legadas.
12. [ ] **Fase 7** — pós-paridade: correções de recorte `start_date`/`end_date` e `fluxo`/`gap` em `engine/dataset.py`, pré-parse de `date` e dedup de `groupby` no TWR, revisão profunda do engine (ADR 0012) (ver §Fase 7).
13. [ ] Decidir prefixo `lab-` / merge de `migracao-django` (§7.3).
