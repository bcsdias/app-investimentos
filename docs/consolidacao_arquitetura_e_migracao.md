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
> **Fases 0 + 1 concluídas em 2026-08-28; Fases 2 + 2.5 em 2026-08-31; Fases 3, 3b e 3c em 2026-10-26; Fases 4, 5 e 6 em 2026-10-27; Fase 7a em 2026-10-28; Fase 7b em 2026-10-29; Fase 7c em 2026-10-30** — ver §2.0. As
> decisões travadas estão em [`adr/`](adr/README.md); a orientação de trabalho no backend em
> [`../backend/CLAUDE.md`](../backend/CLAUDE.md).

### 2.0. Já implementado — Fases 0 + 1 (2026-08-28), Fases 2 + 2.5 (2026-08-31), Fases 3, 3b e 3c (2026-10-26), Fases 4, 5 e 6 (2026-10-27), Fase 7a (2026-10-28), Fase 7b (2026-10-29), Fase 7c (2026-10-30)

| Camada | Local | Situação |
|---|---|---|
| **Infra LAB** | `/data/projetos/lab-postgres/`, `/data/projetos/lab-redis/` (fora do repo) | `lab-postgres` + `lab-redis` compartilhados, *healthy*, na rede externa `lab-net`. `lab-nginx` de teste removido. [ADR 0001](adr/0001-infra-lab-compartilhada.md). |
| **Database do app** | `lab-postgres` → database + role `appinvest` (`CREATEDB`) | Criado por `backend/scripts/bootstrap_db.sh` (idempotente, versionado). Nenhuma tabela do app no `banco_lab`. [ADR 0006](adr/0006-prefixo-lab-containers-appinvest-codigo.md). |
| **Projeto Django** | `backend/config/` | `settings.py` dual host/container: DB `appinvest`, cache `django-redis`, Fernet, logging stdout ([ADR 0008](adr/0008-logging-stdout-apenas.md)), TZ `America/Sao_Paulo`. `urls.py` expõe `/admin/` e `/api/v1/` (`config/api.py`, Swagger em `/api/v1/docs`). Erros de validação Ninja normalizados em `422 invalid_request` no envelope `ApiError` ([ADR 0020](adr/0020-backlog-spa-api-fase-7c.md)). |
| **Usuário** | `backend/apps/accounts/` | `User(AbstractUser)` customizado; `AUTH_USER_MODEL = "accounts.User"`. Endpoints de sessão `/api/v1/auth/{csrf,login,logout,me}` em `apps/accounts/api.py` com rate limit de login de 5 tentativas por 15 min por (usuário, IP) retornando `429 too_many_attempts` ([ADR 0002](adr/0002-modelo-user-customizado-dia-1.md), [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md)). |
| **Cripto** | `backend/apps/core/security.py` | `encrypt()` / `decrypt()` Fernet sobre `settings.FERNET_KEY` — **mesma chave do app legado**, compat. testada nos dois sentidos. [ADR 0003](adr/0003-reuso-da-chave-fernet-do-legado.md). |
| **Modelos** | `backend/apps/core/models.py` | `UserToken` (OneToOne, token cifrado em repouso, `set_token`/`get_token`) e `MarketSeries` (`UniqueConstraint(series_key, reference_date)`). Migration `core.0001`. [ADR 0007](adr/0007-modelagem-usertoken-marketseries.md). |
| **Admin** | `backend/apps/core/admin.py` | `UserToken` sem "add", `encrypted_token` read-only; `MarketSeries` com filtro por `source` e navegação por data. |
| **Testes** | `backend/{apps,engine}/*/tests/` + `backend/pytest.ini` | `pytest-django`, `--reuse-db`, marker `integration` desmarcado por padrão. **Total atual: 469 testes unitários/contrato (+ 10 `integration`: 7 passando — 5 deles do oráculo DLP, que exige o export local `dlp_sheet/` — e 3 pulados sem token real)** verdes; harness de paridade `scripts/parity/` com **41 testes** e linha de base da Fase 7c em `reports/parity/20261029-211840-fase7c/baseline`. Frontend: **103 testes Vitest** (`frontend/src/test/`). |
| **Scripts** | `backend/scripts/` | `bootstrap_db.sh` (cria role/database) e `rotate_db_password.sh` (rotaciona a senha). |
| **Docs** | `backend/README.md`, `backend/CLAUDE.md`, `docs/adr/` | Setup/rodar, convenções do agente, e ADRs 0001–0020. |
| **Serviços de Mercado (Fase 2 + Fase 3 + Fase 7b + Fase 7c)** | `backend/apps/market/` | App `apps.market`: `services/sources/{base,bcb,ptax,yahoo,tesouro,b3,dlp}.py` (erros DLP tipados `DLPAuthError`/`DLPUnavailableError`, invalidação por geração e single-flight com dono via Lua script, geração DLP iniciada em `time_ns`, cache temporário de `DLPAuthError`), `repository.py` (com `load_dates` e bump de geração), `cache.py` (geração global `market:series:gen`), `gaps.py` (detecção de lacunas por cadência 10d/45d, lacunas nas bordas da janela e cauda parada delimitada em `min(crop_end, _today())`), `catalog.py` (`SeriesDef`), `assets.py` + `data/ativos.csv` (universo de ~7.500 tickers + variante BRL), `synthetics.py` (`SyntheticSpec` + `display_name`), `transforms.py`, `series.py` (catálogo + fetch-on-read de tickers sob demanda). Migration `core.0002` (`kind`), comandos `ingest_market_data` (com re-busca de lacunas internas) / `backfill_market_data` / `market_gaps`. [ADR 0011](adr/0011-marketseries-valor-cru-kind.md), [ADR 0014](adr/0014-tickers-sob-demanda.md), [ADR 0019](adr/0019-integridade-dos-dados-de-mercado.md), [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md). |
| **Engine de Cálculo (Fase 2.5 + Fase 7a + Fase 7b + Fase 7c)** | `backend/engine/` | Pacote puro `engine` (irmão de `apps/`, sem `models.py`): metodologia canônica da planilha DLP implementada na Fase 7a, abertura de benchmarks no último valor conhecido (Fase 7b), e correções de borda da Fase 7c — `flows.py` (fluxo líquido com lucro de trades e proventos), `twr.py` (HPR aporte-início/resgate-fim), `irr.py` (XIRR Actual/365 invariante à escala com Newton + fallback `scipy.optimize.brentq`; TIR do período sem anualizar abaixo de 12 meses, campos `xirr_period`, `xirr_is_annual` e `annual_from` conforme convenção ANBIMA), `metrics.py` (Sharpe contra série real de CDI do período), `dataset.py` (abertura no fechamento anterior a `start_date`, benchmarks com lookback de 40 dias e abertura no último valor conhecido, corte por `end_date`, recálculo pós-gap, `DatasetResult.gaps` com bordas e caudas), `compute.py` (sem dupla contagem de saldo inicial no shadow e `/summary`; tabela anual omitindo ano que contém apenas a data de abertura), `report_legacy.py` (export CLI). Validado contra oráculo DLP de 123 meses e contra a janela jan–set/2026 do dashboard DLP (`test_oracle_dlp.py`). [ADR 0012](adr/0012-engine-copia-literal-fase-2-5.md), [ADR 0018](adr/0018-metodologia-de-retorno-dlp.md), [ADR 0019](adr/0019-integridade-dos-dados-de-mercado.md), [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md). |
| **Planejador de IR (Fase 3b + Fase 7c)** | `backend/apps/migration_ir/` | `GET /migration/wallet`, `POST /migration/plan`, `POST /migration/plan.pdf` (fpdf2). Núcleo puro (`wallet`, `tax`, `planner`) + `prices` (carteira → `apps.market`, `assets.lookup`) + router fino. Alvos livres, associação manual `sell`/`count_as`/`ignore`, isenção de R$ 20 mil só para `ACAO`, IR estimado por classe, `price_unavailable` em vez de preço inventado. Aceita planos de até exatamente 240 meses (`MAX_MONTHS = 240`); PDF protegido contra corte de blocos de mês entre páginas (`keep_together`) e repetição de cabeçalho. Stateless. [ADR 0015](adr/0015-planejador-de-ir.md), [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md). |
| **Otimizações DLP (Fase 3c + Fase 7c)** | `backend/apps/market/services/sources/dlp.py`, `backend/apps/api/routers/portfolio.py`, `backend/config/settings.py` | `/historico` cacheado **uma vez por `(usuário, geração)`** (`dlp:hist:<token_hash>:<gen>`) com filtros `ativo`/`classe`/`corretora` aplicados em memória (`_filter_historico`); filtro sobre coluna ausente retorna DataFrame vazio com warning; `_extract_distinct` deduplica com `pd.unique` antes de normalizar; URL base em `settings.DLP_BASE_URL` (default `https://users.dlpinvest.com.br`). Teste `integration` de equivalência com o filtro da DLP escrito e verificado ao vivo. [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md). |
| **Correções de revisão (Fases 2 / 2.5)** | `backend/apps/market/`, `backend/engine/` | Revisão pós-implementação (`code-review` + manual, 2026-08-31): 6 defeitos corrigidos com TDD — ver §2.3. |
| **Frontend SPA (Fase 4 + Fase 7c)** | `frontend/` | React 18 + Vite + Tailwind + TanStack Query v5 + ECharts (`echarts-for-react`) + Vitest. `api/` (Axios com CSRF, `errors.js` com `getApiError`/`shouldRetry`/parse de erro em `Blob`), `context/` (sessão, filtros com reset no logout, `period` sincronizado na URL via `?period=`), `hooks/` (`usePortfolioQueries`, `useMigrationPlan`, `useDlpTokenStatus`), `pages/` (Login preservando `location.state.from`, Portfolio com abas Performance/Risco, Migração de IR), 7 gráficos com rolagem horizontal na legenda + tabela anual + planejador com export PDF (nome via `Content-Disposition`). Token inválido tratado como estado único de página sem reabertura indevida de modal. Componente `Modal` acessível (`role="dialog"`, escape, foco). Campo de data com ícone de calendário e clique expandido (`showPicker`). [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md). |
| **Correções de revisão (Fases 3/4)** | `backend/apps/{market,migration_ir}/`, `frontend/` | Revisão pós-implementação (2026-10-27): plano [`2026-10-27-correcoes-revisao-fases-3-4.md`](superpowers/plans/2026-10-27-correcoes-revisao-fases-3-4.md) — ver §2.3. |
| **API da Carteira (Fase 3 + Fase 7c)** | `backend/config/api.py`, `backend/apps/api/`, `backend/apps/accounts/api.py` | `NinjaAPI` em `/api/v1/` (Swagger em `/api/v1/docs`), `ApiSessionAuth` (sessão + CSRF no envelope `ApiError`), `/auth/{csrf,login,logout,me}`, `/dlp/{token,token-status}` (gravação concorrente sem `IntegrityError`), `/portfolio/{filters,refresh,summary,twr-evolution,drawdown,rolling-metrics,risk-return,shadow,yearly-summary,irr-evolution}` (routers estritamente finos, orquestração centralizada em `analysis.py`), `/benchmarks/catalog`, `/assets/search`. [ADR 0013](adr/0013-contrato-da-api-v1.md), [ADR 0014](adr/0014-tickers-sob-demanda.md), [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md). |
| **Containers do App (Fase 5)** | `docker-compose.yml`, `backend/Dockerfile`, `backend/deploy/`, `frontend/Dockerfile`, `frontend/nginx.conf` | 3 serviços na rede externa `lab-net` (`name: appinvest`): `lab-appinvest-web` (`127.0.0.1:8080:80`, SPA + proxy `/api/` e `/admin/` + `/static/`), `lab-appinvest-backend` (Gunicorn `2w×4t`, `--timeout 180`), `lab-appinvest-ingestion` (Chromium Debian ARM64 + `supercronic` seg–sex 06:00/19:00 `America/Sao_Paulo`). `requirements.txt` (runtime) separado de `requirements-dev.txt` (host). Scraper B3 lê os 2 semestres (`b3:IFIX/SMLL/IDIV` atualizados até 26/10/2026) com fallback CSV que preserva o alerta de falha. [ADR 0016](adr/0016-empacotamento-no-lab.md). |
| **Paridade e remoção do legado (Fase 6)** | `scripts/parity/`, `reports/parity/` (local) | Harness em duas camadas (engine com mercado congelado, tolerância 1e-6; API real × legado ao vivo, 1e-4): 4/4 casos OK na camada 1, camada 2 sem divergência de cálculo. Linha de base da Fase 7c em `reports/parity/20261029-211840-fase7c/baseline`, recarregável sem `src`. Triagem corrigiu o cache de séries sem nome (`apps/market/cache.py`) e o buraco de CDI no Postgres. `src/` e libs legadas removidos. [ADR 0017](adr/0017-paridade-legado.md). |

> Driver Postgres: `psycopg[binary]` 3.x ([ADR 0004](adr/0004-psycopg-binary-3.md)). Sem
> `django-cors-headers` ([ADR 0005](adr/0005-sem-django-cors-headers.md)). Env via
> `python-dotenv` puro ([ADR 0010](adr/0010-python-dotenv-puro.md)).

### 2.1. ~~O que existe — legado Streamlit~~ (removido na Fase 6, 2026-10-27)

O app Streamlit (`src/ui`, `src/engine/financial_report.py`, `src/data/sources/market_data.py`,
`src/data/cache.py` com Upstash, `src/data/user_store.py` com Supabase), `scripts/sync_to_redis.py`,
o workflow `.github/workflows/update_b3.yml`, `redis.py`, o `requirements.txt` da raiz e os
`tests/` da raiz (portados em `backend/engine/tests/`) foram removidos depois da paridade
aprovada ([ADR 0017](adr/0017-paridade-legado.md)). O inventário do legado continua no
histórico do git (commit `3e8f155` e anteriores) e em [`historico/`](historico/README.md).

### 2.2. O que **não** existe ainda

- ~~**Sem Dockerfiles do app** nem compose final/nginx único — o backend roda no host (Fase 5). Só os containers de **infra** (`lab-postgres`, `lab-redis`) existem.~~ **Concluído na Fase 5 (2026-10-27):** `docker-compose.yml` com `lab-appinvest-{web,backend,ingestion}` rodando no LAB ([ADR 0016](adr/0016-empacotamento-no-lab.md)).
- ~~**Dados ainda no legado**: os tokens continuam no Supabase (a migração `user_tokens` → `UserToken` ocorre na fase correspondente); o cache já está integrado no `lab-redis`.~~ **Decidido na Fase 5:** Supabase descartado sem migração ([ADR 0016](adr/0016-empacotamento-no-lab.md)); tokens são cadastrados diretamente na SPA.
- ~~**Validação de paridade numérica e remoção do legado Streamlit (`src/`)**: Fase 6.~~ **Concluído em 2026-10-27** ([ADR 0017](adr/0017-paridade-legado.md)).
- ~~**Correções pós-paridade no `engine/`**: Fase 7.~~ **Fase 7a concluída em 2026-10-28** ([ADR 0018](adr/0018-metodologia-de-retorno-dlp.md)), **Fase 7b concluída em 2026-10-29** ([ADR 0019](adr/0019-integridade-dos-dados-de-mercado.md)) e **Fase 7c concluída em 2026-10-30** ([ADR 0020](adr/0020-backlog-spa-api-fase-7c.md)): integridade dos dados de mercado, cache com invalidação por geração global, re-busca na ingestão, `ffill` com aviso aditivo `meta.gaps` (exibido no `PeriodNotice` da SPA), benchmarks abrindo no último valor conhecido, correções de borda e cauda parada, TIR do período conforme convenção ANBIMA e resolução completa do backlog SPA/API. Próxima e última frente da Fase 7: **7d** otimização do engine (sem mudar números).
- ~~`.devcontainer/devcontainer.json` aponta para `app/web_app_v2.py` (**caminho stale**, não existe mais).~~ **Removido em 2026-10-26**: template Codespaces do app Streamlit (Python 3.11), incompatível com o fluxo atual (backend no host, Python 3.12; Docker só para `lab-postgres`/`lab-redis`).

### 2.3. Dívida técnica / lixo a resolver **durante** a migração (não carregar adiante)

- `financial_report.py` referenciava `self.logger` (inexistente) no branch de dados vazios — **corrigido** em `engine/dataset.py` na Fase 2.5 (mantido o registro histórico).
- ~~Algoritmos numéricos do engine (`irr.py`, `twr.py`, `metrics.py`) copiados literalmente na Fase 2.5 para manter validade de paridade da Fase 6; revisão profunda agendada para **pós-Fase 6** ([ADR 0012](adr/0012-engine-copia-literal-fase-2-5.md)).~~ **Resolvido na Fase 7a (2026-10-28, [ADR 0018](adr/0018-metodologia-de-retorno-dlp.md)):** fluxo externo com lucro realizado (`flows.py`), regra HPR aporte-início/resgate-fim (`twr.py`), XIRR Actual/365 invariante à escala com Newton + fallback `brentq` (`irr.py`), Sharpe contra a série real do CDI do período (`metrics.py`), abertura no fechamento anterior a `start_date`, corte estrito por `end_date`, recálculo pós-gap e remoção de dupla contagem de saldo inicial no shadow e `/summary` (`dataset.py`, `compute.py`). Validado contra o oráculo da planilha DLP de 123 meses (`test_oracle_dlp.py`).
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
  - ~~`test_sources_b3.py::test_fetch_b3_selenium_*` fica verde porque `ChromeDriverManager().install()` falha antes do mock — cobre o `raise FetchError`, mas pelo motivo errado.~~ **Corrigido na Fase 5 (2026-10-27):** `ChromeDriverManager.install` mockado explicitamente e adicionados testes de `CHROME_BINARY`/`CHROMEDRIVER_PATH` e leitura dos 2 semestres da B3.
  - Alguns testes de `engine/tests/test_dataset.py` usam `@pytest.mark.django_db` sem tocar o banco (monkeypatch total).
- ~~**Fase 3b — campos de `/resumo → wallet` não confirmados ao vivo**~~ — **confirmado em 2026-10-27**: `test_real_migration_wallet` verde com `DLP_TEST_TOKEN` real.
- **Fase 3b — regras fiscais simplificadas de propósito** (sem day-trade, sem compensação de prejuízo, BDR 15 %, classes complexas `not_estimated`) — ver [ADR 0015](adr/0015-planejador-de-ir.md).
- ~~**Fase 3c — equivalência do filtro em memória ainda não confirmada ao vivo**~~ — **confirmado em 2026-10-27** com token real: filtro em memória == `GET /historico?classe=…`/`?ativo=…` em linhas e valores. O teste só divergia no dtype de `date` (o envelope do cache devolve `datetime64`); corrigido no teste.
- **Revisão pós-Fases 3/3b/3c e 4 (2026-10-27) — corrigido com TDD** (plano [`2026-10-27-correcoes-revisao-fases-3-4.md`](superpowers/plans/2026-10-27-correcoes-revisao-fases-3-4.md)):
  1. Preset "Tudo" (padrão da SPA) mandava `start: null`, que a API trata como "1 ano" — agora envia `1900-01-01` e mostra `meta.start/end` + `meta.missing`.
  2. Planejador da SPA marcava `sell` por padrão (contra a ADR 0015) — agora `ignore`; campos de caixa inicial e "apenas aportes" expostos; tolerância de pesos igual à do backend (1e-6).
  3. DLP fora do ar: o single-flight serializava timeouts de 60 s — falha `DLPUnavailableError` fica 30 s em cache (`<chave>:err`, liberada pelo refresh); `requests.RequestException` e payload não-objeto viram `DLPUnavailableError` (antes 500).
  4. Planejador sem limite de meses — teto mínimo R$ 100, `MAX_MONTHS = 240`, folga `_EPS` em reais e valores não finitos rejeitados.
  5. SPA: `retry` só em 5xx/rede; queries só com token configurado (fim dos 9× 409 e do modal reabrindo); erros de PDF em `Blob` legíveis; estados de erro com "Tentar novamente"; % de risco sempre convertido de fração; datas locais nos presets; avisos do plano traduzidos e `ir_not_estimated` exibido; plano limpo ao mudar entradas.
- ~~**Backlog da revisão de 2026-10-27:**~~ **Resolvido na Fase 7c (2026-10-30, [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md)):**
  - lock do `single_flight` com uuid próprio do dono e liberação Lua atômica;
  - geração do cache DLP iniciada em `time_ns()` contra colisão em restart ou despejo do Redis;
  - corrida no `save_token` tratada contra `IntegrityError` em submissões concorrentes;
  - 422 de schema do Ninja envelopado em `ApiError(code="invalid_request")`;
  - router `/portfolio/summary` desprovido de cálculos diretos, orquestrado via `analysis.py`;
  - `start` default normalizado na análise analítica;
  - filtro DLP sobre coluna ausente retorna DataFrame vazio com warning;
  - layout do PDF protegido com `keep_together` nos blocos mensais e repetição de cabeçalho;
  - rate limit de login implementado (5 tentativas/15 min por usuário e IP, `429 too_many_attempts`);
  - download de PDF com nome extraído do header `Content-Disposition` e `revokeObjectURL` diferido;
  - login restaura rota de origem através de `location.state.from`;
  - filtros de análise zerados no logout;
  - modal de token com estado limpo em cancelamento;
  - acessibilidade do `Modal` com `role="dialog"`, `aria-modal="true"`, gestão de foco e tecla Escape;
  - `period` na URL sincronizado com `?period=1a|2a|5a|tudo|custom` e `replace: true`;
  - código morto removido (`data?.data?.x`, branch shorthand).
  - **Itens fechados sem código / adiados (§2.3 / ADR 0020):**
    - **busca de ticker sem debounce:** **já resolvido** (`BenchmarkSelector` tem `debouncedQuery`);
    - **`catalog.defaults` não usado pela SPA:** **adiado**. Remover quebraria o contrato v1, e usar exige decidir se o padrão da SPA (`CDI`, `BOVA11`) passa a vir do servidor. Fica para quando houver preferência por usuário;
    - **bundle de 1,4 MB:** desempenho, **adiado** para a Fase 7d ou posterior (spec §4);
    - **`MAX_MONTHS` depois de `resolve_target_prices`:** mantido como aceito ("anotar se a busca ficar cara").
- ~~**Menores da revisão final da branch de correções (2026-10-27):**~~ **Resolvido na Fase 7c (2026-10-30, [ADR 0020](adr/0020-backlog-spa-api-fase-7c.md)):**
  1. ~~**Token DLP rejeitado (422) com `configured: true`:**~~ resolvido como estado único de página (`TokenState`), sem interceptor de rede disparando modais repetidos, e `DLPAuthError` em cache por geração (30 s).
  2. ~~**`PeriodNotice` fora de contexto:**~~ renderizado somente com `tokenConfigured === true`.
  3. ~~**"Personalizado" depois de "Tudo"** mostra `01/01/1900`:~~ campo de data inicial normalizado com base real ou limpo.
  4. ~~**`MAX_MONTHS` pode recusar um plano de exatamente 240 meses:**~~ corrigido em `planner.py` (checa o limite estritamente ao adicionar lotes a um mês novo, aceitando exatamente 240 meses).
  5. **`MAX_MONTHS` é checado depois de `resolve_target_prices`:** mantido como aceito.
- ~~`redis.py` na raiz (script solto de 12 linhas) faz *shadow* do pacote `redis`.~~ Removido na Fase 6.
- ~~`ativos.csv` duplicado (raiz + `data/`).~~ Removido da raiz na Fase 6 (fica `backend/apps/market/data/ativos.csv`).
- ~~Arquivos de scratch do Gemini/Antigravity commitados (`.gemini/antigravity/.../scratch/*.py`).~~ Removidos na Fase 6.
- ~~`requirements.txt`: **16 libs sem nenhum pin de versão**; `selenium` e `webdriver-manager`
  são usados mas **não constam** do arquivo (instalados ad-hoc na Action).~~ Removido na Fase 6 (o backend tem `requirements.txt`/`requirements-dev.txt` próprios).
- ~~Páginas 2 e 3 importam `from src...` sem patch de `sys.path`~~ — some com a remoção de `src/` (Fase 6).
- ~~`deployment_guide.md` contém caminhos locais Windows stale.~~ Removido na Fase 6.
- **Backlog aberto pela paridade (Fase 6, [ADR 0017](adr/0017-paridade-legado.md)):**
  1. Cache `market:series:*` não é invalidado quando a ingestão grava: até 24 h de série defasada (apareceu durante o catch-up da carga inicial).
  2. `ingest_market_data` incremental só olha `last_date` e nunca preenche buraco no meio da série (causa do buraco de 116 dias no CDI, já re-ingerido).
  3. Buracos vindos da fonte: `bcb:12469` (IMA-S, 2024, 275 dias) e `yf:LFTS11.SA` (2025, 151 dias).
  4. Container `lab-appinvest-backend` precisa de rebuild para levar a correção do cache de séries sem nome (`fad1dc8`).

### 2.4. Dependências externas a eliminar

| Dependência | Papel | Situação atual | Nova arquitetura no LAB |
|---|---|---|---|
| **Supabase Cloud** | Armazenamento de tokens DLP (Fernet) | Nuvem (REST, service_role) | **Substituído** por PostgreSQL 16 local (`lab-postgres`), via modelo `UserToken`. **Supabase descartado sem migração de dados** (serviço legado instável e sem dados relevantes; cadastro de token pela SPA — [ADR 0016](adr/0016-empacotamento-no-lab.md)). |
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

### 3.5. Índices B3: Selenium mantido (Chromium ARM64)

- `fetch_b3_selenium` (porta de `buscar_dados_b3`) é **mantido**.
- Na VM `aarch64` (ARM64) do LAB não há Google Chrome oficial e o `webdriver-manager` baixa driver x86 incompatível; por isso o target `ingestion` de `backend/Dockerfile` instala **`chromium` + `chromium-driver` do Debian Bookworm (`arm64`)** e define `CHROME_BINARY=/usr/bin/chromium` e `CHROMEDRIVER_PATH=/usr/bin/chromedriver` ([ADR 0016](adr/0016-empacotamento-no-lab.md)). O scraper itera pelos dois semestres (`1º semestre` e `2º semestre`) da tabela da B3.
- **Selenium só na ingestão.** Nenhum endpoint da API dispara scraping (a imagem `backend` não contém navegador).
- **Fallback de leitura:** se o Selenium falhar num índice durante `ingest_market_data`, o comando lê `read_b3_static` (`data/static/*_all.csv`) e grava o histórico, mas **continua marcando a série como falha (exit 1)** para não mascarar quebra do scraper ([ADR 0016](adr/0016-empacotamento-no-lab.md)).

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

> Tudo dentro de `/data/projetos/app-investimentos`, branch `migracao-django`. O `src/`
> (Streamlit) ficou durante a transição para a **validação de paridade** e foi removido na
> Fase 6 (2026-10-27). Na raiz ficou `scripts/parity/` (linha de base da Fase 7, sem `src`,
> [ADR 0017](adr/0017-paridade-legado.md)).
>
> **Nota:** os blocos `config/`, `apps/accounts/`, `apps/market/`, `apps/api/` e `engine/` abaixo refletem o que foi **efetivamente
> construído** nas Fases 1, 2, 2.5, 3 e 3b.
> `frontend/` foi construído na Fase 4 com layout diferente do alvo abaixo: `src/{api,context,hooks,pages}/` + `components/{common,dlp,layout,portfolio/{charts,tables},migration}/` (7 gráficos: TWR, TIR, Shadow, Drawdown, Sharpe/Vol móveis, Risco×Retorno + `YearlyTable`). O desenho abaixo fica como registro do alvo original.
> Estrutura viva e detalhada do backend em [`../backend/README.md`](../backend/README.md).

```
/data/projetos/app-investimentos/
│
├── docker-compose.yml                    # [FASE 5] lab-appinvest-{web,backend,ingestion} na rede externa lab-net
│
├── backend/                              # [NOVO] Django 5 + Django Ninja
│   ├── Dockerfile                        # [FASE 5] multi-target: backend (Gunicorn, sem navegador) e ingestion (Chromium ARM64 + supercronic)
│   ├── deploy/                           # [FASE 5] entrypoint-web.sh (migrate + collectstatic + gunicorn) e crontab (06:00/19:00 seg-sex)
│   ├── requirements.txt                  # dependências PINADAS de runtime (inclui selenium, webdriver-manager)
│   ├── requirements-dev.txt              # [FASE 5] -r requirements.txt + pytest*, model-bakery, matplotlib (host)
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
│   │   └── migration_ir/                 # [FASE 3b] Planejador de Isenção de IR (ADR 0015)
│   │       ├── wallet.py                 # /resumo -> Holding (puro)
│   │       ├── tax.py                    # regra de isenção por classe + IR estimado (puro)
│   │       ├── planner.py                # validação, guloso no teto, rebalanceamento (puro)
│   │       ├── prices.py                 # cotação dos alvos (carteira -> apps.market)
│   │       ├── schemas.py / pdf.py       # schemas Ninja; PDF fpdf2
│   │       └── api.py                    # /migration/{wallet,plan,plan.pdf}
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

### Fase 3b — Planejador de Migração de IR (`apps/migration_ir/`) — **concluída em 2026-10-26** ([ADR 0015](adr/0015-planejador-de-ir.md))
1. `apps/migration_ir/api.py`:
   - `GET  /api/v1/migration/wallet` — carteira atual (via `/resumo` da DLP).
   - `POST /api/v1/migration/plan` — recebe alocação-alvo, retorna plano de compras mensais
     (respeitando o limite de R$ 20.000/mês).
   - `POST /api/v1/migration/plan.pdf` — mesmo cálculo, retorna os **bytes do PDF** (`fpdf2`);
     o React apenas dispara o download.

### Fase 3c — Otimizações DLP sem impacto no engine (mapeado em 2026-10-26; escopo reduzido em 2026-10-26; **concluída em 2026-10-26**)
> **Referência técnica:** auditoria realizada sobre [`referencia-api-dlp-invest.md`](referencia-api-dlp-invest.md) cruzada com `apps/market/services/sources/dlp.py`, `engine/dataset.py`, `engine/twr.py` e `apps/api/routers/portfolio.py`.
> **Escopo:** só o que **não muda números nem toca `engine/`** — preserva a paridade da Fase 6 ([ADR 0012](adr/0012-engine-copia-literal-fase-2-5.md)). Entra **antes da Fase 4** (a SPA troca filtros o tempo todo). Os itens que tocam o engine ou mudam números foram movidos para a **Fase 7**.

1. **Cache único de `/historico` + filtragem em memória (`dlp.buscar_historico`):** como `GET /historico` sem filtros já traz obrigatoriamente `ativo`, `classe` e `corretora` em todas as linhas (`HistoricModel`) e já é chamado no carregamento inicial por `GET /api/v1/portfolio/filters`, cachear apenas o DataFrame completo por `(token_hash, generation)` no Redis e aplicar os filtros `ativo`/`classe`/`corretora` em memória no Pandas. Elimina chamadas HTTP extras (até 60s) à DLP ao trocar filtros na UI e reduz o armazenamento no Redis para 1 DataFrame por usuário.
2. **Extração rápida de filtros (`portfolio._extract_distinct`):** substituir `df[col].dropna().astype(str).map(str.strip)` na coluna inteira do histórico diário por `pd.unique(df[col].dropna())` antes da normalização e ordenação.
3. **Domínio oficial DLP:** parametrizado em `settings.DLP_BASE_URL` (default `https://users.dlpinvest.com.br`; antes `https://users.dlombelloplanilhas.com`, fixo em `dlp._BASE`).
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

### Fase 5 — Dockerização e unificação — **concluída em 2026-10-27** ([ADR 0016](adr/0016-empacotamento-no-lab.md))
1. `backend/Dockerfile` multi-target: `backend` (Gunicorn, sem navegador) e `ingestion` (com `chromium` + `chromium-driver` ARM64 + `supercronic`).
2. `frontend/Dockerfile` (multi-stage: build Vite → `nginx:alpine` com `frontend/nginx.conf`).
3. `docker-compose.yml` (`name: appinvest`, rede externa `lab-net`): `lab-appinvest-web` (`127.0.0.1:8080:80`, serve a SPA, `/static/` e faz proxy de `/api/` e `/admin/`), `lab-appinvest-backend` e `lab-appinvest-ingestion` (`manage.py ingest_market_data` agendado via `supercronic` às 06:00 e 19:00 seg–sex).
4. Carga inicial de mercado: ~~import dos tokens do Supabase (§7.2)~~ (cancelado — Supabase descartado) + `backfill_market_data` e `ingest_market_data` inicial no container de ingestão (`b3:IFIX/SMLL/IDIV` atualizados até 2026-10-26).

### Fase 6 — Validação de paridade e remoção do legado — **concluída em 2026-10-27** ([ADR 0017](adr/0017-paridade-legado.md))
1. Harness `scripts/parity_check.py` (removido junto com o legado) + pacote `scripts/parity/`: camada 1 (engine, mercado congelado do Postgres nos dois lados, 1e-6) e camada 2 (legado ao vivo × API HTTP dos containers, 1e-4 → `INVESTIGAR`).
2. Matriz: histórico completo, últimos 12 meses, meio da carteira → hoje, e histórico completo com filtro de classe; benchmarks `CDI`, `IPCA + 6%`, `S&P 500 BRL`, `IFIX` (Tesouro e carteiras compostas fora).
3. Resultado: camada 1 toda OK; camada 2 sem divergência de cálculo — divergências de mercado/infra explicadas e aceitas; teste `integration` da Fase 3c verde com token real.
4. Linha de base da Fase 7 salva em `reports/parity/20261027-182533/baseline/` (local).
5. `src/`, scripts e workflow legados, `tests/` da raiz e resíduos removidos; `streamlit`, `altair`, `pydeck`, `supabase*`, `upstash-redis`, `webdriver-manager`, `authlib` desinstalados do `backend/.venv`.

### Fase 7 — Correções e otimizações do engine pós-paridade (movido da 3c em 2026-10-26)
> Dividida em quatro frentes ([spec](superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md) §5): **7a** engine (alteram números), **7b** integridade dos dados de mercado, **7c** backlog SPA/API, **7d** otimização sem mudar números (por último).

#### Fase 7a — Correções matemáticas e confiabilidade — **concluída em 2026-10-28** ([ADR 0018](adr/0018-metodologia-de-retorno-dlp.md))
1. **Fórmula do fluxo externo líquido (`engine.flows`):**
   - $\text{fluxo}_t = \Delta \text{vlr\_investido}_t - \text{lucro}_t - \text{proventos}_t$, descontando o lucro realizado das vendas e proventos recebidos. Na 1ª linha, $\Delta\text{vlr\_investido}_0 = \text{vlr\_investido}_0$; em recortes, o 1º fluxo é o patrimônio de abertura (item 5).
2. **Regra de HPR aporte-início / resgate-fim (`engine.twr`):**
   - Aporte líquido ($\text{fluxo} > 0$): na base do retorno mensal ($\text{pat\_fim} / (\text{pat\_ini} + \text{fluxo})$).
   - Resgate líquido ($\text{fluxo} < 0$): no fim do mês ($(\text{pat\_fim} - \text{fluxo}) / \text{pat\_ini}$).
   - Corrigiu distorção histórica: TWR acumulado em 10 anos passou de 40,87% (legado) para 195,95% (DLP real).
3. **XIRR Actual/365 invariante à escala com Newton + fallback `brentq` (`engine.irr`):**
   - Convenção $(1 + r)^{(d_i - d_0)/365.0}$, ordenação cronológica e agregação de datas.
   - Normalização por $\max(|F_i|)$ contra overflow numérico; tentativa Newton-Raphson com fallback por intervalo (`scipy.optimize.brentq`, dependência `scipy~=1.18`) com teto adaptativo.
4. **Sharpe ratio contra série real de CDI (`engine.metrics`):**
   - Substituição de taxa fixa ou nula pela série diária de CDI do período analisado para carteira e benchmarks.
5. **Recorte de período e recálculo pós-gap (`engine.dataset`):**
   - **Abertura no último fechamento `<= start_date`** (revisão pós-teste do usuário, 2026-10-28): cada linha da DLP é um fechamento mensal; "01/01–28/10" abre em 31/12 e inclui o retorno de janeiro, como o filtro "jan–set" da DLP. Antes, o 1º mês do recorte era descartado (TWR 5,66 % × 12,09 % e XTIR 9,18 % × 16,72 % a.a. na janela jan–set/2026).
   - Corte estrito de `portfolio_df` em `date <= end_date`; o fim é o último mês fechado (com `end` = hoje, set/2026).
   - `fluxo[0]` = patrimônio de abertura após o ajuste de `gap` em janelas com `start_date` intermediário.
6. **Métricas de retorno total e shadow (`engine.compute`):**
   - Eliminação da dupla contagem do saldo inicial nos aportes de `/summary` e na simulação da carteira `/shadow`.
7. **Validação e linha de base:**
   - Oráculo DLP (`engine/tests/test_oracle_dlp.py`): 123/123 meses no fluxo, 122/122 no TWR mensal ($< 0{,}005\text{ pp}$), TWR acumulado idêntico (195,95%) e janela jan–set/2026 igual ao dashboard (TWR 12,09 %, XTIR 16,72 % a.a.).
   - Regressão contra a linha de base da Fase 6 comprovou mudanças restritas à `Carteira` nas métricas de retorno e benchmarks intocados em `twr`/`drawdown`/`rolling_vol`/`yearly`.
   - Matriz de paridade com `w4-fim-passado` e `w5-inicio-mes` (start fora de fechamento). Linha de base vigente: `reports/parity/20261028-180012-fase7-abertura/baseline` (substitui `20261028-163150-fase7`).

#### Fase 7b — Integridade dos dados de mercado — Concluída em 2026-10-29 ([ADR 0019](adr/0019-integridade-dos-dados-de-mercado.md))
Invalidação do cache `market:series:*` na ingestão via geração global no Redis (`market:series:gen`), módulo puro `gaps.py` com detecção por cadência (10d diária / 45d mensal), comando `market_gaps`, re-busca automática na ingestão, buracos de fonte tratados com `ffill` + aviso aditivo em `meta.gaps` (exibido na SPA no `PeriodNotice`), benchmarks com abertura no último valor conhecido via lookback de 40 dias e nova linha de base `reports/parity/20261029-122417-fase7b/baseline`.

#### Fase 7c — Backlog SPA/API e Achados do Teste 7b — Concluída em 2026-10-30 ([ADR 0020](adr/0020-backlog-spa-api-fase-7c.md))
1. **Parte A — Achados do teste da 7b com o usuário:**
   - Detecção de lacunas que cruzam bordas da janela e cauda parada delimitada em `min(crop_end, _today())` em `apps/market/gaps.py`, `engine/dataset.py` e comando `market_gaps`.
   - Tabela anual (`compute_yearly_summary`) omitindo o ano que contém exclusivamente o fechamento da data de abertura (sem linhas de retorno do ano). Paridade regerada para `reports/parity/20261029-211840-fase7c/baseline`.
   - Legendas de gráficos ECharts com rolagem horizontal (`type: 'scroll'`), evitando sobreposição com títulos e áreas de plotagem.
   - TIR exibida para horizontes menores que 12 meses sem anualização, expressando o retorno efetivo do período e estendendo a API com `xirr_period`, `xirr_is_annual` e `annual_from` conforme convenção ANBIMA.
   - Campo de data nativo com suporte a clique expandido (`showPicker`) e ícone de calendário com `color-scheme: dark`.
2. **Parte B — Resolução do backlog SPA/API e robustez operacional:**
   - Rate limit de login: 5 tentativas/15 min por (usuário, IP) com `429 too_many_attempts`.
   - Erros de validação do Ninja envelopados em `ApiError(code="invalid_request")` (HTTP 422).
   - Período sincronizado na URL (`?period=1a|2a|5a|tudo|custom`) com `replace: true`, limpeza de filtros no logout e retorno à rota de origem (`location.state.from`).
   - Token DLP rejeitado/ausente tratado como estado único de página (`TokenState`), sem reabertura indevida de modais via rede.
   - `single_flight` com uuid próprio do dono e liberação Lua atômica; geração do cache DLP baseada em `time_ns`; cache temporário de `DLPAuthError`.
   - Filtro sobre coluna ausente retornando DataFrame vazio com warning; e limite de 240 meses aceito (`MAX_MONTHS = 240`).
   - Paginação e layout do PDF do planejador com blocos mensais protegidos (`keep_together`), repetição de cabeçalhos de tabela e nome no cabeçalho `Content-Disposition`.
   - Acessibilidade do componente `Modal` (`role="dialog"`, `aria-modal="true"`, foco e Escape) e limpeza de código morto.

#### Fase 7d — Otimizações (sem mudar números) — **por último**
1. **Pré-processamento de `date` e remoção de `groupby` duplicado (`dlp.py`, `dataset.py`, `twr.py`):** converter `df["date"] = pd.to_datetime(df["date"])` uma única vez antes de `mc.set_df` em `dlp.buscar_historico` e evitar que `calculate_twr` refaça `pd.to_datetime`, `groupby("date").sum()` e cálculo de `fluxo` quando `fetch_user_portfolio` já entrega o DataFrame agrupado por data.

---

## 6. Seções que faltavam na v1

### 6.1. Autenticação SPA → API
Ver §3.6. Resumo: login Django + cookie de sessão, mesma origem via nginx (sem CORS), CSRF
habilitado, `django_auth` do Ninja nas rotas protegidas. SSO Google fica para fase posterior
opcional (`django-allauth`).

### 6.2. ~~Migração dos tokens do Supabase~~ (Cancelado na Fase 5 — [ADR 0016](adr/0016-empacotamento-no-lab.md))
- ~~Script/`data migration` (`apps/accounts/migrations/0002_import_supabase_tokens.py` **ou** um `manage.py import_supabase_tokens`).~~
- **Motivo do cancelamento:** o Supabase legado encontra-se com problemas e não há dados de usuário relevantes para importar. Usuários são criados via `createsuperuser`/`/admin` e o token DLP é cadastrado pela SPA (a `FERNET_KEY` legada continua sendo usada normalmente — §6.3).

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

### 7.4. Agendador da ingestão (Decidido na Fase 5 — [ADR 0016](adr/0016-empacotamento-no-lab.md))
**Decidido:** **`supercronic`** dentro do container `lab-appinvest-ingestion` (`backend/deploy/crontab`: `0 6,19 * * 1-5`, `TZ=America/Sao_Paulo`), logando em stdout. Preferido a `ofelia` por não exigir montagem do socket do Docker (`/var/run/docker.sock`) dentro do container.

### 7.5. Selenium em container
Fonte de fragilidade. Mitigações implementadas na Fase 5 ([ADR 0016](adr/0016-empacotamento-no-lab.md)): target dedicado (`ingestion`), Chromium + `chromium-driver` nativos do Debian ARM64 (`CHROME_BINARY`/`CHROMEDRIVER_PATH`), leitura dos 2 semestres da tabela da B3 e **fallback automático** para `data/static/*_all.csv` que preserva o histórico inicial mas mantém a série marcada em `failed_keys` (exit 1).

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

# Fase 6 — paridade (executada em 2026-10-27; o CLI saiu junto com src/ — ver ADR 0017)
# export DLP_TOKEN=... PARITY_USER=... PARITY_PASSWORD=...   # só por env, nunca por argumento
# backend/.venv/bin/python scripts/parity_check.py --layers 1,2 --with-integration
# Fase 7 — regressão contra a linha de base (sem src)
cd scripts/parity && ../../backend/.venv/bin/pytest -q
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
- [x] Paridade: números do Streamlit antigo == números da API nova (dentro da tolerância) — 2026-10-27, [ADR 0017](adr/0017-paridade-legado.md).

---

## 9. Ordem de execução recomendada (checklist)

1. [x] **Fase 0** — `docker-compose.yml` com Postgres + Redis; validar.
2. [x] **Fase 1** — Django + `settings.py` (Fernet reusada) + `UserToken`/`MarketSeries` + Admin.
3. [x] **Fase 2** — quebrar `market_data.py` em `apps/market/services/`; portar cache; `ingest_market_data` + `backfill_market_data`.
4. [x] **Fase 2.5** — `engine/compute.py` (extrair dos `plot_*`); `report_legacy.py`; corrigir `self.logger`.
5. [x] **Fase 3** — endpoints Ninja da carteira (auth, token DLP, 8 análises + filters/refresh, benchmarks, busca e fetch-on-read de tickers); Swagger em `/api/v1/docs`.
6. [x] **Fase 3b** — Planejador de Isenção de IR (`/migration/*` + PDF `fpdf2`).
7. [x] **Fase 3c** — Otimizações DLP sem impacto no engine (cache único de `/historico` com filtro em memória, `unique()` em `/filters`, domínio `dlpinvest.com.br` parametrizado) (ver §Fase 3c).
8. [x] **Fase 4** — SPA React (auth, filtros, gráficos ECharts, tabela anual, planejador de IR) + correções da revisão de 2026-10-27.
9. [x] **Fase 5** — Dockerfiles (`backend` multi-target + `frontend` multi-stage), `docker-compose.yml` (`lab-appinvest-{web,backend,ingestion}` em `127.0.0.1:8080`), Chromium ARM64 + `supercronic` ([ADR 0016](adr/0016-empacotamento-no-lab.md)).
10. [x] **Carga inicial de mercado** — ~~import dos tokens do Supabase~~ (cancelado, ADR 0016); `backfill_market_data` + `ingest_market_data` inicial no container (`b3:*` atualizado até 2026-10-26).
11. [x] **Fase 6** — validação de paridade (2 camadas, aprovada em 2026-10-27); `src/` e libs legadas removidos ([ADR 0017](adr/0017-paridade-legado.md)).
12. [x] **Fase 7a** — pós-paridade: correção matemática do engine alinhada à planilha DLP (fluxo com lucro, HPR aporte-início/resgate-fim, XIRR com fallback, rf = CDI, recortes e gap) concluída em 2026-10-28 ([ADR 0018](adr/0018-metodologia-de-retorno-dlp.md)).
13. [x] **Fase 7b** — integridade dos dados de mercado (cache com geração global, re-busca na ingestão, ffill + meta.gaps na API e PeriodNotice na SPA, benchmarks com lookback de 40 dias) concluída em 2026-10-29 ([ADR 0019](adr/0019-integridade-dos-dados-de-mercado.md)).
14. [x] **Fase 7c** — backlog SPA/API e achados do teste da 7b concluída em 2026-10-30 ([ADR 0020](adr/0020-backlog-spa-api-fase-7c.md)); depois **7d** otimização sem mudar números.
15. [ ] **Fase 7d** — otimizações sem mudar números (spec §5.7).
16. [ ] Decidir prefixo `lab-` / merge de `migracao-django` (§7.3).
