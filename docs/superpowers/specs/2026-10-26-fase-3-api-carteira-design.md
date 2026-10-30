# Design — Fase 3 (API da Carteira em Django Ninja)

> **Data:** 2026-10-26
> **Branch:** `migracao-django`
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md)
> **Fases anteriores:** 0 + 1 (2026-08-28) e 2 + 2.5 (2026-08-31) — specs
> [`2026-08-27-fundacao-backend-fases-0-1-design.md`](2026-08-27-fundacao-backend-fases-0-1-design.md) e
> [`2026-08-31-fase-2-e-2-5-servicos-e-engine-design.md`](2026-08-31-fase-2-e-2-5-servicos-e-engine-design.md).
> **Escopo deste design:** terceiro incremento. Cobre a API HTTP (Django Ninja) que a SPA da
> Fase 4 vai consumir: autenticação de sessão, token DLP, análises da carteira, catálogo de
> benchmarks e comparação contra qualquer ticker do `ativos.csv`. **O planejador de IR fica
> para a Fase 3b** (spec própria).

---

## 1. Contexto

Depois das Fases 2 + 2.5 o backend calcula tudo — `apps.market.series` serve séries de mercado
do Postgres/Redis, e `engine/{dataset,compute}.py` monta o dataset e produz os 7 cálculos
puros (`compute_twr_evolution`, `compute_drawdown`, `compute_risk_return`,
`compute_rolling_metrics`, `compute_irr_evolution`, `compute_shadow`, `compute_yearly_summary`).
Mas **não há porta de entrada HTTP**: `config/urls.py` só tem `/admin/` e um stub comentado do
Ninja ([ADR 0009](../../adr/0009-django-ninja-stub-desde-fase-1.md)). 125 testes verdes.

O Streamlit legado (`src/ui/pages/4_configuracoes.py`) faz, num único botão "Gerar Análises":
lê o token DLP → `fetch_user_portfolio` da **carteira inteira** no período → `build_dataset`
com os benchmarks marcados → guarda na sessão; as páginas 1/2 e o dashboard leem dali. A
seleção de comparação (`src/ui/components/sidebar.py`) permite escolher **qualquer ticker** de
`data/ativos.csv` (~7.500 linhas) além dos índices do catálogo, adiciona automaticamente uma
variante "em Reais" para ativos estrangeiros e permite criar sintéticos `IPCA +`, `CDI +`,
`% do CDI`.

Esta fase expõe esse fluxo como API REST JSON, multi-usuário, sem mudar nenhum cálculo do
engine (paridade da Fase 6 — [ADR 0012](../../adr/0012-engine-copia-literal-fase-2-5.md)).

---

## 2. Decisões travadas (brainstorming 2026-10-26)

Decisões **do Bruno** (respondidas no brainstorming):

| # | Tema | Decisão |
|---|---|---|
| 1 | Escopo | Fase 3 = auth + token + carteira + benchmarks + tickers. **Planejador de IR → Fase 3b**, spec separada. |
| 2 | Granularidade | **Um endpoint por gráfico** (não um endpoint único de análise). |
| 3 | Sintéticos | **Estruturados livres** `{base, operador, valor}` — o usuário cria qualquer sintético válido, não só os do catálogo. |
| 4 | Tokens do Supabase | Importação **fica na Fase 5** (roteiro original). Na Fase 3 o token é cadastrado pelo endpoint/Swagger. |
| 5 | Cadastro de usuário | **Só pelo admin** (`/admin` ou `createsuperuser`). Sem endpoint público de registro. |
| 6 | Filtro `ativo`/`classe` da carteira | **Entra, como funcionalidade nova** (o Streamlit nunca usou — fica fora da comparação de paridade). Inclui endpoint que lista as opções. |
| 7 | Atualizar carteira | **Sim** — mecanismo explícito que ignora/invalida o cache DLP de 10 min. |
| 8 | Erros da DLP | **Códigos distintos por tipo** (token ausente, token rejeitado, DLP indisponível, carteira vazia). |
| 9 | Comparação contra tickers | **Sim, sob demanda**: qualquer ticker do universo do `ativos.csv` (com variante BRL), buscado no Yahoo no primeiro uso, gravado em `MarketSeries` e servido do banco depois. Inclui busca de tickers. |
| 10 | Definition of Done | Testes automáticos verdes (DLP simulada) **+** roteiro manual no Swagger com o token real do Bruno. |

Decisões **do designer** (calls razoáveis; registrar em ADR, podem ser revistas):

| # | Tema | Decisão | Por quê |
|---|---|---|---|
| D1 | Verbo dos endpoints de análise | **`POST` com corpo JSON** (`AnalysisRequest`), não `GET` com query string. | O pedido tem lista de sintéticos estruturados + lista de benchmarks + filtros; em query string isso vira mini-linguagem de string (o que a decisão 3 quis evitar). Custo: exige CSRF, que a SPA já precisa de qualquer forma. |
| D2 | "Atualizar" | **Endpoint próprio** `POST /portfolio/refresh` que invalida o cache DLP **do usuário**; os gráficos recarregam normalmente depois. | Um flag `refresh` em cada endpoint dispararia ~8 chamadas DLP paralelas (uma por gráfico). |
| D3 | HTTP status de erro DLP | Token rejeitado → **422**, **nunca 401**. | 401 é reservado para "sem sessão" — se a DLP rejeitar o token e a API devolver 401, a SPA confundiria com sessão expirada e mandaria para o login. |
| D4 | Formato de série | `{"name": str, "data": [["YYYY-MM-DD", float|null], ...]}` | Formato nativo de eixo `time` do ECharts; datas ISO evitam ambiguidade de fuso de timestamp epoch. |
| D5 | Unidades | **A API não converte unidades**: devolve o que o engine devolve e documenta a unidade em cada schema. | Paridade da Fase 6 compara números do engine; conversão é da SPA. |
| D6 | Endpoint extra | Incluir **`/portfolio/risk-return`** (dispersão risco × retorno). | Está no Streamlit (`2_risco.py`) e no engine, mas faltou na lista do doc mestre §5. |
| D7 | Variante BRL | Pedida **explicitamente** pelo cliente (`"AAPL BRL"`), não adicionada automaticamente. | A SPA decide a UX; a API fica previsível. |
| D8 | Cópia do `ativos.csv` | Copiar para `backend/apps/market/data/ativos.csv` (dado empacotado do app). | O backend precisa ser autocontido para a imagem Docker da Fase 5; as cópias da raiz somem com `src/` na Fase 6. |
| D9 | Ingestão dos tickers sob demanda | `ingest_market_data` passa a atualizar também os tickers `yf:` já presentes em `MarketSeries` que não estão no catálogo. | Sem isso, um ticker buscado uma vez fica parado na data do primeiro uso. |
| D10 | Identificador de ticker | O identificador público de um ticker é o **ticker Yahoo** (`PETR4.SA`, `GDX.L`, `GDX`), não a sigla. Variante BRL = `"<ticker yahoo> BRL"`. | 14+ siglas do universo aparecem em duas bolsas (ex.: `GDX` em Londres e NYSE); a sigla sozinha faria uma listagem sobrescrever a outra em silêncio. O legado diferenciava pelo nome de exibição `"GDX - ETF_GB"`. |
| D11 | Histórico do ticker sob demanda | O **primeiro** fetch puxa o histórico completo (mesmo `_EPOCH` da ingestão), não só a janela pedida. Top-up falho com dados já gravados → usa o que há no banco (log `warning`); só vai para `missing` se o banco não tem nada no período. | Evita série começando tarde quando depois se pede uma janela maior (rebase desalinhado sem erro), e evita perder um ticker bom por um feriado longo. |
| D12 | Rajada de chamadas DLP | **Single-flight por usuário**: `dlp.buscar_historico` envolve o fetch num lock Redis (`cache.lock` do `django-redis`) por chave de cache; quem chega depois espera o lock e lê do cache. Idem para o fetch de ticker sob demanda (lock por `series_key`). | Com um endpoint por gráfico (decisão 2), o primeiro carregamento dispara ~8 POSTs paralelos com cache frio; sem lock, seriam ~8 chamadas `/historico` de até 60s cada, prendendo todos os workers. |

---

## 3. Escopo e Definition of Done

### 3.1 Dentro deste design

- `NinjaAPI` montada em `/api/v1/`, Swagger em `/api/v1/docs`.
- Auth de sessão: `csrf`, `login`, `logout`, `me`.
- Token DLP: salvar, consultar status, remover.
- Carteira: `filters`, `refresh`, `summary`, `twr-evolution`, `drawdown`, `rolling-metrics`,
  `risk-return`, `shadow`, `yearly-summary`, `irr-evolution`.
- Benchmarks: `catalog`; tickers: `assets/search`.
- Tickers sob demanda em `apps.market` (resolução via `ativos.csv`, fetch Yahoo, persistência).
- Erros DLP tipados no cliente `dlp.py`.
- Serialização pandas → JSON.
- ADRs novos + checklist de documentação de fim de fase.

### 3.2 Fora deste design

| Item | Onde |
|---|---|
| Planejador de IR (`/migration/*`, PDF `fpdf2`) | **Fase 3b** |
| Importar `user_tokens` do Supabase | Fase 5 |
| SPA React | Fase 4 |
| Docker / nginx / HTTPS / flags `Secure` de cookie | Fase 5 |
| Rate limiting de login, troca/recuperação de senha | Futuro (LAB, poucos usuários, contas criadas pelo admin) |
| SSO Google (`django-allauth`) | Futuro (doc mestre §7.1) |
| Corrigir câmbio de ativos de Londres (ver §8) | Pós-Fase 6 |
| Pré-busca assíncrona da carteira após login (doc mestre §3.4) | Futuro |

### 3.3 Definition of Done

1. `.venv/bin/pytest -q` verde (os 125 atuais + os novos), `manage.py check` limpo, sem
   migrations pendentes.
2. `/api/v1/docs` abre e lista todos os endpoints da §5 com schemas.
3. **Roteiro manual no Swagger** (executado com o token DLP real do Bruno, registrado no fecho
   da fase): criar usuário no admin → `GET /auth/csrf` → `POST /auth/login` → `POST /dlp/token`
   → `GET /dlp/token-status` → cada endpoint de carteira com os benchmarks default, um
   sintético livre (ex.: `CDI + 2`) e um ticker sob demanda (ex.: `AAPL BRL`) → `GET
   /portfolio/filters` e um endpoint com `classe` → `POST /portfolio/refresh` → `POST
   /auth/logout` e confirmar que um endpoint protegido devolve 401.
4. Nenhum token (plano ou cifrado) aparece em resposta, log ou mensagem de erro.
5. Checklist de documentação de [`docs/README.md`](../../README.md#checklist-de-atualização-ao-fechar-uma-fase).

---

## 4. Arquitetura

### 4.1 Estrutura de arquivos (novos / alterados)

```
backend/
├── config/
│   ├── api.py                  # NOVO — NinjaAPI(), add_router(...), exception handlers
│   ├── urls.py                 # path("api/v1/", api.urls)
│   └── settings.py             # + "apps.api", CSRF_COOKIE_HTTPONLY=False, MARKET_ASSETS_CSV
├── apps/
│   ├── accounts/
│   │   └── api.py              # NOVO — router /auth
│   ├── api/                    # NOVO app (sem models.py)
│   │   ├── apps.py
│   │   ├── errors.py           # ApiError + códigos (§6)
│   │   ├── schemas.py          # Schemas Ninja (request/response)
│   │   ├── serializers.py      # pandas → JSON (§7)
│   │   ├── analysis.py         # orquestração: token → carteira → dataset (§5.4)
│   │   ├── routers/
│   │   │   ├── dlp.py
│   │   │   ├── portfolio.py
│   │   │   ├── benchmarks.py
│   │   │   └── assets.py
│   │   └── tests/
│   └── market/
│       ├── assets.py           # NOVO — universo de tickers do ativos.csv (§5.6)
│       ├── data/ativos.csv     # NOVO — cópia (D8)
│       ├── series.py           # resolve nome desconhecido via assets + fetch sob demanda
│       ├── services/sources/
│       │   ├── base.py         # + DLPAuthError, DLPUnavailableError
│       │   └── dlp.py          # levanta os erros tipados; invalidate_user_cache()
│       └── management/commands/ingest_market_data.py   # D9
```

A organização interna de `apps/api/` (um arquivo por router, onde ficam helpers) é do builder;
o que importa é a fronteira: **routers finos**, orquestração em `analysis.py`, nenhum cálculo
fora do `engine`.

### 4.2 Direção das dependências

`apps.api` → `engine` → `apps.market` → `apps.core`. `apps.api` também lê `apps.core.models.UserToken`.
`engine` e `apps.market` **não** importam `ninja` nem `apps.api`.

---

## 5. Endpoints

Todos sob `/api/v1/`. Todos exigem sessão (`django_auth` do Ninja) **exceto** `GET /auth/csrf`
e `POST /auth/login`. Toda escrita (POST/DELETE) exige CSRF. O usuário é sempre
`request.user` — **nenhum endpoint recebe id/e-mail de usuário** (isolamento por construção).

### 5.1 Auth — `apps/accounts/api.py`

| Método | Rota | Corpo / resposta |
|---|---|---|
| GET | `/auth/csrf` | Seta o cookie `csrftoken` (`ensure_csrf_cookie`). `204`. |
| POST | `/auth/login` | `{username, password}` → `200 {id, username, email}`; inválido → `401 {code: "invalid_credentials"}`. Usa `django.contrib.auth.authenticate` + `login`. |
| POST | `/auth/logout` | `204`. |
| GET | `/auth/me` | `200 {id, username, email, has_dlp_token}`; sem sessão → `401`. |

O login também exige CSRF (proteção contra login CSRF). O builder deve confirmar como o Ninja
1.4 aplica CSRF a rota **sem** `auth` e ajustar (ex.: `csrf_protect` explícito).

Settings: `CSRF_COOKIE_HTTPONLY = False` (a SPA lê o cookie para mandar `X-CSRFToken`); demais
defaults do Django (`SESSION_COOKIE_HTTPONLY=True`, `SameSite=Lax`). Flags `Secure` ficam para a
Fase 5 (HTTPS via nginx).

### 5.2 Token DLP — `routers/dlp.py`

| Método | Rota | Comportamento |
|---|---|---|
| POST | `/dlp/token` | `{token: str}` (não vazio, trim) → `UserToken.set_token` (cria ou substitui) + invalida o cache DLP do token anterior → `200 {configured: true, updated_at}`. **Não** valida o token contra a DLP (a validação acontece no primeiro uso, com erro tipado). |
| GET | `/dlp/token-status` | `200 {configured: bool, updated_at: datetime|null}`. Não devolve nenhum caractere do token. |
| DELETE | `/dlp/token` | Remove o `UserToken` (e invalida o cache) → `204`; inexistente → `204` (idempotente). |

### 5.3 Corpo comum das análises — `AnalysisRequest`

```jsonc
{
  "start": "2025-10-26",           // opcional; default = hoje − 1 ano (igual ao Streamlit)
  "end":   "2026-10-26",           // opcional; default = hoje
  "ativo":  null,                  // opcional — filtro da carteira (decisão 6)
  "classe": null,                  // opcional — idem
  "benchmarks": ["CDI", "IMA-B", "AAPL BRL", "PETR4.SA"],   // opcional; omitido => catalog.DEFAULT_BENCHMARKS
  "synthetics": [{"base": "CDI", "operador": "+", "valor": 2.0}]   // opcional
}
```

Validação (falha → `422` com `code` específico, §6):

- `start <= end`.
- Cada item de `benchmarks` é: nome do catálogo (`catalog.all_names()`), **ou** ticker Yahoo do
  universo de tickers (§5.6, D10), **ou** ticker Yahoo + sufixo `" BRL"` de um ticker **estrangeiro**.
  Nome do catálogo tem precedência em colisão (ex.: `IVVB11`). Desconhecido → `unknown_benchmark`
  com a lista dos inválidos.
- `synthetics[]`: `base ∈ {"IPCA", "CDI"}`, `operador ∈ {"+", "%"}`;
  `"+"` → `0 < valor <= 50`; `"%"` → `0 < valor <= 500`. Fora disso → `invalid_synthetic`.
- Limite de **15** itens somando `benchmarks` + `synthetics` (protege o Yahoo e o tempo de
  resposta) → `too_many_benchmarks`.
- `benchmarks: []` explícito é válido (só a carteira).

### 5.4 Orquestração — `apps/api/analysis.py`

Uma função usada por todos os endpoints de carteira:

```python
def run_analysis(user, req: AnalysisRequest) -> AnalysisContext:
    token = _require_token(user)                       # ausente -> ApiError dlp_token_missing
    user_series, portfolio_df = fetch_user_portfolio(  # DLP*Error -> ApiError tipado (§6)
        token, ativo=req.ativo, classe=req.classe,
        start_date=req.start, end_date=req.end)
    if user_series is None:                            # ApiError portfolio_empty
        ...
    result = build_dataset(user_series=user_series, benchmarks=req.benchmarks,
                           extra_specs=[SyntheticSpec(...) for s in req.synthetics],
                           start_date=req.start, end_date=req.end,
                           portfolio_df=portfolio_df)
    return AnalysisContext(df_combined=..., portfolio_df=..., selic=..., missing=[...])
```

- Cada endpoint chama `run_analysis` e **um** `compute_*`. A carteira vem do cache DLP (10 min),
  as séries do cache Redis (24h) — recomputar o dataset por endpoint é aceitável (decisão 2).
- `missing`: benchmarks pedidos que voltaram vazios (`get_many` os descarta em silêncio). Vai no
  envelope de resposta (§7) para a SPA avisar, em vez de sumir sem explicação.
- Chamada síncrona: o primeiro acesso aguarda a DLP (até 60s de timeout do cliente atual). A SPA
  mostra loading (doc mestre §3.4).

### 5.5 Carteira — `routers/portfolio.py`

Todos `POST` com `AnalysisRequest` (D1), exceto `filters`/`refresh`.

| Rota | `compute_*` | Resposta (`data` do envelope §7) | Unidade |
|---|---|---|---|
| `GET /portfolio/filters` | — | `{ativos: [str], classes: [str]}` da carteira inteira, ordenados. | — |
| `POST /portfolio/refresh` | — | Invalida o cache DLP do usuário (`dlp.invalidate_user_cache(token)`). `204`. | — |
| `/portfolio/summary` | vários | `{patrimonio_inicial, aportes_periodo, patrimonio, ganho_periodo, twr_total, cagr, volatilidade, sharpe, max_drawdown, xirr, data_inicio, data_fim}` — da coluna `Carteira`. | ver abaixo |
| `/portfolio/twr-evolution` | `compute_twr_evolution` | `{series: [Serie], final_returns: {nome: float}}` | base 100; `final_returns` em pontos % |
| `/portfolio/drawdown` | `compute_drawdown` | `{series: [Serie]}` | a do `calculate_drawdown` |
| `/portfolio/rolling-metrics` | `compute_rolling_metrics` | `{volatility: [Serie], sharpe: [Serie]}`; aceita `window` (int, default 252, `21..756`) no corpo. | fração a.a. / adimensional |
| `/portfolio/risk-return` | `compute_risk_return` | `{rows: [{name, volatilidade, retorno_cagr, sharpe}]}` | fração |
| `/portfolio/shadow` | `compute_shadow` | `{series: [Serie]}` | R$ |
| `/portfolio/yearly-summary` | `compute_yearly_summary` | `{years: [int], rows: [{name, values: {ano: float|null}, total: float}]}` | fração |
| `/portfolio/irr-evolution` | `compute_irr_evolution` | `{series: [Serie]}` (nome `"TIR"`) | % (já ×100 no engine) |

`summary` — valores em R$ **relativos ao período filtrado** (decisão do Bruno, 2026-10-26; não
existe "total investido na vida" nesta fase):
`patrimonio_inicial` = `vlr_mercado` da primeira linha de `portfolio_df`;
`aportes_periodo` = `vlr_investido` da última linha − da primeira (o `vlr_investido` já vem
ajustado pelo gap em `fetch_user_portfolio`, então a diferença é o aporte líquido no período);
`patrimonio` = `vlr_mercado` da última linha;
`ganho_periodo` = `patrimonio − patrimonio_inicial − aportes_periodo`.
Sem alterar o engine (ADR 0012). `twr_total` =
`final_returns["Carteira"]`; `cagr`/`volatilidade`/`sharpe` = linha `Carteira` de
`compute_risk_return`; `max_drawdown` = mínimo da coluna `Carteira` do drawdown; `xirr` = último
ponto de `compute_irr_evolution` (ou `null`). Não inventar métrica nova.

`filters`: deriva os valores distintos de ativo e classe do `/historico` da carteira inteira.
**O builder deve confirmar os nomes das colunas no payload real da DLP** (o código atual só
garante `date`, `vlr_mercado`, `vlr_investido`, `proventos`); se a DLP não trouxer classe por
linha, usar `/resumo` e registrar a escolha.

### 5.6 Benchmarks e tickers

| Método | Rota | Resposta |
|---|---|---|
| GET | `/benchmarks/catalog` | `{groups: [{name, items: [str]}], defaults: [str], synthetic: {bases: ["IPCA","CDI"], operadores: ["+","%"]}}`. Grupos derivados do `SeriesDef` (ex.: por `source`/derivação — agrupamento é do builder). |
| GET | `/assets/search?q=PETR&limit=20` | `[{sigla, nome, classe, mercado: "BR"|"INTL", brl_variant: bool}]`; busca por prefixo de sigla e substring do nome, case-insensitive; `q` com ≥ 2 caracteres; `limit` máx. 50. |

**`apps/market/assets.py`** — universo de tickers:

- Lê `settings.MARKET_ASSETS_CSV` (default `apps/market/data/ativos.csv`; `;`-separado, colunas
  `sigla;market_cod;classe;razao_social`, BOM UTF-8), uma vez por processo (cache em memória).
- **Mesmo universo do Streamlit** (`sidebar.load_assets_from_csv`): só classes
  `ACAO, ETF, ETF RF, FI_INFRA, FI_SET, FIDC, FII, FIP, FIP_IE, BDR, BDR ETF` (BR) e
  `ETF_GB, ETF_US, REIT, STOCK` (INTL). `CRIPTO`, `INDICE`, `OURO` ficam de fora (cripto já
  está no catálogo).
- Ticker Yahoo: `BVMF:` → `<sigla>.SA`; `LON:` **e `LSE:`** → `<sigla>.L`; demais → `<sigla>`.
  (O legado não tratava `LSE:`; corrigido aqui por ser bug de mapeamento, não de cálculo.)
- O identificador é o ticker Yahoo (D10); a busca devolve também `yf_ticker` e a SPA exibe
  `sigla — nome (classe)`.
- `brl_variant = mercado == "INTL"`.
- `resolve(name) -> SeriesDef | None`: `"PETR4.SA"` → `SeriesDef("PETR4.SA", "YF", "yf:PETR4.SA", "level")`;
  `"AAPL BRL"` → `SeriesDef("AAPL BRL", None, None, "level", derivation=ToBRL("AAPL"))`.

**`series.get_series` com nome fora do catálogo:**

1. `catalog.resolve` falha → `assets.resolve`; se também falhar, `KeyError` (a validação da API
   já barrou antes).
2. Para série crua `yf:` vinda de `assets` (sob lock por `series_key`, D12):
   - `repository.last_date(key)` é `None` → **histórico completo** desde `_EPOCH` (D11);
   - `last_date` anterior a `end − 3 dias` → top-up de `last_date + 1` até hoje;
   - grava com `repository.upsert_points(key, "YF", "level", ...)` e segue o fluxo normal
     (Redis → repository). Nesse caminho a chave Redis da série só é gravada **depois** do upsert.
3. `FetchError` do Yahoo: se o banco já tem pontos no período, usa-os (log `warning`); senão o
   ticker entra em `missing` (§5.4). Nunca derruba a análise.
4. Corrida residual (lock expirado): aceitável — `upsert_points` é idempotente.

`ingest_market_data` (D9): além de `catalog.ingestible()`, atualiza as chaves `yf:*` de
`repository.available_keys()` que não pertencem ao catálogo, com a mesma política de falha por
série (marca em `failed_keys`, não aborta o lote).

---

## 6. Erros

Corpo padrão de erro de negócio: `{"code": str, "message": str, "details": object|null}`
(`message` em português, para exibir). Implementado com uma exceção `ApiError(status, code,
message, details)` + `api.exception_handler`.

| Situação | Status | `code` |
|---|---|---|
| Sem sessão | 401 | (padrão do Ninja) |
| Login inválido | 401 | `invalid_credentials` |
| Usuário sem token DLP | 409 | `dlp_token_missing` |
| DLP rejeitou o token (401/403 da DLP) | **422** (D3) | `dlp_token_rejected` |
| DLP fora do ar (timeout, conexão, 5xx, JSON inválido) | 502 | `dlp_unavailable` |
| Carteira sem histórico no período/filtro | 404 | `portfolio_empty` |
| Validação do `AnalysisRequest` | 422 | `invalid_period` / `unknown_benchmark` / `invalid_synthetic` / `too_many_benchmarks` |

**Mudança em `dlp.py`/`base.py`:** hoje `buscar_historico`/`buscar_resumo_carteira` embrulham
tudo num `DLPError` genérico. Passam a levantar `DLPAuthError(DLPError)` para 401/403 e
`DLPUnavailableError(DLPError)` para timeout/conexão/5xx/parse. Outros 4xx → `DLPError` base
(mapeado para `dlp_unavailable` com log de erro). **O builder deve verificar com um token
inválido (teste `integration`) qual status a DLP realmente devolve**; se ela responder `200` com
corpo de erro, detectar pelo corpo e registrar.

Mensagens de erro e logs **nunca** incluem o token; o identificador de log é o
`_token_hash` (12 hex) já usado nas chaves de cache.

`dlp.invalidate_user_cache(token)`: apaga `market:dlp:*:<token_hash>*` via
`cache.delete_pattern` do `django-redis` (as chaves `hist` e `resumo` já contêm o hash).

---

## 7. Serialização — `apps/api/serializers.py`

- Envelope dos endpoints de análise: `{"data": <payload da §5.5>, "meta": {"start", "end",
  "missing": [str]}}` — `start`/`end` = período efetivo do `df_combined`.
- `Serie = {"name": str, "data": [["YYYY-MM-DD", float|null], ...]}` (D4), uma por coluna, na
  ordem do DataFrame (o `compute_twr_evolution` já ordena por retorno final).
- `NaN`/`±inf` → `null`. Datas sem hora (`date().isoformat()`).
- Funções puras e testadas isoladamente: `series_to_json(pd.Series)`,
  `frame_to_series_list(pd.DataFrame)`, `frame_to_rows(pd.DataFrame, ...)`. Os schemas Ninja
  descrevem o formato no Swagger.

---

## 8. Riscos e pendências

| Risco | Tratamento |
|---|---|
| Formato real do payload DLP (colunas de ativo/classe; status de token inválido) não está documentado no repo. | Teste `integration` com token real/inválido via env (`DLP_TEST_TOKEN`), executado no roteiro manual; ajustar e registrar. |
| Ticker sob demanda deixa a primeira requisição lenta (Yahoo, histórico completo — D11). | Limite de 15 itens; lock single-flight (D12); depois serve do banco e só faz top-up. |
| Ativos de Londres (`.L`) são cotados em GBP/GBp, mas o legado converte todo estrangeiro por USD→BRL. | **Mantido igual ao legado** (paridade); registrar como débito pós-Fase 6. |
| `compute_risk_return` usa taxa livre de risco fixa de 10% (não a SELIC). | Mantido (ADR 0012); débito pós-Fase 6. |
| `make_key` não sanitiza nomes (`"S&P 500"`, `"AAPL BRL"`) — `CacheKeyWarning`. | Tolerado com Redis (já registrado no doc mestre §2.3). |
| Filtro `ativo`/`classe` não tem referência no Streamlit. | Fora da paridade da Fase 6; teste de contrato com DLP simulada garante que o filtro chega à DLP. |

---

## 9. Testes

- **Contrato dos endpoints** (`apps/api/tests/`, `apps/accounts/tests/`) com o test client do
  Django/Ninja: status, schema e códigos de erro de cada rota; 401 sem sessão em todas as
  protegidas; CSRF exigido em escrita; isolamento (usuário A nunca vê token/carteira de B).
- **DLP simulada**: monkeypatch de `requests.get` no `dlp.py` (padrão dos testes da Fase 2),
  cobrindo 200, 401/403, timeout, 5xx e histórico vazio.
- **Mercado**: séries vindas de fixtures/`MarketSeries` populado no teste (padrão da Fase 2);
  Yahoo mockado para tickers sob demanda (primeiro uso faz fetch; segundo não).
- **Serializers e `assets.py`**: testes unitários puros.
- **Paridade de formato**: um teste que monta o dataset de fixture e confere que os valores do
  JSON de `twr-evolution`/`yearly-summary` batem com o `compute_*` direto (a API não altera números).
- **`integration`** (desmarcado por padrão): DLP real com `DLP_TEST_TOKEN`; Yahoo real para um ticker.
- TDD conforme `backend/CLAUDE.md` (ver falhar → implementar → ver passar).

---

## 10. Documentação e ADRs

- **ADR 0013** — contrato da API: `POST` + `AnalysisRequest`, envelope `data/meta`, formato de
  série, tabela de códigos de erro (D1–D5).
- **ADR 0014** — tickers sob demanda: universo do `ativos.csv`, fetch-on-read + persistência,
  atualização pela ingestão, cópia empacotada do CSV, identificador = ticker Yahoo, histórico
  completo no primeiro fetch, single-flight (D7–D12).
- Checklist de fim de fase de `docs/README.md` (doc mestre §2.0/§2.2/§4/§9 — inclusive
  registrar a Fase 3b do IR no roteiro —, `backend/README.md`, `backend/CLAUDE.md` com as
  convenções novas de `apps/api`).

---

## 11. Deixado para o builder

- Ordem das tasks e divisão em commits (um por task, conforme `backend/CLAUDE.md`).
- Organização interna de `apps/api/` e nomes dos schemas.
- Agrupamento exibido em `/benchmarks/catalog`.
- Como garantir CSRF no `POST /auth/login` no Ninja 1.4.
- Algoritmo exato da busca de tickers (respeitando a §5.6).

## 12. Próximo passo

Aprovado este design → `superpowers:writing-plans` gera o plano de implementação da Fase 3 em
`docs/superpowers/plans/`.
