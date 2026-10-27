# Correções da Revisão das Fases 3/3b/3c e 4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrigir os defeitos Críticos/Importantes (e alguns Menores baratos) apontados pela revisão de 2026-10-27 do backend (`6e83734..aa45d1a`) e da SPA (`aa45d1a..f6a6533`), sem mexer em `engine/`.

**Architecture:** Backend: endurecer `apps/market/services/sources/dlp.py` (cache curto de falha + captura ampla de erros de rede) e `apps/migration_ir/planner.py` (limites de teto/meses, folga em R$, valores finitos). Frontend: um helper único de erro (`src/api/errors.js`) usado por interceptor, retry, gráficos, PDF e planejador; preset "Tudo" com data inicial explícita; gating das queries pelo status do token DLP; estados de erro nos gráficos; ajustes do planejador (padrão `ignore`, caixa, avisos legíveis).

**Tech Stack:** Django 5 + Django Ninja + pytest-django (backend); React 18 + Vite + TanStack Query v5 + Axios + ECharts + Vitest/Testing Library (frontend).

**Spec:** não há spec novo — os requisitos são os achados abaixo, julgados contra os specs/ADRs existentes: `docs/superpowers/specs/2026-10-26-fase-3-api-carteira-design.md`, `...-fase-3b-planejador-ir-design.md`, `...-fase-4-frontend-react-design.md`, `docs/adr/0013-contrato-da-api-v1.md`, `docs/adr/0015-planejador-de-ir.md`.

### Achados que este plano corrige

| Achado | Severidade | Task |
|---|---|---|
| Preset "Tudo" (padrão) manda `start: null` → backend usa 1 ano | Crítico | 4 |
| Planejador marca `sell` por padrão (ADR 0015 §2 diz `ignore`) | Importante | 9 |
| Erro do PDF chega como `Blob`: mensagem perdida e modal DLP não abre | Importante | 3 |
| `retry: 1` em 4xx + 9 requests com 409 sem token + modal reabrindo | Importante | 3, 6, 7 |
| DLP fora do ar: single-flight serializa timeouts de 60 s | Importante | 1 |
| Planejador sem limite de meses (teto minúsculo → plano gigante) | Importante | 2 |
| Gráficos e carteira de IR sem estado de erro | Importante | 8 |
| Unidade de % adivinhada por magnitude (CAGR 214% → 2,14%) | Importante | 5 |
| `meta.missing` e período efetivo nunca exibidos | Importante | 4 |
| Avisos do plano como códigos crus; `ir_not_estimated` oculto | Importante | 10 |
| Sem campo de caixa inicial / "apenas aportes" | Importante | 9 |
| `_EPS` aplicado em unidades de ação | Menor | 2 |
| `NaN`/`Infinity` aceitos em valores monetários do plano | Menor | 2 |
| `_get` não captura `RequestException`; payload não-dict → 500 | Menor | 1 |
| Datas dos presets em UTC (`toISOString`) | Menor | 4 |
| Tolerância da soma de pesos: UI ±0,01 vs backend ±1e-6 | Menor | 9 |
| Plano exibido/PDF divergem quando as entradas mudam | Menor | 10 |

**Fora do escopo (backlog, registrar no §2.3 na Task 11):** lock do `single_flight` sem dono; geração do cache despejada pelo LRU; corrida no `save_token` (`IntegrityError` em POST concorrente); 422 de schema do Ninja fora do envelope `ApiError`; cálculo dentro do router `summary`; `start` default relativo a hoje quando só `end` vem; filtro DLP ignorado se a coluna faltar (depende do teste `integration` pendente); layout do PDF em quebra de página; rate limit de login; timeout DLP (60 s) vs gunicorn (30 s) — decidir na Fase 5; `catalog.defaults` não usado; busca de ticker sem debounce; `Content-Disposition`/`revokeObjectURL`; `location.state.from` no login; filtros sobrevivendo ao logout; token não salvo persistindo no modal; acessibilidade do `Modal`; mensagem de "primeira carga DLP"; D6 (`period` na URL); bundle de 1,4 MB; código morto (`data?.data?.x || data?.x`, branch shorthand do `SummaryCards`).

## Global Constraints

- **Não tocar em `backend/engine/`** (cópia literal, ADR 0012) — nenhum número do engine pode mudar.
- Contrato da API v1 (ADR 0013) não muda de forma; só o planejador ganha validações novas (códigos existentes `invalid_parameters`).
- Backend: `cd backend && .venv/bin/pytest -q` precisa continuar verde (hoje 331 passed, 5 deselected).
- Frontend: `cd frontend && npx vitest run` verde (hoje 37) e `npm run build` sem erro.
- Mensagens ao usuário em pt-BR; commits no padrão do repo (gitmoji + `tipo(escopo): descrição` em pt-BR), terminando com `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Frontend sem dependências novas.

## Review Focus

1. **Usuário sem token DLP abre o dashboard** → espera um único aviso/CTA "configure o token", não 9 requests 409, retries e modal reabrindo após "Cancelar". (Task 7 testa: com `configured: false`, nenhuma chamada a `portfolioApi.*`.)
2. **DLP fora do ar com várias abas/gráficos** → espera erro rápido e legível, não minutos de espera. (Task 1 testa: segunda chamada após falha não toca `requests.get`; Task 8 testa mensagem de 502 no gráfico.)
3. **Preset "Tudo" em carteira com mais de 1 ano** → espera o histórico completo e ver as datas efetivas. (Task 4 testa `start === '1900-01-01'` e o aviso com `meta.start`/`meta.end`.)
4. **Usuário abre o planejador e clica "Calcular" sem marcar nada** → espera nenhuma venda (ou o 422 `nothing_to_plan` com mensagem legível), nunca vender a carteira toda. (Task 9 testa payload com todas as ações `ignore`.)
5. **Usuário às 22h em São Paulo escolhe "1A"** → espera `end` = data local de hoje, não amanhã. (Task 4 testa com `baseDate` 2026-10-27T22:30 local.)

---

## Task 0: Ignorar localmente os hooks do plugin `impeccable` (sem commit)

`.gemini/settings.json` e `.github/hooks/impeccable.json` foram gravados pelo instalador global do `impeccable` e apontam para caminhos da máquina (`/home/ubuntu/.gemini/skills/...`, `$HOME/.github/skills/...`). Não pertencem ao repositório (que é público) nem servem a outro clone.

- [x] **Step 1:** Acrescentar ao `.git/info/exclude` (não ao `.gitignore`, para não versionar uma regra pessoal):

```
.gemini/settings.json
.github/hooks/impeccable.json
```

- [x] **Step 2:** Verificar: `git status --short` → não lista mais `.gemini/settings.json` nem `.github/hooks/`.

---

## Task 1: DLP — cache curto de falha e captura ampla de erros

**Files:**
- Modify: `backend/apps/market/services/sources/dlp.py` (`_get` linhas ~55-74, `buscar_historico` ~96-117, `buscar_resumo_carteira` ~120-137)
- Test: `backend/apps/market/tests/test_sources_dlp.py`

**Interfaces:**
- Produces: constante `ERR_TTL = 30` (segundos) em `dlp.py`; chave de falha `<key>:err` (onde `<key>` é a chave já existente `dlp:hist:<hash>:<gen>` / `dlp:resumo:<hash>:<gen>`). Comportamento público inalterado: continuam levantando `DLPUnavailableError` / `DLPAuthError` / `DLPError`.

- [x] **Step 1: Testes que falham** (usar `_Resp`, `monkeypatch` e o fixture `_locmem_cache` já existentes no arquivo)

```python
def test_recent_unavailable_is_cached_and_skips_dlp(monkeypatch):
    calls = []
    def fake_get(*a, **k):
        calls.append(1)
        raise requests.Timeout()
    monkeypatch.setattr(dlp.requests, "get", fake_get)
    with pytest.raises(DLPUnavailableError):
        dlp.buscar_historico("DLP-tok")
    with pytest.raises(DLPUnavailableError):
        dlp.buscar_historico("DLP-tok", ativo="PETR4")
    assert len(calls) == 1

def test_refresh_clears_cached_failure(monkeypatch):
    # falha -> invalidate_user_cache -> próxima chamada vai à DLP de novo e tem sucesso
    ...
    assert len(calls) == 2 and isinstance(df, pd.DataFrame)

def test_auth_error_is_not_cached_as_failure(monkeypatch):
    # 401 duas vezes -> requests.get chamado 2x (token trocado invalida geração; não mascarar)
    ...
    assert len(calls) == 2

def test_resumo_failure_is_cached(monkeypatch):
    # mesmo que o primeiro teste, para buscar_resumo_carteira
    ...

@pytest.mark.parametrize("exc", [requests.exceptions.ChunkedEncodingError, requests.exceptions.TooManyRedirects])
def test_other_request_exceptions_raise_unavailable(monkeypatch, exc):
    ...  # pytest.raises(DLPUnavailableError)

@pytest.mark.parametrize("payload", [[], None, "x"])
def test_non_object_json_raises_unavailable(monkeypatch, payload):
    ...  # _Resp(payload) -> pytest.raises(DLPUnavailableError) em buscar_historico
```

- [x] **Step 2:** `cd backend && .venv/bin/pytest apps/market/tests/test_sources_dlp.py -q` → os 6 novos FAIL.

- [x] **Step 3: Implementar em `dlp.py`**
  - `_get`: trocar `except (requests.Timeout, requests.ConnectionError)` por `except requests.RequestException`; após `resp.json()`, se `not isinstance(payload, dict)` → `raise DLPUnavailableError(f"DLP {path}: resposta não é um objeto JSON")`. A mensagem nunca inclui o token.
  - Helpers privados `_failed_recently(key) -> bool` (`cache.get(key + ":err")`, com `try/except` + warning como `_generation`) e `_mark_failed(key) -> None` (`cache.set(key + ":err", 1, ERR_TTL)`).
  - Em `buscar_historico` e `buscar_resumo_carteira`: checar `_failed_recently(key)` **antes** do `single_flight` e de novo **dentro** dele (depois do re-check do cache); se verdadeiro, `raise DLPUnavailableError("DLP indisponível (falha recente)")`. Envolver só o `_get(...)` em `try/except DLPUnavailableError: _mark_failed(key); raise`. `DLPAuthError`/`DLPError` não marcam.
  - Como `<key>` já contém a geração, `POST /portfolio/refresh` (que incrementa a geração) libera nova tentativa sem código extra.

- [x] **Step 4:** `.venv/bin/pytest apps/market/tests/test_sources_dlp.py apps/api -q` → PASS.

- [x] **Step 5: Commit** — `🐛 fix(market): DLP indisponível fica em cache por 30s e erros de rede não viram 500`

---

## Task 2: Planejador — limites de teto e meses, folga em R$, valores finitos

**Files:**
- Modify: `backend/apps/migration_ir/planner.py` (constantes ~20-23, `_validate_params` ~86-90, `_schedule_stocks` ~250-294)
- Test: `backend/apps/migration_ir/tests/test_planner_validate.py`, `backend/apps/migration_ir/tests/test_planner_build.py`

**Interfaces:**
- Produces: `MIN_CAP = 100.0`, `MAX_MONTHS = 240` em `planner.py`. Erros novos usam `PlanError("invalid_parameters", ...)` (já mapeado para 422 por `plan_errors()`).

- [x] **Step 1: Testes que falham**

```python
# test_planner_validate.py (usar o helper _v existente)
@pytest.mark.parametrize("cap", [99.99, float("nan"), float("inf")])
def test_cap_below_min_or_non_finite_is_invalid(cap):
    with pytest.raises(PlanError) as e:
        _v(monthly_cap=cap)
    assert e.value.code == "invalid_parameters"

@pytest.mark.parametrize("field", ["initial_cash", "already_sold"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_money_is_invalid(field, value):
    with pytest.raises(PlanError) as e:
        _v(**{field: value})
    assert e.value.code == "invalid_parameters"

def test_cap_100_is_valid():
    assert _v(monthly_cap=100).monthly_cap == 100

# test_planner_build.py
def test_plan_longer_than_max_months_is_rejected():
    # ação ACAO com qtd=300, price=150 e monthly_cap=100 -> 1 unidade/mês -> 300 meses
    with pytest.raises(PlanError) as e:
        build_plan(inp, prices)
    assert e.value.code == "invalid_parameters" and "240" in e.value.message

def test_unit_slightly_over_remaining_is_not_sold():
    # price = 20000/3 + 1e-4 (não múltiplo de centavo), cap=20000, qtd=3
    # deve vender 2 no mês 1 e 1 no mês 2; nenhum mês com "exceeds_exemption"
    data, _ = build_plan(inp, prices)
    assert [sum(s["qtd"] for s in m["sales"]) for m in data["months"]] == [2, 1]
    assert all("exceeds_exemption" not in m["warnings"] for m in data["months"])
```

(Montar `inp` como os testes existentes de `test_planner_build.py` fazem; o executor segue o padrão local.)

- [x] **Step 2:** `.venv/bin/pytest apps/migration_ir -q` → novos FAIL.

- [x] **Step 3: Implementar**
  - `_validate_params`: rejeitar se qualquer um de `monthly_cap`, `initial_cash`, `already_sold` não for `math.isfinite`; trocar a faixa para `MIN_CAP <= monthly_cap <= MAX_CAP` com mensagem `"O teto mensal deve estar entre R$ 100 e R$ 20.000."`.
  - `_schedule_stocks`: linha do `n = ...` passa a `math.floor((remaining + _EPS) / price)`; em `close()`, depois do `append`, se `len(months) > MAX_MONTHS` → `raise PlanError("invalid_parameters", f"O plano passaria de {MAX_MONTHS} meses; aumente o teto mensal ou venda menos papéis.")`. Isso limita CPU antes de montar compras.

- [x] **Step 4:** `.venv/bin/pytest -q` (suíte inteira) → PASS.

- [x] **Step 5: Commit** — `🐛 fix(migration_ir): limita teto e meses do plano, folga em reais e rejeita NaN`

---

## Task 3: Frontend — helper de erro e parse de erro em `Blob`

**Files:**
- Create: `frontend/src/api/errors.js`
- Modify: `frontend/src/api/client.js` (interceptor de resposta)
- Test: `frontend/src/test/api-errors.test.js` (novo)

**Interfaces:**
- Produces:
  - `getApiError(err) -> { status: number|null, code: string|null, message: string|null }` — lê `err.response.data` (`{code, message}` do `ApiError`; `{detail: [...]}` do Ninja vira `code: 'invalid_request'` e `message` = primeiro `detail[].msg`); erro sem `response` → `{status: null, code: 'network_error', message: 'Falha de conexão com o servidor.'}`.
  - `async parseBlobErrorData(data) -> any` — se `data instanceof Blob` e `data.type` contém `json`, devolve `JSON.parse(await data.text())`; caso contrário devolve `data`.
  - `shouldRetry(failureCount, err) -> boolean` — `true` só se `failureCount < 1` **e** (`!err.response` ou `status >= 500`).

- [x] **Step 1: Testes que falham**

```js
it('getApiError lê code/message do envelope ApiError', ...)        // {status:422, code:'invalid_period', message:'...'}
it('getApiError traduz detail do Ninja para invalid_request', ...)
it('getApiError sem response vira network_error', ...)
it('parseBlobErrorData converte Blob JSON em objeto', async () => {
  const blob = new Blob([JSON.stringify({ code: 'nothing_to_plan', message: 'Nada a planejar' })], { type: 'application/json' });
  expect(await parseBlobErrorData(blob)).toEqual({ code: 'nothing_to_plan', message: 'Nada a planejar' });
});
it('shouldRetry: não repete 4xx, repete uma vez 5xx/rede', () => {
  expect(shouldRetry(0, { response: { status: 409 } })).toBe(false);
  expect(shouldRetry(0, { response: { status: 502 } })).toBe(true);
  expect(shouldRetry(1, { response: { status: 502 } })).toBe(false);
  expect(shouldRetry(0, {})).toBe(true);
});
it('interceptor abre modal DLP quando erro 409 vem como Blob', async () => {
  // usar o handler de erro registrado em api.interceptors.response (handlers[0].rejected)
  // com error.response = { status: 409, data: Blob({code:'dlp_token_missing'}) }
  // espera evento 'open-dlp-modal' com reason 'missing' e error.response.data já objeto
});
```

- [x] **Step 2:** `cd frontend && npx vitest run src/test/api-errors.test.js` → FAIL.

- [x] **Step 3: Implementar** `errors.js`; no `client.js` tornar o handler de erro `async`, fazer `error.response.data = await parseBlobErrorData(error.response.data)` antes de ler `code`/`message` (o resto do handler fica igual).

- [x] **Step 4:** `npx vitest run` → PASS (inclusive `api-auth.test.jsx`).

- [x] **Step 5: Commit** — `✨ feat(frontend): helper getApiError, retry só em 5xx e erros em Blob legíveis`

---

## Task 4: Preset "Tudo", datas locais e aviso de período/benchmarks ausentes

**Files:**
- Modify: `frontend/src/context/PortfolioFilterContext.jsx` (`computePresetDates`, `resetFilters`)
- Create: `frontend/src/components/portfolio/PeriodNotice.jsx`
- Modify: `frontend/src/pages/PortfolioPage.jsx` (renderizar `PeriodNotice` logo abaixo de `<FiltersBar />`)
- Test: `frontend/src/test/filters-bar.test.jsx` (corrigir o teste "null ao selecionar Tudo"), `frontend/src/test/risk-charts-page.test.jsx` ou novo `period-notice.test.jsx`

**Interfaces:**
- Produces: `export const ALL_TIME_START = '1900-01-01'`; `export function toLocalISODate(date: Date) -> 'YYYY-MM-DD'` (com `getFullYear/getMonth/getDate`) em `PortfolioFilterContext.jsx`; `computePresetDates('Tudo') -> { start: ALL_TIME_START, end: null }`.
- `PeriodNotice({ meta })` — `meta` é o `meta` do envelope de `/summary` (`{start, end, missing}`, ADR 0013). Renderiza `Período analisado: dd/mm/aaaa a dd/mm/aaaa` e, se `missing.length > 0`, `Sem dados no período: X, Y`. Não renderiza nada sem `meta`.

Por que `1900-01-01`: o backend recorta pelo início real do `df_combined` (`analysis.py`, `effective_start`) e devolve a data efetiva em `meta.start`; mandar uma data antiga não muda o contrato (escolha recomendada na revisão, em vez de redefinir `start: null` na ADR 0013).

- [x] **Step 1: Testes que falham**

```js
it('Tudo envia start 1900-01-01 e end null', ...)            // substitui o teste que afirmava null
it('estado inicial (Tudo) já usa ALL_TIME_START', ...)
it('1A às 22:30 locais usa a data local de hoje', () => {
  const base = new Date(2026, 9, 27, 22, 30);
  expect(computePresetDates('1A', base)).toEqual({ start: '2025-10-27', end: '2026-10-27' });
});
it('PeriodNotice mostra período efetivo e benchmarks sem dados', () => {
  render(<PeriodNotice meta={{ start: '2019-03-04', end: '2026-10-26', missing: ['IFIX', 'IPCA + 6%'] }} />);
  expect(screen.getByText(/04\/03\/2019 a 26\/10\/2026/)).toBeInTheDocument();
  expect(screen.getByText(/Sem dados no período: IFIX, IPCA \+ 6%/)).toBeInTheDocument();
});
```

- [x] **Step 2:** `npx vitest run src/test/filters-bar.test.jsx` → FAIL.

- [x] **Step 3: Implementar.** `computePresetDates` usa `toLocalISODate` nos presets; `resetFilters` usa `computePresetDates('Tudo')`. Formatar `dd/mm/aaaa` a partir da string `YYYY-MM-DD` por split (sem `new Date(str)`, que interpreta como UTC). Na `PortfolioPage`, `<PeriodNotice meta={summary.data?.meta} />`.

- [x] **Step 4:** `npx vitest run` → PASS.

- [x] **Step 5: Commit** — `🐛 fix(frontend): preset Tudo analisa o histórico completo e datas locais nos presets`

---

## Task 5: Percentuais sem adivinhar unidade

**Files:**
- Modify: `frontend/src/components/portfolio/charts/RiskReturnChart.jsx` (`toPct`), `RollingSharpeChart.jsx` (`toVolPercent`), `DrawdownChart.jsx` (helper equivalente)
- Test: `frontend/src/test/risk-charts-page.test.jsx`

**Interfaces:** contrato de backend: `volatilidade`, `retorno_cagr`, volatilidade móvel e drawdown são **frações** (ADR 0013). Sharpe não é percentual e não muda.

- [x] **Step 1: Teste que falha** — exportar o helper de `RiskReturnChart.jsx` como `toPct` e testar `toPct(2.14) === 214`, `toPct(0.15) === 15`, `toPct(null) === 0` (manter o comportamento atual para nulo). Idem `toVolPercent(2.5) === 250`.

- [x] **Step 2:** `npx vitest run src/test/risk-charts-page.test.jsx` → FAIL.

- [x] **Step 3:** Multiplicar por 100 sempre nos três helpers (remover o `Math.abs(num) <= 2 ? ... : num`).

- [x] **Step 4:** `npx vitest run` → PASS.

- [x] **Step 5: Commit** — `🐛 fix(frontend): percentuais de risco sempre convertidos de fração`

---

## Task 6: Retry só em 5xx/rede

**Files:**
- Modify: `frontend/src/App.jsx:19`
- Test: `frontend/src/test/app-routing.test.jsx` (ou o novo `api-errors.test.js`)

**Interfaces:** Consumes `shouldRetry` (Task 3).

- [x] **Step 1: Teste que falha** — renderizar um componente com `useQuery` sob o `QueryClient` criado pelo `App` (extrair `export function createQueryClient()` de `App.jsx` para isso) cujo `queryFn` rejeita com `{ response: { status: 409 } }`; esperar que o `queryFn` tenha sido chamado exatamente 1 vez.

- [x] **Step 2:** FAIL (hoje chama 2x).

- [x] **Step 3:** `createQueryClient()` com `retry: shouldRetry`; `App` usa `useMemo(createQueryClient, [])`.

- [x] **Step 4:** `npx vitest run` → PASS.

- [x] **Step 5: Commit** — `🐛 fix(frontend): não repetir requisições com erro 4xx`

---

## Task 7: Queries da carteira e do planejador só com token configurado

**Files:**
- Create: `frontend/src/hooks/useDlpTokenStatus.js`
- Modify: `frontend/src/components/layout/Header.jsx` (trocar o `useQuery` local pelo hook), `frontend/src/hooks/usePortfolioQueries.js`, `frontend/src/hooks/useMigrationPlan.js` (`walletQuery`), `frontend/src/pages/PortfolioPage.jsx`, `frontend/src/pages/MigrationPage.jsx`
- Test: `frontend/src/test/header-dlp.test.jsx`, `frontend/src/test/risk-charts-page.test.jsx`, `frontend/src/test/migration-ir.test.jsx`

**Interfaces:**
- Produces: `useDlpTokenStatus() -> UseQueryResult<{configured: boolean, ...}>` com `queryKey: ['dlp', 'token-status']`, `queryFn: dlpApi.getTokenStatus`, `staleTime: 60_000` (mesmos valores de hoje no `Header`).
- `usePortfolioQueries()` e `useMigrationPlan()` passam `enabled: tokenStatus?.configured === true` em todas as queries e expõem `tokenConfigured: boolean | undefined` (undefined enquanto o status carrega).

- [x] **Step 1: Testes que falham**
  - `PortfolioPage` com `dlpApi.getTokenStatus` → `{configured:false}`: nenhum `portfolioApi.fetch*` chamado; aparece `EmptyState` com título `Configure seu token DLP` e botão `Configurar token` que dispara `window` event `open-dlp-modal` (`reason: 'missing'`).
  - `MigrationPage` com `configured:false`: `migrationApi.getWallet` não chamado; mesmo `EmptyState`.
  - Com `configured:true`: comportamento atual (testes existentes continuam passando — ajustar seus mocks para `getTokenStatus` → `{configured:true}`).

- [x] **Step 2:** FAIL.

- [x] **Step 3: Implementar** o hook, o `enabled` e o `EmptyState` (enquanto `tokenConfigured === undefined`, mostrar os skeletons atuais).

- [x] **Step 4:** `npx vitest run` → PASS.

- [x] **Step 5: Commit** — `🐛 fix(frontend): carteira e planejador só consultam a API com token DLP configurado`

---

## Task 8: Estados de erro nos gráficos, tabela anual e carteira do planejador

**Files:**
- Create: `frontend/src/components/common/ErrorState.jsx`
- Modify: `frontend/src/components/portfolio/charts/BaseChart.jsx`, os 7 `charts/*Chart.jsx`, `frontend/src/components/portfolio/tables/YearlyTable.jsx`, `frontend/src/components/portfolio/SummaryCards.jsx`, `frontend/src/pages/PortfolioPage.jsx`, `frontend/src/pages/MigrationPage.jsx`
- Test: `frontend/src/test/risk-charts-page.test.jsx`, `frontend/src/test/migration-ir.test.jsx`

**Interfaces:**
- Consumes `getApiError` (Task 3).
- Produces: `ErrorState({ error, onRetry })` — mostra `getApiError(error).message` (fallback `'Não foi possível carregar estes dados.'`) e, se `onRetry`, botão `Tentar novamente`; `role="alert"`.
- `BaseChart` ganha props `error` e `onRetry`; com `error` renderiza `ErrorState` no lugar do gráfico (mesma altura). Cada gráfico/tabela repassa `error`/`onRetry`. Na `PortfolioPage`: `error={q.error} onRetry={q.refetch}` para cada query. Erro 404 `portfolio_empty` continua no `EmptyState` existente.
- `MigrationPage`: `walletQuery.isError` que não seja `portfolio_empty` → `ErrorState` com `onRetry={walletQuery.refetch}`; a mensagem de erro do plano passa a usar `getApiError(planMutation.error).message`.

- [x] **Step 1: Testes que falham**
  - `PortfolioPage` com `fetchDrawdown` rejeitando `{response:{status:502,data:{code:'dlp_unavailable',message:'DLP indisponível'}}}` (aba Risco): aparece `DLP indisponível` e clicar `Tentar novamente` chama `fetchDrawdown` de novo.
  - `MigrationPage` com `getWallet` rejeitando 502: aparece a mensagem e o botão; `HoldingsTable` não é renderizada.

- [x] **Step 2:** FAIL. **Step 3:** implementar. **Step 4:** `npx vitest run` → PASS.

- [x] **Step 5: Commit** — `✨ feat(frontend): estados de erro com nova tentativa nos gráficos e no planejador`

---

## Task 9: Planejador — padrão `ignore`, caixa inicial, "apenas aportes", tolerância de pesos

**Files:**
- Modify: `frontend/src/hooks/useMigrationPlan.js` (`planPayload`), `frontend/src/components/migration/HoldingsTable.jsx:63`, `frontend/src/components/migration/TargetAllocationForm.jsx` (`isTargetListValid`, novos campos), `frontend/src/pages/MigrationPage.jsx`
- Test: `frontend/src/test/migration-ir.test.jsx`

**Interfaces:**
- Produces: `export function defaultActionFor(holding, isTargetHolding) -> 'ignore'` em `useMigrationPlan.js` (sempre `'ignore'`, ADR 0015 §2), usada pelo hook e pela `HoldingsTable` — uma fonte só.
- `TargetAllocationForm` ganha props `initialCash`, `onChangeInitialCash`, `contributionsOnly`, `onChangeContributionsOnly` (rótulos `Caixa disponível para aportes (R$)` e `Apenas aportes (não vender nada)`); `MigrationPage` liga ao hook (os setters já existem).
- `isTargetListValid`: tolerância `Math.abs(total - 100) > 1e-6` (igual a `_WEIGHT_TOL` do backend).

- [x] **Step 1: Testes que falham**
  - Sem nenhuma escolha do usuário, `planPayload.holdings` tem todas as ações `ignore` (inclusive papéis com cotação).
  - `HoldingsTable` mostra `Ignorar` selecionado por padrão num papel com cotação.
  - Digitar `5000` em `Caixa disponível…` → `planPayload.initial_cash === 5000`; marcar `Apenas aportes` → `contributions_only === true`.
  - `isTargetListValid([{ticker:'A',weight:33.333},{ticker:'B',weight:33.333},{ticker:'C',weight:33.334}])` → `true`; com `33.333 × 3` → `false`.
  - Ajustar o teste existente "alinha planPayload com validate_plan_input" para o novo padrão.

- [x] **Step 2:** FAIL. **Step 3:** implementar. Drafts antigos em `sessionStorage` com `action: 'sell'` explícito continuam respeitados (só o padrão muda). **Step 4:** `npx vitest run` → PASS.

- [x] **Step 5: Commit** — `🐛 fix(frontend): planejador não vende nada por padrão e aceita caixa inicial`

---

## Task 10: Avisos do plano legíveis e plano desatualizado

**Files:**
- Create: `frontend/src/components/migration/planWarnings.js`
- Modify: `frontend/src/components/migration/MonthlyPurchasesTable.jsx`, `frontend/src/hooks/useMigrationPlan.js`
- Test: `frontend/src/test/migration-ir.test.jsx`

**Interfaces:**
- Produces: `formatPlanWarning(code: string) -> string`. Códigos do backend (`planner.py`, `prices.py`, `wallet.py`) e textos:

| Código | Texto |
|---|---|
| `exceeds_exemption` | Vendas de ações do mês passam de R$ 20.000: o lucro do mês é tributado. |
| `already_over_exemption` | As vendas já feitas neste mês passam de R$ 20.000. |
| `exceeds_cap` | Uma única unidade ultrapassa o teto mensal informado. |
| `cost_basis:<ATIVO>` | <ATIVO>: saldo calculado pelo custo (sem cotação de mercado). |
| `fx_approx:<ATIVO>` | <ATIVO>: cotação convertida pela PTAX (aproximada). |
| `not_estimated:<ATIVO>` | <ATIVO>: IR não estimado para esta classe. |
| `mixed_class:<ATIVO>` | <ATIVO>: aparece com mais de uma classe na carteira. |
| desconhecido | o próprio código |

- `MonthlyPurchasesTable` aplica `formatPlanWarning` em `meta.warnings` e `m.warnings`, e, se `totals.ir_not_estimated.length > 0`, mostra `IR não estimado para: X, Y` no subtítulo do card.
- `useMigrationPlan`: quando `planPayload` muda (comparar `JSON.stringify`) depois de um plano calculado, chamar `planMutation.reset()` — o plano exibido e o PDF sempre correspondem às entradas atuais.

- [x] **Step 1: Testes que falham** — `formatPlanWarning('fx_approx:VWRA11')` contém `VWRA11` e `PTAX`; `MonthlyPurchasesTable` com `m.warnings: ['exceeds_exemption']` não mostra o código cru; com `totals.ir_not_estimated: ['FI_X']` mostra `IR não estimado para: FI_X`; após calcular um plano, mudar o peso de um alvo remove o `MonthlyPurchasesTable`.

- [x] **Step 2:** FAIL. **Step 3:** implementar. **Step 4:** `npx vitest run` → PASS.

- [x] **Step 5: Commit** — `✨ feat(frontend): avisos do plano de IR em português e plano limpo ao mudar entradas`

---

## Task 11: Documentação e verificação final

**Files:**
- Modify: `docs/consolidacao_arquitetura_e_migracao.md` (§2.0 linha da Fase 4 + contagem de testes; §2.3 — adicionar "Revisão pós-Fases 3/4 (2026-10-27)" com o que foi corrigido e o backlog "Fora do escopo" deste plano)
- Modify: `docs/adr/0015-planejador-de-ir.md` (teto mínimo R$ 100, `MAX_MONTHS = 240`, padrão `ignore` agora também na UI)
- Modify: `docs/adr/0013-contrato-da-api-v1.md` (nota: `start` nulo = 1 ano; a SPA envia `1900-01-01` para "Tudo" e lê `meta.start`)

- [x] **Step 1:** Editar os três documentos.
- [x] **Step 2:** `cd backend && .venv/bin/pytest -q && .venv/bin/python manage.py check` → tudo verde, 0 issues.
- [x] **Step 3:** `cd frontend && npx vitest run && npm run build` → verde, build ok.
- [x] **Step 4: Commit** — `📝 docs: registra correções da revisão das Fases 3/4 e backlog`
