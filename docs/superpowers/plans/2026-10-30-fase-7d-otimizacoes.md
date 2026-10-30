# Fase 7d — Otimizações sem mudar números Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deixar a tela da carteira mais leve sem mudar nenhum número: um cálculo de análise compartilhado pelos 8 endpoints, o engine sem retrabalho e o bundle da SPA dividido.

**Architecture:**
- **Backend:** `run_analysis` passa a memoizar o `AnalysisContext` no Redis, por (usuário, versão do cache DLP, geração das séries de mercado, dia, pedido), com `single_flight`. Com isso, as 8 chamadas paralelas de uma tela fazem um único cálculo do dataset.
- **Engine:** limpezas da §5.7 da spec (`calculate_twr` sem reagrupar o que já chega agrupado; `date` em `datetime64` desde a DLP) e detecção de lacunas vetorizada.
- **Frontend:** rotas carregadas sob demanda (`React.lazy`) e ECharts importado por módulos (`echarts/core`), em vez do pacote inteiro.

**Tech Stack:** Python 3.12, Django 5, django-redis, pandas/numpy, pytest; React 18, Vite 6, ECharts 5 + `echarts-for-react`, Vitest.

**Spec:** [`../specs/2026-10-28-fase-7-confiabilidade-design.md`](../specs/2026-10-28-fase-7-confiabilidade-design.md) §5.7 (otimização), §4 (bundle) e §5.8. Doc mestre §5 "Fase 7d". Plano anterior: [7c.1](2026-10-30-fase-7c1-indices-anbima.md).

## Fatos medidos (2026-10-30, janela `w1-completo` da linha de base 7c, 2.950 linhas)

| Etapa | Tempo |
|---|---|
| `run_engine` completo (`fetch_user_portfolio` + `build_dataset` + 7 `compute_*`) | ~260 ms |
| `compute_irr_evolution` (122 XIRR) | ~100 ms |
| `build_dataset` (inclui `get_many` e `set_df` das séries) | ~110 ms |
| `gaps.find_gaps` + `gaps.stale_tail` (15.502 chamadas a `_to_date`) | ~33 ms |
| `calculate_twr` (reagrupa o `grp` já agrupado) | ~10 ms |
| leitura do histórico DLP no cache (`pd.read_json(orient="table")`) | ~35 ms |
| bundle `frontend/dist/assets/index-*.js` | 1.416.824 bytes, um único chunk |

Consequências para o escopo (decisão do usuário em 2026-10-30: "medido + bundle"):
- A tela da carteira dispara 8 POSTs (`summary`, `twr-evolution`, `drawdown`, `rolling-metrics`, `risk-return`, `shadow`, `yearly-summary`, `irr-evolution`), e cada um chama `run_analysis` do zero. **O memo da Task 4 é o ganho principal.**
- O pré-parse de `date` antes de `mc.set_df` (item 1 da §5.7) **não acelera a leitura**: o `orient="table"` já devolve `datetime64`. Fica só para o dtype ser o mesmo no caminho frio e no quente.
- **Fora do escopo:** chute quente no XIRR (usar a TIR do mês anterior como chute). Ele muda os últimos bits da TIR e quebra o critério de diferença zero. Registrar no ADR.
- O memo usa `cache.set` direto (pickle do django-redis, sem perda), **não** `mc.set_df`: o `to_json` arredonda para 10 casas decimais (`double_precision=10`) e mudaria número.

## Global Constraints

- **Critério da fase (spec §5.7):** **zero** diferença contra a linha de base `reports/parity/20261029-211840-fase7c/baseline` e gabarito/oráculo verdes. Nenhuma task pode mudar número; se a regressão acusar diferença, a task volta.
- TDD conforme `backend/CLAUDE.md`: escrever o teste, rodar, **ver falhar**, implementar, ver passar.
- Contrato da API v1 inalterado (ADR 0013): nenhum campo novo nem removido.
- O cache é otimização: qualquer falha do Redis no memo é logada (`logger.warning`) e o cálculo segue normalmente, sem propagar.
- Isolamento: a chave do memo **sempre** contém `user.pk`. Nunca servir o contexto de um usuário a outro.
- Commits: um por task, Conventional Commits + Gitmoji, imperativo em português, **nunca** `Co-Authored-By`. `git add` só com caminhos explícitos (nunca `-A`/diretório).
- `reports/` e `dlp_sheet/` nunca versionados; nada de valores em R$ em docs ou commits.
- Comandos (da raiz do repo):
  - backend: `cd backend && .venv/bin/pytest -q`
  - oráculo: `cd backend && .venv/bin/pytest -q -m integration engine/tests/test_oracle_dlp.py`
  - paridade: `cd scripts/parity && ../../backend/.venv/bin/pytest -q && cd .. && ../backend/.venv/bin/python -m parity.regress --baseline ../reports/parity/20261029-211840-fase7c/baseline`
  - SPA: `cd frontend && npm test -- --run && npm run build`

## Review Focus

1. **Dois usuários com o mesmo pedido** (mesmo período e benchmarks): cada um recebe a própria carteira, nunca o contexto memoizado do outro. Teste na Task 4.
2. **"Atualizar" depois de dado novo na DLP** (`POST /portfolio/refresh` sobe a geração DLP): a próxima chamada recalcula, não serve o contexto antigo por até 5 min. Teste na Task 4.
3. **Ingestão gravando pontos novos durante o dia** (`bump_series_generation`) ou **virada do dia** (o aviso de cauda parada depende de `today`): o memo não serve benchmark defasado nem aviso de lacuna velho. Testes na Task 4.
4. **Redis fora do ar ou lançando exceção** no `get`/`set` do memo: a análise responde normalmente (sem memo), sem 500. Teste na Task 4.
5. **Componente ECharts não registrado** depois do tree-shaking (ex.: legenda `type: 'scroll'`, `dataZoom` `slider`/`inside`, `markLine`, eixo `time`, `scatter`): o gráfico sai em branco em silêncio e só aparece um erro `[ECharts]` no console. Verificação dos 7 gráficos no LAB com o console limpo, Task 6.

---

### Task 1: Medidor de desempenho do engine (`parity.perf`)

Serve para medir antes e depois com o mesmo dado congelado. Os números entram no ADR da Task 6.

**Files:**
- Create: `scripts/parity/perf.py`
- Test: `scripts/parity/tests/test_perf.py`

**Interfaces:**
- Consumes: `parity.baseline.load_baseline(path) -> Baseline` (`cases`, `snapshot`, `hists`); `parity.engine_side.run_engine(case, snapshot, hist)`.
- Produces:
  - `time_case(case, snapshot, hist, repeat: int = 5) -> dict[str, float]`: mediana em ms de cada etapa, com as chaves `"fetch_user_portfolio"`, `"build_dataset"`, `"compute_irr_evolution"` e `"total"`. Uma execução de aquecimento antes, descartada. Roda dentro de `engine_patches(snapshot, hist)`, como `run_engine`.
  - CLI `python -m parity.perf --baseline <dir> [--repeat N]`: uma linha por caso, `caso  total=…ms  fetch=…ms  build=…ms  irr=…ms`. Sem valores em R$.

- [ ] **Step 1: Escrever o teste** `test_time_case_reports_positive_medians`: usa o mesmo caso sintético dos testes existentes (`parity/tests/synthetic.py`); `time_case(..., repeat=2)` devolve exatamente as 4 chaves, todos os valores `> 0`, e `total >= fetch_user_portfolio`.
- [ ] **Step 2: Rodar e ver falhar**
  Run: `cd scripts/parity && ../../backend/.venv/bin/pytest -q tests/test_perf.py`
  Expected: FAIL com `ModuleNotFoundError: parity.perf`
- [ ] **Step 3: Implementar `time_case` e o `main`** em `scripts/parity/perf.py`, com o mesmo bootstrap de `sys.path`/Django de `regress.py`. Medir com `time.perf_counter`, usando `statistics.median`.
- [ ] **Step 4: Rodar e ver passar**, depois medir o estado atual
  Run: `cd scripts/parity && ../../backend/.venv/bin/pytest -q && cd .. && ../backend/.venv/bin/python -m parity.perf --baseline ../reports/parity/20261029-211840-fase7c/baseline --repeat 5`
  Expected: testes verdes, e uma linha por caso com `total` próximo de 260 ms em `w1-completo`. **Anotar a saída** (vai para o ADR).
- [ ] **Step 5: Medir o LAB antes da otimização.** No navegador, com o DevTools → Network: aba Performance da carteira, período "Tudo", benchmarks padrão, após um F5 (DLP já em cache). Anotar o tempo de cada uma das 8 chamadas `/api/v1/portfolio/*` e o tamanho transferido do JS. Sem valores em R$.
- [ ] **Step 6: Commit**
  ```bash
  git add scripts/parity/perf.py scripts/parity/tests/test_perf.py
  git commit -m "🔧 chore(parity): medidor de desempenho do engine sobre a linha de base"
  ```

---

### Task 2: Engine sem retrabalho de agrupamento e `date` tipado desde a DLP (§5.7)

**Files:**
- Modify: `backend/engine/flows.py` (nova `is_grouped`)
- Modify: `backend/engine/twr.py:49-57` (atalho quando a entrada já está agrupada)
- Modify: `backend/apps/market/services/sources/dlp.py:168-174` (`date` → `datetime64` antes de `mc.set_df`)
- Test: `backend/engine/tests/test_flows.py`, `backend/engine/tests/test_twr.py`, `backend/apps/market/tests/test_sources_dlp.py` (ou o arquivo onde estão os testes de `buscar_historico`; localizar com `grep -rln buscar_historico backend/apps/market/tests`)

**Interfaces:**
- Produces:
  - `flows.is_grouped(df: pd.DataFrame) -> bool`: `True` se e somente se `df` tem a coluna `date` com dtype `datetime64`, essa coluna é única e crescente (`is_unique` e `is_monotonic_increasing`), e todas as `FLOW_COLUMNS` estão presentes e são numéricas. `None`/vazio → `False`.
  - `calculate_twr`: se `is_grouped(df)`, usa `df.reset_index(drop=True)` (cópia) como `df_grp` e não chama `group_portfolio`. Senão, o caminho atual, inalterado. Assinatura e retorno iguais.
  - `dlp.buscar_historico`: no caminho frio, `df["date"] = pd.to_datetime(df["date"])` logo após `_norm_date_col`, se a coluna existir. Os caminhos frio e quente devolvem `date` em `datetime64[ns]`.

- [ ] **Step 1: Escrever os testes**
  - `test_is_grouped_true_for_group_portfolio_output`: `is_grouped(group_portfolio(df_cru))` é `True`.
  - `test_is_grouped_false_cases`: datas repetidas; datas fora de ordem; `date` como string; falta `lucro`; DataFrame vazio → todos `False`.
  - `test_calculate_twr_skips_regroup_when_grouped`: `monkeypatch` de `engine.twr.group_portfolio` com uma função que levanta `AssertionError`; `calculate_twr(grp_pronto)` não levanta.
  - `test_calculate_twr_same_result_grouped_or_raw`: para a mesma carteira (com venda com lucro, provento e duas linhas por data), `calculate_twr(df_cru)` e `calculate_twr(group_portfolio(df_cru) + fluxo)` são idênticos com `pd.testing.assert_series_equal(check_exact=True)`.
  - `test_buscar_historico_date_is_datetime_cold_and_warm`: payload DLP com `"date": "2024-01-31"`. Na 1ª chamada (fria) e na 2ª (quente, com `locmem_cache`), `pd.api.types.is_datetime64_any_dtype(df["date"])` é verdadeiro.
- [ ] **Step 2: Rodar e ver falhar**
  Run: `cd backend && .venv/bin/pytest -q engine/tests/test_flows.py engine/tests/test_twr.py apps/market/tests -k "is_grouped or regroup or grouped_or_raw or date_is_datetime"`
  Expected: FAIL (`AttributeError: ... is_grouped`; `AssertionError` do `group_portfolio`; dtype `object` no caminho frio)
- [ ] **Step 3: Implementar** `is_grouped`, o atalho em `calculate_twr` e o `to_datetime` em `buscar_historico`.
- [ ] **Step 4: Rodar a suíte, o oráculo e a paridade**
  Run: comandos de backend, oráculo e paridade da seção Global Constraints.
  Expected: tudo verde e `regress` com **todas** as métricas/janelas `=` (zero colunas alteradas).
- [ ] **Step 5: Commit**
  ```bash
  git add backend/engine/flows.py backend/engine/twr.py backend/apps/market/services/sources/dlp.py backend/engine/tests/test_flows.py backend/engine/tests/test_twr.py <arquivo de teste da dlp>
  git commit -m "⚡️ perf(engine): calculate_twr não reagrupa carteira já agrupada e date tipado desde a DLP"
  ```

---

### Task 3: Detecção de lacunas vetorizada (`apps/market/gaps.py`)

**Files:**
- Modify: `backend/apps/market/gaps.py` (`find_gaps`, `stale_tail`; `_to_date` continua só para `until`)
- Test: `backend/apps/market/tests/test_gaps.py`

**Interfaces:**
- Produces (assinaturas e retornos **inalterados**, `Gap` com `dt.date`):
  - `_clean_dates(dates: Iterable) -> pd.DatetimeIndex`: `pd.DatetimeIndex(pd.to_datetime(list(dates) if not isinstance(dates, pd.Index) else dates)).normalize().unique().sort_values()`. Normalizar equivale ao `_to_date` atual, que descarta a hora.
  - `find_gaps`: as diferenças em dias saem de `np.diff(idx.values).astype("timedelta64[D]").astype(int)`. Só as posições com `> tolerance` viram `Gap(idx[i].date(), idx[i+1].date())`.
  - `stale_tail`: `last = idx[-1].date()`; o resto não muda.

- [ ] **Step 1: Escrever o teste de equivalência** `test_vectorized_gaps_match_reference`: uma implementação de referência no próprio teste (cópia literal da lógica atual com `_to_date` + `sorted(set(...))` + `zip`) comparada com `find_gaps`/`stale_tail` (`==` nas listas de `Gap`), para cada entrada:
  - `DatetimeIndex` diário com horas diferentes de zero e duplicatas;
  - lista de `dt.date` fora de ordem;
  - `pd.Timestamp` misturado com `dt.date`;
  - mensal com buraco de 60 dias;
  - vazio;
  - um elemento só;
  - ambas as periodicidades.

  Junto, `test_find_gaps_does_not_call_to_date_per_element`: `monkeypatch` de `gaps._to_date` com uma função que conta chamadas. Com um índice de 1.000 datas, `find_gaps` faz 0 chamadas e `stale_tail` faz no máximo 1 (o `until`).
- [ ] **Step 2: Rodar e ver falhar**
  Run: `cd backend && .venv/bin/pytest -q apps/market/tests/test_gaps.py`
  Expected: a equivalência passa (a referência é o código atual) e `test_find_gaps_does_not_call_to_date_per_element` FALHA com contagem 1.000.
- [ ] **Step 3: Implementar `_clean_dates`** e reescrever `find_gaps`/`stale_tail` sobre ele.
- [ ] **Step 4: Rodar a suíte, a paridade e o medidor**
  Run: backend + paridade (Global Constraints) + `python -m parity.perf ...` da Task 1.
  Expected: verde, `regress` com zero colunas alteradas, `meta.gaps` igual (os testes de `test_dataset`/`test_analysis` cobrem). O `total` de `w1-completo` cai uns ~30 ms em relação à Task 1.
- [ ] **Step 5: Commit**
  ```bash
  git add backend/apps/market/gaps.py backend/apps/market/tests/test_gaps.py
  git commit -m "⚡️ perf(market): detecção de lacunas vetorizada, sem conversão de data por elemento"
  ```

---

### Task 4: Memo do `AnalysisContext` compartilhado entre os endpoints

**Files:**
- Modify: `backend/apps/market/services/sources/dlp.py` (nova `user_cache_version`)
- Modify: `backend/apps/api/analysis.py` (`run_analysis` vira memo sobre `_compute_analysis`)
- Test: `backend/apps/api/tests/test_analysis_memo.py` (novo); `backend/apps/market/tests/<testes da dlp>`

**Interfaces:**
- Consumes: `apps.market.cache.series_generation() -> int`, `mc.make_key(*parts) -> str`, `mc.single_flight(key, timeout=90)`, `dlp.invalidate_user_cache(token)`, `cache.bump_series_generation()`.
- Produces:
  - `dlp.user_cache_version(token: str) -> str`: `f"{_token_hash(token)}:{_generation(token)}"`. Muda quando o token muda e quando há `invalidate_user_cache`.
  - Em `analysis.py`:
    - `ANALYSIS_TTL = 300` (segundos; menor que `TTL_DLP = 600`).
    - `_request_digest(valid: ValidRequest, req: AnalysisRequest) -> str`: `hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]`, com `payload = {"start": valid.start.isoformat(), "end": valid.end.isoformat(), "ativo": req.ativo, "classe": req.classe, "benchmarks": valid.benchmarks, "synthetics": [[s.base, s.operador, s.valor] for s in valid.specs]}`. A ordem de `benchmarks` é mantida, porque define a ordem das colunas. Campos fora dessa lista (ex.: `window` do `RollingRequest`) **não** entram.
    - `_analysis_key(user, token, valid, req) -> str`: `mc.make_key("analysis", user.pk, dlp.user_cache_version(token), mc.series_generation(), dt.date.today().isoformat(), _request_digest(valid, req))`.
    - `_compute_analysis(valid: ValidRequest, req: AnalysisRequest, token: str) -> AnalysisContext`: o corpo atual de `run_analysis` depois de `validate_request`/`get_token`, sem mudança de comportamento (inclusive os `ApiError` 404 e de DLP).
    - `run_analysis(user, req) -> AnalysisContext` (mesma assinatura): `validate_request`, `get_token`, monta a chave; `cache.get(key)` → se for `AnalysisContext`, devolve. Senão, `with mc.single_flight(key):` relê o cache e, se ainda faltar, chama `_compute_analysis` e grava com `cache.set(key, ctx, timeout=ANALYSIS_TTL)`. Exceções de `cache.get`/`cache.set` são capturadas com `logger.warning` e seguem sem memo. `ApiError` **não** é memoizado (propaga). Usar `django.core.cache.cache` direto (pickle, sem perda), **não** `mc.set_df`.

- [ ] **Step 1: Escrever os testes** em `test_analysis_memo.py`, sobre o fixture `portfolio_env`/`user_with_token` do `conftest.py` da API, com um contador em volta do `fetch_user_portfolio` falso (`monkeypatch` de `apps.api.analysis.fetch_user_portfolio`):
  - `test_same_request_computes_once`: duas chamadas a `run_analysis` com o mesmo pedido → contador 1. `df_combined`, `portfolio_df` e `rf` iguais aos da 1ª chamada com `assert_frame_equal`/`assert_series_equal(check_exact=True)`; `gaps`, `missing`, `start` e `end` iguais.
  - `test_rolling_window_shares_context`: `AnalysisRequest(...)` e depois `RollingRequest(..., window=63)` com os mesmos campos → contador 1.
  - `test_different_request_recomputes`: benchmarks diferentes → 2; ordem de benchmarks diferente → 2; `classe` diferente → 2.
  - `test_users_never_share_context`: `user_with_token` e `other_user` (com token próprio) com o mesmo pedido → contador 2, e cada contexto vem do `fetch` do próprio token (o falso devolve carteiras distintas por token).
  - `test_dlp_refresh_invalidates`: chamada, `dlp.invalidate_user_cache(token)`, chamada → 2.
  - `test_series_generation_bump_invalidates`: chamada, `apps.market.cache.bump_series_generation()`, chamada → 2.
  - `test_day_rollover_invalidates`: `monkeypatch` de `apps.api.analysis.dt` (ou de um `_today()` extraído para isso, de preferência) para o dia D e depois D+1 → 2.
  - `test_cache_failure_falls_back_to_compute`: `monkeypatch` de `apps.api.analysis.cache.get` e `.set` levantando `ConnectionError` → `run_analysis` devolve o contexto normalmente (contador 1 por chamada) e sem exceção.
  - `test_api_error_not_memoized`: o `fetch` falso devolve `(None, None)` na 1ª chamada e a carteira na 2ª → a 1ª levanta `ApiError` 404 e a 2ª devolve o contexto (contador 2).
  - Teste da DLP: `test_user_cache_version_changes_on_invalidate_and_token`.
- [ ] **Step 2: Rodar e ver falhar**
  Run: `cd backend && .venv/bin/pytest -q apps/api/tests/test_analysis_memo.py apps/market/tests -k "memo or user_cache_version or share or invalidat or rollover or not_memoized or falls_back or computes_once or recomputes"`
  Expected: FAIL (contador 2 onde se espera 1; `AttributeError: user_cache_version`)
- [ ] **Step 3: Implementar** `dlp.user_cache_version` e o memo em `analysis.py`. Se o teste de virada de dia pedir, extrair `def _today() -> dt.date` em `analysis.py` e usá-lo também em `validate_request` (o default `today` dela continua igual).
- [ ] **Step 4: Rodar a suíte inteira** (os testes de contrato da API em `test_portfolio_*.py` precisam continuar verdes sem mudança) **e a paridade**
  Run: backend + paridade (Global Constraints).
  Expected: verde; `regress` com zero colunas alteradas (a paridade não passa pelo `run_analysis`, então isso confirma que o engine não mudou).
- [ ] **Step 5: Commit**
  ```bash
  git add backend/apps/api/analysis.py backend/apps/market/services/sources/dlp.py backend/apps/api/tests/test_analysis_memo.py <arquivo de teste da dlp>
  git commit -m "⚡️ perf(api): memo do contexto de análise compartilhado pelos endpoints da carteira"
  ```

---

### Task 5: Bundle da SPA dividido (rotas sob demanda + ECharts por módulos)

**Files:**
- Create: `frontend/src/components/portfolio/charts/echarts.js`
- Modify: `frontend/src/components/portfolio/charts/BaseChart.jsx:2,~105` (usa `echarts-for-react/lib/core`)
- Modify: `frontend/src/App.jsx:9-10` e as `Route` (`React.lazy` + `Suspense`)
- Modify (mocks): `frontend/src/test/{performance-charts,error-states,token-gating,app-routing,summary-cards-chart,header-dlp,risk-charts-page}.test.jsx`: `vi.mock('echarts-for-react', …)` → `vi.mock('echarts-for-react/lib/core', …)`. O conteúdo do mock não muda.

**Interfaces:**
- Produces:
  - `echarts.js`: `import * as echarts from 'echarts/core'` e `echarts.use([...])` com exatamente `LineChart`, `ScatterChart` (`echarts/charts`); `GridComponent`, `TooltipComponent`, `LegendComponent` (já inclui `plain` e `scroll`), `DataZoomComponent` (`inside` e `slider`), `MarkLineComponent`, `TitleComponent` (`echarts/components`); `CanvasRenderer` (`echarts/renderers`). `export { echarts };`
  - `BaseChart`: `<ReactEChartsCore echarts={echarts} …/>` com as mesmas props de hoje (`opts={{ renderer: 'canvas' }}`, `lazyUpdate`, etc.).
  - `App.jsx`: `const PortfolioPage = lazy(() => import('./pages/PortfolioPage').then((m) => ({ default: m.PortfolioPage })))`, e igual para `MigrationPage`. As `Routes` ficam dentro de `<Suspense fallback={<Skeleton … />}>`, reaproveitando `components/common/Skeleton`. `LoginPage` continua eager.

- [ ] **Step 1: Registrar o estado atual como falha do critério**
  Run: `cd frontend && npm run build && ls -l dist/assets/*.js && for f in dist/assets/index-*.js; do grep -c "echarts" "$f"; done`
  Expected (FALHA do critério): um único `index-*.js` com ~1,4 MB (> 500.000 bytes), contendo `echarts`.
- [ ] **Step 2: Ajustar os mocks dos 7 testes** para `echarts-for-react/lib/core`. Nos testes de rota (`app-routing.test.jsx` e os que renderizam `App`), trocar `getBy…` por `await findBy…` onde a página agora é lazy.
  Run: `cd frontend && npm test -- --run`
  Expected: FAIL nos testes de gráfico (o `BaseChart` ainda importa `echarts-for-react`, o mock não pega e o ECharts real quebra no jsdom).
- [ ] **Step 3: Implementar** `echarts.js`, o `BaseChart` com `ReactEChartsCore` e as rotas lazy no `App.jsx`.
- [ ] **Step 4: Rodar os testes e o critério do bundle**
  Run: `cd frontend && npm test -- --run && npm run build && ls -l dist/assets/*.js && for f in dist/assets/index-*.js; do grep -c "echarts" "$f"; done`
  Expected: 107+ testes verdes; o build **sem** o aviso "Some chunks are larger than 500 kB"; nenhum `.js` > 500.000 bytes; o chunk de entrada `index-*.js` com contagem `0` de `echarts`. Anotar os tamanhos (vão para o ADR).
- [ ] **Step 5: Conferir no navegador local** (`npm run dev` + `runserver`): abrir as abas Performance e Risco e a Migração de IR, com o console sem nenhum erro `[ECharts]` e os 7 gráficos desenhados (legenda com rolagem, zoom por slider e por roda, linha "a.a. a partir daqui" na TIR, dispersão risco × retorno).
- [ ] **Step 6: Commit**
  ```bash
  git add frontend/src/components/portfolio/charts/echarts.js frontend/src/components/portfolio/charts/BaseChart.jsx frontend/src/App.jsx frontend/src/test/performance-charts.test.jsx frontend/src/test/error-states.test.jsx frontend/src/test/token-gating.test.jsx frontend/src/test/app-routing.test.jsx frontend/src/test/summary-cards-chart.test.jsx frontend/src/test/header-dlp.test.jsx frontend/src/test/risk-charts-page.test.jsx
  git commit -m "⚡️ perf(frontend): rotas sob demanda e ECharts por módulos para dividir o bundle"
  ```

---

### Task 6: Verificação final, LAB, ADR 0022 e fechamento da Fase 7

**Files:**
- Create: `docs/adr/0022-otimizacoes-fase-7d.md`
- Modify: `docs/adr/README.md`, `docs/consolidacao_arquitetura_e_migracao.md` (§2.0, §2.3, §5 "Fase 7d", §9 item 16), `docs/README.md`, `backend/README.md`, `backend/CLAUDE.md` (se surgiu convenção nova: "memo de análise via `cache.set` direto, nunca `set_df`, por causa da precisão")

- [ ] **Step 1: Definition of Done**
  Run: backend + oráculo + paridade + SPA (Global Constraints) + `cd backend && .venv/bin/python manage.py check` + `python -m parity.perf ...`
  Expected: tudo verde; `regress` com zero colunas alteradas em todas as métricas/janelas; medidor anotado (antes × depois).
- [ ] **Step 2: Rebuild do LAB**
  Run: `docker compose up -d --build && docker compose ps`
  Expected: `backend`, `web` e `ingestion` de pé (os dois primeiros healthy).
- [ ] **Step 3: Medir o LAB depois** com o mesmo roteiro da Task 1 Step 5. Esperado: na 2ª carga da tela, as 8 chamadas sem refazer o dataset (latência bem menor que antes na maioria delas); o JS inicial bem menor que 1,4 MB. Conferir também os 7 gráficos com o console limpo (Review Focus 5), e que "Atualizar" depois de mudar o filtro de classe traz número novo.
- [ ] **Step 4: ADR 0022** (sem valores em R$), com as seções:
  - **Contexto:** os 8 endpoints refazendo `run_analysis`; a tabela de "Fatos medidos".
  - **Decisão:** memo por (usuário, versão DLP, geração de mercado, dia, pedido), TTL 300 s, `single_flight`, pickle (por que não `set_df`); `calculate_twr` sem reagrupar; gaps vetorizado; rotas lazy + ECharts por módulos.
  - **Rejeitado:** chute quente no XIRR (muda bits, viola a diferença zero); pré-parse de data como ganho de velocidade (o `orient="table"` já tipa).
  - **Consequências:** tempos antes × depois (medidor e LAB); bundle antes × depois; um componente ECharts novo exige registro em `echarts.js`.

  Depois, o checklist de `docs/README.md` (contagem de testes, §9 item 16 marcado, Fase 7 encerrada no doc mestre).
- [ ] **Step 5: Commit**
  ```bash
  git add docs/adr/0022-otimizacoes-fase-7d.md docs/adr/README.md docs/consolidacao_arquitetura_e_migracao.md docs/README.md backend/README.md backend/CLAUDE.md
  git commit -m "📝 docs: ADR 0022 (otimizações da Fase 7d) e fechamento da Fase 7"
  ```
- [ ] **Step 6: PARAR para o usuário testar** (spec §5.6). Mensagem ao usuário:
  - abrir a carteira em "Tudo" e trocar de aba/período: tem que estar mais rápido, com os mesmos números de antes;
  - "Atualizar" continua trazendo dado novo da DLP na hora;
  - os 7 gráficos e a Migração de IR desenham normalmente.

  A Fase 7 só fecha quando os achados dele tiverem decisão.
