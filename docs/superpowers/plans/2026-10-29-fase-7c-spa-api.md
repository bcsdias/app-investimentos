# Fase 7c — Achados do teste da 7b e backlog SPA/API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corrigir os achados do teste da Fase 7b feito pelo usuário (lacunas silenciosas nas bordas da janela, coluna de ano vazia, legenda sobreposta) e zerar o backlog SPA/API da §2.3 do doc mestre, com adiamentos justificados.

**Architecture:** Duas partes.
- **Parte A (Tasks 1–3b), achados do usuário:** cada achado vira primeiro um teste vermelho (spec §2, decisão 3). Essa parte mexe em `engine/dataset.py`, `apps/market/gaps.py`, `engine/compute.py`, `engine/irr.py` e `BaseChart.jsx`, e acrescenta campos aditivos de TIR do período à API.
- **Parte B (Tasks 4–11, com 10b), backlog:** agrupado por área.
  - `apps/market` (cache e DLP);
  - `apps/api` e `apps/accounts` (contrato, token, login);
  - `apps/migration_ir` (planner e PDF);
  - SPA (fluxo do token, filtros e URL, componentes).
- Contrato da API v1 só com mudanças aditivas.

**Tech Stack:** Python 3.12, Django 5 + django-ninja, django-redis, pandas, fpdf2, pytest; React + React Router + TanStack Query + ECharts, Vitest.

**Spec:** [`docs/superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md`](../specs/2026-10-28-fase-7-confiabilidade-design.md): §5.3.1 (linha "Buraco de dado de mercado"), §5.5 (backlog e políticas), §5.6 (ponto de espera). Backlog de origem: [`docs/consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md) §2.3 ("Backlog da revisão de 2026-10-27" e "Menores da revisão final"). Planos anteriores: [7a](2026-10-28-fase-7a-engine.md), [7b](2026-10-28-fase-7b-mercado.md); [ADR 0019](../../adr/0019-integridade-dos-dados-de-mercado.md).

## Achados do teste do usuário (2026-10-29)

Período personalizado 01/01/2024–31/12/2024 com CDI, BOVA11, IMA-S (BCB) e LFTS11.SA (screenshots locais `docs/teste_img{1..4}.png`, ignorados pelo git). Cruzados com `reports/fase7b/gaps-depois.txt`:

| # | Sintoma | Causa |
|---|---|---|
| A1 | IMA-S só começa em 22/02/2024 no TWR, sai "—" na tabela anual e some do shadow. O aviso **não** lista lacuna | A lacuna `bcb:12469` vai de 2023-05-22 a 2024-02-21 e atravessa a abertura (31/12/2023). O lookback de 40 dias não alcança o último ponto antes dela. O `find_gaps` só vê a série a partir de 22/02 e não acha nada |
| A2 | LFTS11.SA fica reta de 26/09 a 31/12/2024 (+7,93 % × CDI +10,88 %). O aviso só mostra a lacuna de junho | A lacuna `yf:LFTS11.SA` de 2024-09-26 a 2025-02-24 atravessa o **fim** da janela. A série é buscada só até `end`, então o ponto de 2025-02-24 nunca é visto |
| A3 | Pelo diagnóstico, `bcb:12469` tem última data 2026-04-10: série parada há 6 meses, sem aviso | O `find_gaps` só olha pares de pontos. Cauda parada não é detectada nem no `meta` nem no `market_gaps` |
| A4 | Tabela anual com coluna "2023" toda 0,00 % | Desde a 7a, a abertura em 31/12/2023 cria um "ano" com uma linha só |
| A5 | Legenda do shadow: "LFTS11.SA" desenhado em cima de "Total Investido" | A legenda quebra em 2 linhas e invade a área do gráfico (`BaseChart` com `legend.top: 0` e `grid` fixo) |

| A6 | "Evolução Histórica da TIR" começa em −32 % a.a. em fevereiro | Anualizar um mês amplifica o resultado. A convenção de mercado (ANBIMA) não divulga rentabilidade anualizada de período menor que 12 meses |
| A7 | O calendário do campo de data só abre clicando num espaço vazio à direita do texto | O ícone nativo existe, mas o Chrome o desenha preto sobre `bg-zinc-950`, porque falta `color-scheme: dark` |

O restante bate com o relatório da 7b: o aviso de lacuna de junho do LFTS11 aparece, a abertura dos benchmarks está em 31/12/2023, e no host rodam 421 testes backend + 71 da SPA verdes.

**Decisões do usuário (2026-10-29):**
1. `period` na URL com `1a|2a|5a|tudo|custom`, espelhando os botões existentes.
2. Adiamentos da Task 12 aprovados.
3. TIR com **menos de 12 meses** desde a abertura do período: mostrar a TIR **do período** (acumulada, sem anualizar). Task 3b.
4. Campo de data: manter o `<input type="date">` nativo. O formato (DD/MM ou MM/DD) segue o idioma do navegador do usuário. Adicionar um ícone de calendário visível e abrir o calendário com um clique em qualquer parte do campo. Task 10b.

## Global Constraints

- TDD conforme `backend/CLAUDE.md`: escrever o teste, rodar e **ver falhar**, implementar, ver passar.
- Commits: um por task, Conventional Commits + Gitmoji, imperativo em português, **nunca** `Co-Authored-By`. `git add` só com caminhos explícitos (há screenshots sensíveis soltos em `docs/`).
- Buraco de dado de mercado: `ffill` (último valor conhecido) **mais** aviso em `meta.gaps`; **nunca** interpolação.
- Contrato da API v1 ([ADR 0013](../../adr/0013-contrato-da-api-v1.md)) só com mudanças aditivas: nenhuma chave existente some ou muda de tipo.
- Erros de API sempre no envelope `{"code", "message", "details"}` em português. DLP rejeitando o token é sempre `422 dlp_token_rejected`, nunca `401`.
- Routers finos: nenhum cálculo em `apps/api/routers/*`.
- Cache é otimização: toda falha de Redis é logada e engolida. A falha nunca pode servir dado antigo nem liberar o lock de outro.
- Rate limit de login: **5 falhas em 15 min por (usuário, IP)**, contador no Redis existente, sem dependência nova. Resposta `429` no envelope `ApiError`.
- `period` na URL: `?period=1a|2a|5a|tudo|custom&start=AAAA-MM-DD&end=AAAA-MM-DD`. A spec escreveu `3a`, mas os presets reais da SPA são `1A/2A/5A`, então valem esses.
- PDF: nenhuma linha de tabela nem bloco de mês cortado entre páginas.
- TIR exibida: com data do ponto `< abertura + 12 meses` → TIR do período `(1 + xirr)^(dias/365) − 1` (Actual/365, a mesma base do XIRR); a partir de 12 meses → `xirr` a.a. O `xirr` a.a. continua sendo calculado e devolvido como hoje.
- `reports/` e `dlp_sheet/` nunca versionados; relatórios, ADR e commits sem valores em R$ da carteira.
- Rodar a partir de `backend/`: `.venv/bin/pytest -q`. Paridade: `cd scripts/parity && ../../backend/.venv/bin/pytest -q`. SPA: `cd frontend && npm test -- --run`.
- Linha de base vigente: `reports/parity/20261029-122417-fase7b/baseline`.

## Review Focus

1. Benchmark que **começa depois** da abertura (ETF lançado no meio da janela) não ganha lacuna inventada, e a re-busca do histórico acontece uma vez só, sem laço. Teste na Task 1.
2. Série parada **antes da janela inteira** (última data < abertura) aparece como lacuna de `<última data>` até o fim da janela, com o valor repetido, em vez de sumir. Teste na Task 1.
3. Token rejeitado com as 8 queries do dashboard em paralelo: o backend chama a DLP no máximo uma vez por geração, a página mostra **um** estado e o modal não abre sozinho. Testes nas Tasks 4 e 9.
4. Rate limit: as falhas de um usuário não bloqueiam outro usuário nem o mesmo usuário em outro IP; login certo zera o contador; Redis fora do ar não bloqueia ninguém. Teste na Task 6.
5. URL com `period` desconhecido ou data malformada cai em "Tudo" sem quebrar a página. Teste na Task 10.

---

## Parte A — Achados do teste da 7b

### Task 1: Lacunas nas bordas da janela e séries paradas (A1, A2, A3)

**Files:**
- Modify: `backend/apps/market/gaps.py`, `backend/engine/dataset.py` (`build_dataset`), `backend/apps/market/management/commands/market_gaps.py`
- Test: `backend/apps/market/tests/test_gaps.py`, `backend/engine/tests/test_dataset.py`, `backend/apps/market/tests/test_commands.py`

**Interfaces:**
- Produces:
  - `gaps.stale_tail(dates: Iterable, periodicity: str, until: dt.date) -> Gap | None`: com `last` = maior data, devolve `Gap(last, until)` se `(until − last).days > TOLERANCE_DAYS[periodicity]`; senão `None`. Série vazia → `None`.
  - `dataset._today() -> dt.date` (monkeypatchável nos testes).
  - Em `build_dataset`, para cada benchmark e para a base de cada sintético:
    1. busca em `[start − BENCH_LOOKBACK_DAYS, max(end, _today())]`;
    2. se a série não tem nenhum ponto `<= crop_start`, busca **uma** vez mais a partir de `catalog.EPOCH` (mesmo fim);
    3. faz `concat`, `ffill` e recorte em `[crop_start, crop_end]` como hoje;
    4. as lacunas são as de `find_gaps` sobre a série inteira buscada que cruzam `[crop_start, crop_end]` (filtro atual), mais `stale_tail(..., until=min(crop_end, _today()))`, se houver, com o mesmo filtro;
    5. `SeriesGap.end` da cauda é o fim da janela (documentar no ADR).
  - `rf = get_series("CDI", start, end)` não muda.
  - `market_gaps`: depois das lacunas, imprime `"  defasada desde <última> (<n> dias)"` quando `stale_tail(..., until=hoje)` acha algo.

- [ ] **Step 1: Escrever os testes**

```python
# test_gaps.py
def test_stale_tail_beyond_tolerance():
    assert stale_tail([D(2026,4,1), D(2026,4,10)], "daily", D(2026,10,29)) == Gap(D(2026,4,10), D(2026,10,29))
def test_stale_tail_within_tolerance_is_none():
    assert stale_tail([D(2026,10,20)], "daily", D(2026,10,29)) is None
def test_stale_tail_empty_is_none():
    assert stale_tail([], "monthly", D(2026,10,29)) is None

# test_dataset.py — get_many/get_series fakes respeitam (start, end) e registram as chamadas;
# carteira em 2023-12-31 e 2024-12-31; start_date="2024-01-01", end_date="2024-12-31"; _today=2026-10-29
def test_gap_crossing_opening_beyond_lookback_is_reported_and_ffilled(...):
    # IMA-S com pontos em 2023-05-22 (100.0) e 2024-02-21 (101.0) e diários depois
    assert SeriesGap("IMA-S", D(2023,5,22), D(2024,2,21)) in res.gaps
    assert res.df_combined["IMA-S"].iloc[0] == pytest.approx(100.0)       # rebase na abertura, sem NaN
def test_gap_crossing_window_end_is_reported(...):
    # LFTS11.SA diário até 2024-09-26, depois 2025-02-24 em diante
    assert SeriesGap("LFTS11.SA", D(2024,9,26), D(2025,2,24)) in res.gaps
def test_series_stale_until_today_reports_tail(...):
    # IMA-S diário de 2023-01-02 até 2024-04-10 e nada depois
    assert SeriesGap("IMA-S", D(2024,4,10), D(2024,12,31)) in res.gaps
def test_series_starting_after_opening_has_no_gap_and_refetches_once(...):
    # ETF com primeiro ponto em 2024-06-03 -> res.gaps == []; o fake registra
    # exatamente 2 buscas desse nome (janela + EPOCH)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/market/tests/test_gaps.py engine/tests/test_dataset.py apps/market/tests/test_commands.py -q`
Expected: FAIL (`ImportError: stale_tail`; lacunas ausentes em `res.gaps`; `IMA-S` NaN na abertura)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar suítes e regressão**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q && ../../backend/.venv/bin/python -m parity.regress --baseline ../../reports/parity/20261029-122417-fase7b/baseline`
Expected: tudo `=`. Os benchmarks da matriz (`CDI`, `IPCA + 6%`, `S&P 500 BRL`, `IFIX`) não têm lacuna maior que 40 dias na abertura, e o snapshot congelado termina na mesma data. Qualquer diferença → **parar e investigar**.

- [ ] **Step 5: Diagnosticar `bcb:12469` parada desde 2026-04-10**

Run: `cd backend && .venv/bin/python manage.py ingest_market_data --only bcb:12469` e consultar a fonte direto: `curl -s "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12469/dados/ultimos/5?formato=json"`.
Expected: um de dois resultados, a registrar no ADR (Task 11):
- a fonte também parou: buraco de fonte, coberto pelo aviso desta task;
- a fonte tem dado: bug de ingestão, **parar e reportar** antes de seguir.

- [ ] **Step 6: Commit**

```bash
git add backend/apps/market/gaps.py backend/engine/dataset.py backend/apps/market/management/commands/market_gaps.py backend/apps/market/tests/test_gaps.py backend/engine/tests/test_dataset.py backend/apps/market/tests/test_commands.py
git commit -m "🐛 fix(engine): avisa lacunas que cruzam as bordas da janela e séries paradas"
```

---

### Task 2: Tabela anual sem o ano que só tem a linha de abertura (A4)

**Files:**
- Modify: `backend/engine/compute.py` (`compute_yearly_summary`)
- Test: `backend/engine/tests/test_compute.py`

**Interfaces:**
- Produces: `compute_yearly_summary(df_combined)`. Se o 1º ano do índice tem **uma única** linha e existe ano seguinte, esse ano sai do resultado. O retorno do ano seguinte continua medido a partir dela (`pct_change` sobre o fechamento da abertura). Janela de um ano só com uma linha: comportamento atual.

- [ ] **Step 1: Escrever o teste**

```python
def test_yearly_drops_opening_only_year():
    df = pd.DataFrame({"Carteira": [100.0, 105.0, 110.0]},
                      index=pd.to_datetime(["2023-12-31", "2024-06-30", "2024-12-31"]))
    out = compute.compute_yearly_summary(df)
    assert "2023" not in [str(c) for c in out.columns]     # ajustar ao rótulo real da coluna
    assert out.loc["Carteira"].iloc[0] == pytest.approx(0.10)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine/tests/test_compute.py -q`
Expected: FAIL (coluna 2023 presente)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar suítes, regressão e nova linha de base**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q && ../../backend/.venv/bin/python -m parity.regress --baseline ../../reports/parity/20261029-122417-fase7b/baseline`
Expected: só `yearly` muda, e só nos casos cuja abertura é o último dia de um ano (perde a coluna desse ano). Qualquer outra mudança → **parar e investigar**. Depois gravar com `--save ../../reports/parity/$(date +%Y%m%d-%H%M%S)-fase7c` e re-verificar contra ela: tudo `=`.

- [ ] **Step 5: Commit**

```bash
git add backend/engine/compute.py backend/engine/tests/test_compute.py
git commit -m "🐛 fix(engine): tabela anual sem o ano que só contém a abertura"
```

---

### Task 3: Legenda dos gráficos sem sobrepor o traçado (A5)

**Files:**
- Modify: `frontend/src/components/portfolio/charts/BaseChart.jsx`
- Test: `frontend/src/test/performance-charts.test.jsx`

**Interfaces:**
- Produces: a legenda base passa a ser `{ type: 'scroll', top: 0, ... }`, numa linha só com setas de paginação. O `grid.top` base comporta essa linha. Um `option.legend` do gráfico continua sobrescrevendo, e `legend: false` continua escondendo.

- [ ] **Step 1: Escrever o teste** (usando o mock de ECharts que o arquivo já usa para ler a `option`)

```jsx
it('legenda base é de rolagem para não quebrar sobre o gráfico', () => {
  // renderiza ShadowChart com 6 séries -> option.legend.type === 'scroll'
});
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd frontend && npm test -- --run src/test/performance-charts.test.jsx`
Expected: FAIL (`legend.type` undefined)

- [ ] **Step 3: Implementar e conferir visualmente** no `npm run dev` com 6 séries no shadow (sem texto sobreposto)

- [ ] **Step 4: Rodar e ver passar**

Run: `cd frontend && npm test -- --run`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/portfolio/charts/BaseChart.jsx frontend/src/test/performance-charts.test.jsx
git commit -m "🐛 fix(spa): legenda dos gráficos em uma linha com rolagem"
```

---

### Task 3b: TIR do período quando há menos de 12 meses (A6)

**Files:**
- Modify: `backend/engine/irr.py`, `backend/engine/compute.py`, `backend/apps/api/schemas.py`, `backend/apps/api/routers/portfolio.py` (`summary`, `irr_evolution`), `frontend/src/components/portfolio/SummaryCards.jsx`, `frontend/src/components/portfolio/charts/IrrEvolutionChart.jsx`
- Test: `backend/engine/tests/test_irr.py`, `backend/engine/tests/test_compute.py`, `backend/apps/api/tests/test_portfolio_misc.py`, `frontend/src/test/summary-cards-chart.test.jsx`

**Interfaces:**
- Produces:
  - `irr.period_return(annual_rate: float, start, end) -> float`: `(1 + annual_rate) ** (dias / 365) − 1`.
  - `compute.IRR_ANNUAL_AFTER = pd.DateOffset(months=12)`.
  - `compute.irr_period_series(irr: pd.Series, opening: pd.Timestamp) -> pd.Series`: cada ponto vira `period_return(valor, opening, data)`.
  - `compute_irr_evolution` **não muda** (paridade de `irr`).
  - `/summary`, campos aditivos:
    - `xirr_period: float | None`, a TIR acumulada da abertura até o fim;
    - `xirr_is_annual: bool`, verdadeiro se `fim >= abertura + 12 meses`.
    - `xirr` continua a.a.
    - A abertura é `portfolio_df["date"].iloc[0]`. O cálculo fica no engine (`compute_summary` da Task 5; se a Task 5 ainda não existir, no próprio `compute.py`), nunca no router.
  - `/irr-evolution`, aditivo em `data`:
    - `period: [série "TIR no período"]`;
    - `annual_from: "AAAA-MM-DD" | null`, a 1ª data `>= abertura + 12 meses`.
    - `series` continua igual.
  - SPA, card:
    - `xirr_is_annual === false` → rótulo `TIR no período` e valor `xirr_period`;
    - senão, `TIR (XIRR a.a.)` e `xirr`, como hoje;
    - campo ausente (resposta antiga) → comportamento atual.
  - SPA, gráfico:
    - título `Evolução Histórica da TIR`;
    - uma linha só: pontos antes de `annual_from` vêm de `period`, os demais de `series`;
    - tooltip com sufixo `(no período)` ou `(a.a.)`;
    - `markLine` vertical em `annual_from` com o rótulo `a.a. a partir daqui`;
    - sem `annual_from`, a linha toda vem de `period`.

- [ ] **Step 1: Escrever os testes**

```python
def test_period_return_one_month_loss():
    # -1000 em 2024-01-31, +970 em 2024-02-29: XIRR a.a. ≈ -32 %; TIR do período = -3 %
    r = calculate_xirr([-1000, 970], ["2024-01-31", "2024-02-29"])
    assert period_return(r, "2024-01-31", "2024-02-29") == pytest.approx(-0.03, abs=1e-9)

def test_period_return_one_year_equals_annual():
    assert period_return(0.10, "2023-01-01", "2024-01-01") == pytest.approx(0.10)

def test_summary_short_window_exposes_period_irr(...):
    # janela de 3 meses -> body["data"]["xirr_is_annual"] is False e "xirr_period" presente; "xirr" mantido
def test_irr_evolution_annual_from(...):
    # 14 meses -> data["annual_from"] == data da abertura + 12 meses (1º fechamento >=) e len(data["period"][0]["points"]) == len(series)
```

```jsx
it('card mostra TIR no período quando xirr_is_annual é false', ...);
it('card mantém TIR a.a. quando o campo não vem', ...);
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest engine apps/api -q && cd ../frontend && npm test -- --run src/test/summary-cards-chart.test.jsx`
Expected: FAIL (`ImportError: period_return`; chaves ausentes; rótulo antigo)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar suítes e regressão**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q && ../../backend/.venv/bin/python -m parity.regress --baseline <linha de base da Task 2>`
Expected: tudo `=` (`irr` e `kpis` intocados; os campos novos não entram na paridade). Frontend: `cd frontend && npm test -- --run` PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/engine/irr.py backend/engine/compute.py backend/engine/tests backend/apps/api frontend/src/components/portfolio/SummaryCards.jsx frontend/src/components/portfolio/charts/IrrEvolutionChart.jsx frontend/src/test/summary-cards-chart.test.jsx
git commit -m "✨ feat(engine): TIR do período sem anualizar quando há menos de 12 meses"
```

---

## Parte B — Backlog SPA/API

### Task 4: Cache DLP e `single_flight` robustos (`apps/market`)

Itens do backlog: lock do `single_flight` sem dono; geração do cache DLP despejável pelo LRU; `DLPAuthError` fora do cache de falha (9 chamadas serializadas); filtro DLP ignorado se a coluna faltar.

**Files:**
- Modify: `backend/apps/market/cache.py` (`single_flight`), `backend/apps/market/services/sources/dlp.py` (`_generation`, `_get_or_mark`, `_filter_historico`, `buscar_historico`, `buscar_resumo_carteira`)
- Test: `backend/apps/market/tests/test_cache.py`, `backend/apps/market/tests/test_dlp.py` (ou o arquivo de testes DLP existente)

**Interfaces:**
- Produces:
  - `single_flight`: o valor do lock é um `uuid4().hex` próprio. Ao sair, só apaga a chave se `cache.get(lock_key)` for o próprio valor.
  - `dlp._generation(token)`: geração ausente → `cache.add(gen_key, time.time_ns(), None)` e relê (mesmo padrão de `cache.series_generation`, ADR 0019). `invalidate_user_cache` faz `incr`; com a chave ausente, `add` com `time.time_ns()`.
  - Falha de autenticação memorizada: `DLPAuthError` grava `<key>:auth` por `ERR_TTL`. `_failed_recently` passa a devolver `"auth" | "unavailable" | None`. Com `"auth"`, a chamada levanta `DLPAuthError` sem ir à DLP; com `"unavailable"`, `DLPUnavailableError` como hoje. Uma nova geração (refresh / novo token) libera a nova tentativa.
  - `_filter_historico`: filtro ativo sobre coluna ausente → `None` (nenhuma linha casa) + `logger.warning`, em vez de devolver o histórico sem filtro.

- [ ] **Step 1: Escrever os testes**

```python
def test_single_flight_does_not_release_foreign_lock(locmem_cache):
    with mc.single_flight("k", timeout=0.05):
        locmem_cache.set("k:lock", "outro-dono", 60)          # lock expirou e outro pegou
    assert locmem_cache.get("k:lock") == "outro-dono"

def test_dlp_generation_after_eviction_does_not_reuse_old(locmem_cache, monkeypatch):
    # time_ns=1_000 -> gen inicial; delete da chave; time_ns=2_000 -> nova gen == 2_000

def test_auth_error_is_memoized_per_generation(locmem_cache, monkeypatch):
    # _get levanta DLPAuthError e conta chamadas; 3x buscar_historico -> 3x DLPAuthError e 1 chamada;
    # invalidate_user_cache(token) -> próxima busca chama _get de novo (2 chamadas)

def test_filter_on_missing_column_matches_nothing():
    df = pd.DataFrame({"date": ["2024-01-31"], "ativo": ["X"]})
    assert dlp._filter_historico(df, classe="Ações") is None
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/market -q`
Expected: FAIL nos 4 testes novos

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/apps/market/cache.py backend/apps/market/services/sources/dlp.py backend/apps/market/tests
git commit -m "🐛 fix(market): lock com dono, geração DLP estável e token rejeitado em cache"
```

---

### Task 5: Contrato da API — 422 no envelope, token sem corrida, `start` relativo ao `end`, `/summary` fino

**Files:**
- Modify: `backend/config/api.py`, `backend/apps/api/routers/dlp.py` (`save_token`), `backend/apps/api/analysis.py` (`validate_request`), `backend/engine/compute.py` (nova `compute_summary`), `backend/apps/api/routers/portfolio.py` (`summary`)
- Test: `backend/apps/api/tests/test_errors.py` (criar se não existir), `backend/apps/api/tests/test_dlp_token.py` (ou o existente de token), `backend/apps/api/tests/test_analysis.py`, `backend/engine/tests/test_compute.py`, `backend/apps/api/tests/test_portfolio_misc.py`

**Interfaces:**
- Produces:
  - Handler para `ninja.errors.ValidationError`: `422 {"code": "invalid_request", "message": "Requisição inválida.", "details": exc.errors}`.
  - `save_token`: grava com `UserToken.objects.update_or_create(user=..., defaults=...)` dentro de `transaction.atomic()`. Num `IntegrityError` concorrente, relê e atualiza uma vez. Invalida o cache do token antigo como hoje.
  - `validate_request`: só `end` informado → `start = _one_year_ago(end)`; nenhum dos dois → como hoje.
  - `compute.compute_summary(df_combined, portfolio_df, rf_series) -> dict[str, float | None]` com as chaves numéricas atuais do `/summary` (`patrimonio_inicial`, `aportes_periodo`, `patrimonio`, `ganho_periodo`, `twr_total`, `cagr`, `volatilidade`, `sharpe`, `max_drawdown`, `xirr`). O router só aplica `clean` e acrescenta `data_inicio`/`data_fim`. Resposta JSON idêntica.

- [ ] **Step 1: Escrever os testes**

```python
def test_schema_error_uses_api_error_envelope(auth_client):
    r = auth_client.post("/api/v1/portfolio/summary", data={"start": "não-data"}, content_type="application/json")
    assert r.status_code == 422 and r.json()["code"] == "invalid_request" and "details" in r.json()

def test_save_token_survives_concurrent_insert(auth_client, monkeypatch):
    # cria UserToken do usuário e faz o .filter(...).first() do router devolver None (simula a corrida)
    # -> 200 e UserToken.objects.filter(user=...).count() == 1 com o token novo

def test_only_end_defaults_start_one_year_before_end():
    v = validate_request(AnalysisRequest(end=dt.date(2020, 6, 30)), today=dt.date(2026, 10, 29))
    assert v.start == dt.date(2019, 6, 30)

def test_compute_summary_keys():
    # fixtures de test_compute -> set(out) == {as 10 chaves}
```

O teste de contrato existente do `/summary` (`test_portfolio_misc.py`) tem que continuar verde sem mudança.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/api engine/tests/test_compute.py -q`
Expected: FAIL (`422` com corpo do Ninja; `IntegrityError`; `start == 2025-10-29`; `AttributeError: compute_summary`)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/config/api.py backend/apps/api backend/engine/compute.py backend/engine/tests/test_compute.py
git commit -m "🐛 fix(api): 422 no envelope ApiError, token sem corrida e /summary sem cálculo no router"
```

---

### Task 6: Rate limit de login

**Files:**
- Create: `backend/apps/accounts/ratelimit.py`
- Modify: `backend/apps/accounts/api.py` (`login_view`)
- Test: `backend/apps/accounts/tests/test_ratelimit.py`, `backend/apps/accounts/tests/test_api.py` (ou o de login existente)

**Interfaces:**
- Produces (`ratelimit.py`):
  - `MAX_FAILURES = 5`, `WINDOW_SECONDS = 900`
  - `key_for(username: str, ip: str) -> str` (username normalizado em minúsculas e com `strip`, sem expor senha)
  - `is_blocked(key) -> bool`, `register_failure(key) -> None` (`cache.add(key, 0, WINDOW_SECONDS)` + `incr`), `reset(key) -> None`
  - Toda falha de cache é logada e tratada como "não bloqueado".
  - IP: `request.META["REMOTE_ADDR"]`. O nginx do LAB já repassa? Conferir; se o backend só vê o IP do nginx, usar `X-Real-IP` **apenas** quando `REMOTE_ADDR` for o do proxy configurado em settings.
- `login_view`: bloqueado → `ApiError(429, "too_many_attempts", "Muitas tentativas de login. Tente novamente em alguns minutos.")` **antes** de `authenticate`. Falha → `register_failure`; sucesso → `reset`.

- [ ] **Step 1: Escrever os testes**

```python
def test_sixth_attempt_is_blocked(client, locmem_cache):   # 5x senha errada -> 401; 6ª (mesmo com senha certa) -> 429 code too_many_attempts
def test_other_user_or_ip_not_blocked(client, locmem_cache):  # após 5 falhas de "a"@ip1: "b"@ip1 e "a"@ip2 -> 401/200 normais
def test_success_resets_counter(client, locmem_cache):    # 4 falhas, 1 sucesso, 4 falhas -> ainda não bloqueado
def test_cache_down_does_not_block(client, monkeypatch):  # cache.add/incr/get levantam -> login certo 200
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/accounts -q`
Expected: FAIL (`ModuleNotFoundError: apps.accounts.ratelimit`; 6ª tentativa → 401)

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/apps/accounts
git commit -m "🔒️ feat(auth): rate limit de login por usuário e IP"
```

---

### Task 7: `MAX_MONTHS` aceita exatamente 240 meses

**Files:**
- Modify: `backend/apps/migration_ir/planner.py` (`_schedule_stocks`)
- Test: `backend/apps/migration_ir/tests/test_planner.py`

**Interfaces:**
- Produces: o limite é checado só quando um lote é adicionado a um mês novo (`len(months) == MAX_MONTHS` ao abrir o mês 241). O `close()` de um mês vazio no fim não conta.

- [ ] **Step 1: Escrever os testes**

```python
def test_plan_with_exactly_max_months_is_accepted():   # entrada cujo cronograma dá 240 meses -> sem PlanError, len == 240
def test_plan_with_max_months_plus_one_is_rejected():  # 241 meses -> PlanError com a mensagem atual
```

Cobrir o ramo "unidade acima do teto" (o lote de um papel não cabe no teto e ocupa o mês sozinho).

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/migration_ir -q`
Expected: FAIL no caso de 240 meses

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest apps/migration_ir -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/apps/migration_ir/planner.py backend/apps/migration_ir/tests/test_planner.py
git commit -m "🐛 fix(migration_ir): plano de exatamente 240 meses é aceito"
```

---

### Task 8: PDF do plano sem bloco cortado entre páginas

**Files:**
- Modify: `backend/apps/migration_ir/pdf.py`
- Test: `backend/apps/migration_ir/tests/test_pdf.py`

**Interfaces:**
- Produces:
  - `_month_height(m: dict) -> float`: altura em mm do bloco do mês (título + cabeçalhos + `max(len(vendas), len(compras), 1)` linhas, com as alturas de linha usadas em `_month`).
  - `_month` faz `if pdf.will_page_break(_month_height(m)): pdf.add_page()` antes de desenhar. Bloco maior que uma página inteira: desenha sem `add_page` em laço.
  - `_initial_balances`: a mesma regra por linha da tabela, e o cabeçalho é repetido na página nova.

- [ ] **Step 1: Escrever o teste**

```python
def test_month_blocks_never_split(monkeypatch):
    # plano sintético com 30 meses de 3–6 linhas; envolve pdf._month para registrar
    # (page_no antes, page_no depois) -> todos iguais; e o PDF tem > 1 página
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd backend && .venv/bin/pytest apps/migration_ir/tests/test_pdf.py -q`
Expected: FAIL (algum mês começa numa página e termina na seguinte)

- [ ] **Step 3: Implementar e abrir o PDF gerado** (salvar em `reports/fase7c/`, fora do git) para conferir visualmente

- [ ] **Step 4: Rodar e ver passar**

Run: `cd backend && .venv/bin/pytest apps/migration_ir -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/apps/migration_ir/pdf.py backend/apps/migration_ir/tests/test_pdf.py
git commit -m "🐛 fix(migration_ir): PDF não corta bloco de mês nem linha entre páginas"
```

---

### Task 9: Fluxo do token DLP na SPA como estado único de página

Itens do backlog: token rejeitado/ilegível → 8 `ErrorState` iguais e modal reabrindo; `PeriodNotice` fora de contexto; token não salvo persistindo no modal; mensagem de "primeira carga DLP".

**Files:**
- Create: `frontend/src/components/dlp/DlpTokenErrorState.jsx`
- Modify: `frontend/src/api/client.js` (interceptor), `frontend/src/pages/PortfolioPage.jsx`, `frontend/src/pages/MigrationPage.jsx`, `frontend/src/components/dlp/DlpTokenModal.jsx`, `frontend/src/components/portfolio/SummaryCards.jsx`
- Test: `frontend/src/test/token-gating.test.jsx`, `frontend/src/test/header-dlp.test.jsx`

**Interfaces:**
- Produces:
  - `export const DLP_TOKEN_ERROR_CODES = ['dlp_token_rejected', 'dlp_token_missing']` e `dlpTokenErrorCode(error) -> string | null` (em `DlpTokenErrorState.jsx`).
  - `DlpTokenErrorState({ code })`: texto por código ("Token DLP rejeitado…" / "Não foi possível ler seu token DLP…") e botão "Atualizar token", que dispara `open-dlp-modal` com o `reason` correspondente.
  - O interceptor de `client.js` **deixa de** disparar `open-dlp-modal` em 409/422. O modal só abre por ação do usuário.
  - `PortfolioPage`: `dlpTokenErrorCode(summary.error)` → renderiza **só** `DlpTokenErrorState` abaixo dos filtros (sem cards, gráficos nem `PeriodNotice`). `PeriodNotice` só aparece com `tokenConfigured === true`. `MigrationPage`: a mesma regra sobre a query da carteira.
  - `DlpTokenModal`: ao fechar (botão, Esc ou sucesso), limpa `token`, `showToken` e `error`.
  - `SummaryCards` em carregamento sem dados: texto `Buscando sua carteira na DLP — a primeira carga pode levar até um minuto.`

- [ ] **Step 1: Escrever os testes**

```jsx
it('token rejeitado mostra um único estado e não abre o modal', async () => {
  // MSW/mocks: token-status configured:true; todas as rotas /portfolio/* -> 422 dlp_token_rejected
  // -> getAllByText(/Token DLP rejeitado/).length === 1; queryByRole('dialog') === null
});
it('PeriodNotice some sem token configurado', ...);          // meta em cache + configured:false -> sem "Período analisado"
it('modal descarta token digitado e não salvo ao fechar', ...); // digita, fecha, reabre -> campo vazio
it('mensagem de primeira carga durante o loading do summary', ...);
```

Ajustar os testes existentes que esperavam o modal abrindo pelo interceptor.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd frontend && npm test -- --run src/test/token-gating.test.jsx src/test/header-dlp.test.jsx`
Expected: FAIL

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd frontend && npm test -- --run && npm run build`
Expected: PASS e build OK

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/dlp frontend/src/api/client.js frontend/src/pages frontend/src/components/portfolio/SummaryCards.jsx frontend/src/test
git commit -m "🐛 fix(spa): token DLP inválido como estado único de página, sem modal reabrindo"
```

---

### Task 10: Filtros — `period` na URL, "Personalizado" após "Tudo", logout e `location.state.from`

**Files:**
- Modify: `frontend/src/context/PortfolioFilterContext.jsx`, `frontend/src/components/portfolio/FiltersBar.jsx`, `frontend/src/App.jsx`, `frontend/src/pages/LoginPage.jsx`
- Test: `frontend/src/test/filters-bar.test.jsx`, `frontend/src/test/app-routing.test.jsx`

**Interfaces:**
- Produces:
  - `parsePeriodParams(search: string) -> { period, start, end }` e `periodToParams({ period, start, end }) -> URLSearchParams`, exportados de `PortfolioFilterContext.jsx`. Mapeamento `1a↔1A`, `2a↔2A`, `5a↔5A`, `tudo↔Tudo`, `custom↔Personalizado`. `start`/`end` só com `custom` e só se casarem `^\d{4}-\d{2}-\d{2}$` e forem datas válidas. Qualquer coisa inválida → `Tudo`.
  - O provider lê o período inicial da URL (`useSearchParams`) e mantém a URL sincronizada com `replace: true`, sem empilhar histórico.
  - `FiltersBar`: com `start === ALL_TIME_START`, o campo de data inicial do "Personalizado" mostra vazio. O estado não muda até o usuário escolher uma data.
  - `App.jsx`: `<PortfolioFilterProvider key={user?.id ?? 'anon'}>`, para os filtros voltarem ao padrão após logout ou troca de usuário.
  - `LoginPage`: após login, `navigate(from.pathname + from.search, { replace: true })` com `from = location.state?.from`, e `'/'` se ausente. Ignorar `from` que aponte para `/login`.

- [ ] **Step 1: Escrever os testes**

```jsx
it('lê period=custom com datas da URL', ...);          // /?period=custom&start=2024-01-01&end=2024-12-31 -> inputs com essas datas
it('period inválido cai em Tudo', ...);                // /?period=xyz&start=banana -> preset "Tudo" ativo, sem erro
it('trocar preset atualiza a URL', ...);               // clica 1A -> location.search contém period=1a
it('Personalizado após Tudo não mostra 01/01/1900', ...);
it('filtros voltam ao padrão após logout e novo login', ...);
it('login volta para a rota de origem', ...);          // /migracao sem sessão -> login -> /migracao
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd frontend && npm test -- --run src/test/filters-bar.test.jsx src/test/app-routing.test.jsx`
Expected: FAIL

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd frontend && npm test -- --run && npm run build`
Expected: PASS e build OK

- [ ] **Step 5: Commit**

```bash
git add frontend/src/context frontend/src/components/portfolio/FiltersBar.jsx frontend/src/App.jsx frontend/src/pages/LoginPage.jsx frontend/src/test
git commit -m "✨ feat(spa): período na URL, filtros zerados no logout e retorno à rota de origem"
```

---

### Task 10b: Campo de data com ícone de calendário visível (A7)

**Files:**
- Create: `frontend/src/components/common/DateInput.jsx`
- Modify: `frontend/src/components/portfolio/FiltersBar.jsx` (os dois `<input type="date">`)
- Test: `frontend/src/test/filters-bar.test.jsx`

**Interfaces:**
- Produces:
  - `DateInput({ value, onChange, ariaLabel, min, max })`: um `<input type="date">` nativo. O formato de exibição segue o idioma do navegador (decisão do usuário). `value` e `onChange` continuam em ISO `AAAA-MM-DD`.
  - Estilo: `[color-scheme:dark]`, para o calendário aberto ficar no tema escuro; o indicador nativo fica escondido (`[&::-webkit-calendar-picker-indicator]:opacity-0`).
  - Ícone `CalendarDays` (lucide-react) dentro do campo, à direita, como `<button type="button" aria-label="Abrir calendário">`.
  - Clique no campo ou no ícone chama `inputRef.current.showPicker()` dentro de `try/catch`; navegador sem suporte ou bloqueio não quebram nada.
  - Digitar no campo continua funcionando.

- [ ] **Step 1: Escrever os testes**

```jsx
it('campo de data mostra botão de calendário', ...);      // getAllByRole('button', { name: 'Abrir calendário' }).length === 2 no Personalizado
it('clicar no ícone chama showPicker', ...);             // HTMLInputElement.prototype.showPicker = vi.fn() -> chamado 1x
it('showPicker ausente não quebra', ...);                // delete showPicker -> clique sem erro
it('mudar a data mantém ISO no filtro', ...);            // fireEvent.change(..., '2024-12-31') -> setCustomDates recebe '2024-12-31'
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd frontend && npm test -- --run src/test/filters-bar.test.jsx`
Expected: FAIL (botão "Abrir calendário" ausente)

- [ ] **Step 3: Implementar e conferir no navegador** (`npm run dev`): ícone visível nos dois campos; clicar no texto ou no ícone abre o calendário escuro

- [ ] **Step 4: Rodar e ver passar**

Run: `cd frontend && npm test -- --run && npm run build`
Expected: PASS e build OK

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/common/DateInput.jsx frontend/src/components/portfolio/FiltersBar.jsx frontend/src/test/filters-bar.test.jsx
git commit -m "💄 feat(spa): campo de data com ícone de calendário e abertura por clique"
```

---

### Task 11: Acessibilidade do `Modal`, download do PDF e código morto

**Files:**
- Modify: `frontend/src/components/common/Modal.jsx`, `frontend/src/components/migration/ExportPdfButton.jsx`, `frontend/src/api/migration.js` (`downloadPdf`), os componentes com `data?.data?.x || data?.x` (`charts/*.jsx`, `tables/YearlyTable.jsx`, `SummaryCards.jsx`)
- Test: `frontend/src/test/modal-a11y.test.jsx` (criar), `frontend/src/test/migration-ir.test.jsx`, os testes de gráficos existentes

**Interfaces:**
- Produces:
  - `Modal`:
    - `aria-labelledby` apontando para o `id` do `<h2>` (`useId`);
    - foco inicial no 1º elemento focável;
    - `Tab`/`Shift+Tab` presos dentro do diálogo;
    - foco devolvido ao elemento que abriu, ao fechar;
    - clique no fundo chama `onClose`.
  - `migrationApi.downloadPdf(payload) -> { blob, filename }`. O `filename` vem do `Content-Disposition` (`filename*=` ou `filename=`), com fallback `plano_migracao_ir_<AAAAMMDD>.pdf`. O backend expõe o header (conferir; se faltar, adicionar no `api.py` da migração).
  - `ExportPdfButton`: usa esse `filename` e adia `revokeObjectURL` com `setTimeout(..., 0)` após o `click`.
  - Código morto: cada componente lê **um** formato. É o que `portfolioApi` devolve hoje; conferir em `api/portfolio.js`. O ramo alternativo e o shorthand do `SummaryCards` saem.

- [ ] **Step 1: Escrever os testes**

```jsx
it('Modal prende o foco e devolve ao fechar', ...);
it('Modal é rotulado pelo título', ...);               // getByRole('dialog', { name: 'Título' }) via aria-labelledby
it('download usa o nome do Content-Disposition', ...); // header 'attachment; filename="x.pdf"' -> link.download === 'x.pdf'
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cd frontend && npm test -- --run src/test/modal-a11y.test.jsx src/test/migration-ir.test.jsx`
Expected: FAIL

- [ ] **Step 3: Implementar**

- [ ] **Step 4: Rodar e ver passar**

Run: `cd frontend && npm test -- --run && npm run build`
Expected: PASS e build OK

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components frontend/src/api/migration.js frontend/src/test backend/apps/migration_ir/api.py
git commit -m "♿️ fix(spa): modal acessível, nome do PDF pelo servidor e remoção de código morto"
```

---

### Task 12: ADR, documentação, rebuild e ponto de espera

**Itens do backlog fechados sem código (justificar no doc mestre §2.3):**
- busca de ticker sem debounce: **já resolvido** (`BenchmarkSelector` tem `debouncedQuery`);
- `catalog.defaults` não usado pela SPA: **adiado**. Remover quebraria o contrato v1, e usar exige decidir se o padrão da SPA (`CDI`, `BOVA11`) passa a vir do servidor. Fica para quando houver preferência por usuário;
- bundle de 1,4 MB: desempenho, vai para a **7d** ou fica adiado (spec §4);
- `MAX_MONTHS` depois de `resolve_target_prices`: mantido como aceito ("anotar se a busca ficar cara").

**Files:**
- Create: `docs/adr/0020-backlog-spa-api-fase-7c.md`
- Modify:
  - `docs/adr/0019-integridade-dos-dados-de-mercado.md`: seção "Revisão pós-teste", com lacunas nas bordas, cauda parada, `SeriesGap.end` da cauda = fim da janela e resultado do diagnóstico de `bcb:12469`;
  - `docs/adr/0018-metodologia-de-retorno-dlp.md`: seção "TIR exibida", com a TIR do período abaixo de 12 meses (Task 3b), a convenção ANBIMA e os campos aditivos `xirr_period`, `xirr_is_annual` e `annual_from`;
  - `docs/adr/README.md`;
  - `docs/consolidacao_arquitetura_e_migracao.md`: §2.0, §2.2, §2.3 (riscar itens e justificar adiamentos), §5 Fase 7c, §9;
  - `docs/README.md`;
  - `backend/README.md` (contagem de testes);
  - `backend/CLAUDE.md` (linha de base nova da Task 2);
  - `docs/superpowers/plans/2026-10-28-fase-7b-mercado.md` (nota de que o teste do usuário gerou a Parte A deste plano).

- [ ] **Step 1: Escrever o ADR 0020**
  - **Contexto:** backlog §2.3.
  - **Decisões:**
    - rate limit 5/15 min por (usuário, IP), `429 too_many_attempts`;
    - `422 invalid_request` para erro de schema;
    - `period` na URL com `1a|2a|5a|tudo|custom`;
    - token inválido como estado de página e modal só por ação do usuário;
    - lock com dono;
    - geração DLP iniciada em `time_ns`;
    - falha de auth em cache por geração;
    - filtro sobre coluna ausente = nenhuma linha;
    - PDF sem quebra de bloco.
  - **Consequências:** `meta`/contrato só aditivos.

- [ ] **Step 2: Checklist de `docs/README.md` e verificação completa**

Run: `cd backend && .venv/bin/pytest -q && .venv/bin/python manage.py check && .venv/bin/pytest -m integration engine/tests/test_oracle_dlp.py -q && cd ../scripts/parity && ../../backend/.venv/bin/pytest -q && cd ../../frontend && npm test -- --run && npm run build`
Expected: tudo verde.

- [ ] **Step 3: Commit**

```bash
git add docs/adr/0020-backlog-spa-api-fase-7c.md docs/adr/0019-integridade-dos-dados-de-mercado.md docs/adr/0018-metodologia-de-retorno-dlp.md docs/adr/README.md docs/consolidacao_arquitetura_e_migracao.md docs/README.md docs/superpowers/plans/2026-10-28-fase-7b-mercado.md backend/README.md backend/CLAUDE.md
git commit -m "📝 docs: ADR 0020 (backlog SPA/API) e fechamento da Fase 7c"
```

- [ ] **Step 4: Rebuild e PARAR para o usuário testar** (spec §5.6, 2º ponto de espera)

Run: `docker compose up -d --build && docker compose ps && docker compose exec -T backend python manage.py market_gaps`
Expected: `backend` e `web` healthy; o diagnóstico mostra as séries "defasada desde" (se houver).

Mensagem ao usuário:
- repetir o período 01/01/2024–31/12/2024 com IMA-S e LFTS11.SA: devem aparecer **três** avisos, com IMA-S desde 22/05/2023 e LFTS11 de junho e de 26/09/2024 a 24/02/2025; IMA-S deve abrir na mesma base das outras séries e sem "—" na tabela anual, e não deve haver coluna 2023;
- num período de menos de 12 meses, o card deve mostrar "TIR no período" e o gráfico da TIR deve começar sem o −32 %; num período mais longo, deve aparecer a marca "a.a. a partir daqui";
- o campo de data do "Personalizado" deve ter o ícone de calendário e abrir com um clique em qualquer parte;
- testar token inválido, logout/login, link com `?period=` e o PDF do planejador.

**Não iniciar a 7d sem o retorno do usuário.**
