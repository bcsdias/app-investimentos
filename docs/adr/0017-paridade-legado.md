# ADR 0017 — Paridade com o legado em duas camadas e linha de base da Fase 7

- **Status:** Aceita
- **Data:** 2026-10-27
- **Fase:** 6
- **Spec:** [`../superpowers/specs/2026-10-27-fase-6-paridade-design.md`](../superpowers/specs/2026-10-27-fase-6-paridade-design.md)
- **Plano:** [`../superpowers/plans/2026-10-27-fase-6-paridade.md`](../superpowers/plans/2026-10-27-fase-6-paridade.md)

## Contexto

O app é financeiro: antes de apagar o Streamlit (`src/`) era preciso provar que o engine
novo (`backend/engine/`) e a API (`/api/v1/portfolio/*`) entregam os mesmos números que o
`FinancialReport` legado para a mesma carteira e o mesmo período. A Fase 7 vai mudar números
de propósito (recortes `end_date`/`gap`), então a Fase 6 também precisa congelar uma linha de
base que sobreviva à remoção do legado. O Streamlit não sobe mais (Supabase descartado,
[ADR 0016](0016-empacotamento-no-lab.md)), e a ingestão do LAB roda em paralelo (cron
06:00/19:00).

## Decisão

### Método: duas camadas

| | Camada 1 — engine | Camada 2 — ponta a ponta |
|---|---|---|
| Legado | `FinancialReport` sem UI, com as funções `buscar_*` de `market_data` trocadas por fakes no formato bruto de cada uma (caminho de *cache hit*), de modo que `processar_benchmarks` roda de verdade | `FinancialReport` ao vivo (DLP, yfinance, BCB, Selenium B3), Upstash desligado |
| Novo | `engine.dataset` + `engine.compute` com `repository.load_series` trocado pelo snapshot, `DummyCache`, `_ensure_fresh` no-op, `upsert_points` proibido | API real dos containers via HTTP (`PARITY_API_URL`, login por `PARITY_USER`/`PARITY_PASSWORD`) |
| Mercado | **um** `MarketSnapshot` lido do Postgres uma vez e entregue aos dois lados (imune à ingestão concorrente) | cada lado com sua fonte |
| Tolerância | `1e-6` relativo, piso absoluto `1e-9`, NaN só casa com NaN → `FALHA` | `1e-4` relativo → `INVESTIGAR` (não reprova) |

Desvio do [ADR 0012](0012-engine-copia-literal-fase-2-5.md) (`pct_change(fill_method=None)`):
uma diferença restrita a vol/Sharpe móveis, risco × retorno ou tabela anual que **some** ao
repetir o legado com `fill_method=None` vira `DESVIO CONHECIDO (ADR 0012)` — ainda reprova o
código de saída, para o usuário decidir. Não apareceu.

Token DLP e credenciais só por variável de ambiente; `summary.md` sem valores em R$ (só razões
relativas, contagens e datas); detalhes e baseline em `reports/parity/<timestamp>/`,
ignorado pelo git.

### Matriz

- Janelas calculadas a partir do histórico DLP: `w1-completo` (primeira → última data),
  `w2-12m` (última − 1 ano → última), `w3-meio` (data mediana → hoje, exercita o `gap`),
  mais `w1-completo-classe` (filtro de classe com mais datas, cobre o filtro da Fase 3c).
- Benchmarks: `CDI` (BCB taxa), `IPCA + 6%` (sintético), `S&P 500 BRL` (Yahoo × PTAX),
  `IFIX` (B3).
- Fora: Tesouro (`CATALOGO_TD` do legado vazio, catálogo novo sem TD) e carteiras compostas
  (perda aceita pelo usuário — decisão 4 da spec; reavaliar no futuro).

## Resultado (execução de 2026-10-27, aprovado pelo usuário)

- **Camada 1:** 4/4 casos com todas as séries OK (TWR, drawdown, vol e Sharpe móveis 252,
  risco × retorno, XIRR, shadow, tabela anual, KPIs do `/summary`, `portfolio_df`). Zero
  `FALHA`, zero `DESVIO`.
- **Camada 2:** carteira (TWR, TIR, Total Investido) igual entre legado e API nas datas comuns
  (≤ 1e-11). As 127 linhas `INVESTIGAR` vieram de mercado/infra, nenhuma de cálculo:
  - **(A) bug do cache de `apps/market`:** série sem nome (`to_brl`, `splice_proxy`) ia para o
    JSON `orient="table"` como coluna `0` e voltava toda NaN — `S&P 500 BRL`, `IMID BRL`,
    `Bitcoin BRL` e a família IMA-B sumiam a partir do 2º pedido. **Corrigido** (commit
    `fad1dc8`, autorizado como exceção à regra "não tocar `apps/market`").
  - **(B)** IFIX do legado congelado em 2025-11-18: o Selenium da B3 falha no host ARM e o
    legado cai nos CSVs antigos de `data/raw`. Esperado pela spec.
  - **(C)** Buraco de ~116 dias no CDI (`bcb:12`, 2026-04-09 → 2026-08-03): o backfill do CSV
    legado parou em 04-09 e a ingestão começou em 08-03; a ingestão incremental só olha
    `last_date` e não preenche buracos. **Re-ingerido** (`ingest_market_data --only bcb:12
    --since 2026-04-01`, +143 pontos).
  - **(D)** Redis do app com séries gravadas durante o catch-up da ingestão (TTL 24 h, sem
    invalidação quando a ingestão grava) — datas a mais no legado e CDI/IPCA + 6% defasados.
- Teste `integration` da Fase 3c (`test_real_dlp_in_memory_filter_matches_server_filter`) verde
  com token real.
- A triagem também achou um bug **do harness** (o `mock.patch` importava `financial_report`
  já com o fake e "restaurava" o fake); corrigido antes do relatório aprovado.

## Linha de base da Fase 7

`reports/parity/20261027-182533/baseline/` (local, nunca versionado — contém a carteira):
`manifest.json` + `market/*.pkl` (snapshot) + `hist/<caso>.pkl` (histórico DLP congelado) +
`outputs/<caso>/<métrica>.pkl` (saídas do engine novo). Recarrega só com `backend/`:

```python
from parity.baseline import load_baseline
from parity.engine_side import METRICS, run_engine
from parity.compare import compare_frames

b = load_baseline(Path("reports/parity/20261027-182533/baseline"))
for case in b.cases:
    again = run_engine(case, b.snapshot, b.hists[case.id])
    for m in METRICS:
        compare_frames(m, b.outputs[case.id][m], again[m], rel_tol=0.0, fail_status="FALHA", abs_floor=0.0)
```

Verificado após a remoção de `src/`: 0 diferenças nos 4 casos. Na Fase 7, cada mudança
intencional de número deve aparecer só nas métricas e datas que ela promete afetar.
`scripts/parity/` mantém só os módulos sem `src` (`compare`, `cases`, `frozen`, `kpis`,
`engine_side`, `baseline`) e seus testes (`cd scripts/parity && ../../backend/.venv/bin/pytest -q`).

## Consequências

- `src/`, `redis.py`, `requirements.txt` da raiz, `.streamlit/`, scripts e workflow que
  importavam `src`, `tests/` da raiz (12 testes já portados em `backend/engine/tests/`) e
  resíduos foram removidos; libs só do legado desinstaladas do `backend/.venv`.
- Backlog aberto pela triagem: invalidar `market:series:*` quando a ingestão grava (D);
  ingestão detectar buracos e não só `last_date` (C); buracos de fonte em `bcb:12469` (2024,
  275 dias) e `yf:LFTS11.SA` (2025, 151 dias).
- O container do backend precisa ser reconstruído para levar a correção (A); as chaves
  `market:series:*` gravadas antes dela expiram em até 24 h.
