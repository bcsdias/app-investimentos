# ADR 0019 — Integridade dos dados de mercado, invalidação de cache e tratamento de lacunas

- **Status:** Aceita
- **Data:** 2026-10-29
- **Fase:** 7b
- **Spec:** [`../superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md`](../superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md) (§3.3, §5.3.1, §5.4, §5.6, §5.8)
- **Plano:** [`../superpowers/plans/2026-10-28-fase-7b-mercado.md`](../superpowers/plans/2026-10-28-fase-7b-mercado.md)

## Contexto

Os dados de mercado consumidos pelo engine analítico (`apps/market` e `engine/dataset.py`) apresentavam fragilidades estruturais de integridade e consistência:

1. **Cache sem invalidação:** As séries eram cacheadas no Redis com chaves estáticas `market:series:<nome>:<start>:<end>` e TTL de 24 horas. Novas ingestões gravavam pontos no banco e aqueciam o cache para o dia corrente, mas chaves existentes para janelas históricas continuavam servindo dados defasados. Séries derivadas (ex.: `S&P 500 BRL` derivado de `^GSPC` × `PTAX`) e sintéticas (`IPCA + 6%`, `CDI + 2%`) permaneciam desatualizadas caso qualquer uma das séries base recebesse novos pontos ou correções.
2. **Ingestão incremental cega a lacunas internas:** O comando `ingest_market_data` iniciava a coleta a partir de `last_date + 1 dia`, ignorando buracos ou falhas temporárias ocorridas no meio do histórico das séries.
3. **Buracos reais nas fontes de dados:** Séries oficiais possuem períodos sem negociação ou descontinuidade de publicação nas APIs de origem (ex.: `bcb:12469` / IMA-S com buraco de 275 dias em 2023–2024 e `yf:LFTS11.SA` com buraco de 151 dias em 2024–2025). A ausência desses dados não pode ser mascarada com interpolação inventada nem tratada como falha fatal de ingestão.
4. **Defasagem na abertura dos benchmarks:** Pendência identificada na Fase 7a ([ADR 0018](0018-metodologia-de-retorno-dlp.md)): a carteira abre num fechamento mensal (ex.: 31/12), enquanto benchmarks diários começam no primeiro dia útil subsequente (ex.: 02/01) e o IPCA no dia 01/01. Sem propagar o último valor conhecido até a abertura da carteira, o primeiro ponto do benchmark sofria distorção de base (`NaN` ou rebase incorreto).

## Decisão

### 1. Invalidação de Cache por Geração Global no Redis
- Adotada uma geração global armazenada na chave `market:series:gen` (Redis).
- Chaves de séries passam a incluir a geração ativa: `market:series:g{gen}:{name}:{start}:{end}`.
- O valor inicial é gerado a partir de `time.time_ns()` quando ausente no Redis, impedindo colisões mesmo após reinicialização ou despejo por LRU.
- O único ponto de escrita de pontos de mercado, `repository.upsert_points`, incrementa atomicamente a geração (`bump_series_generation`) sempre que ao menos uma linha válida é persistida.
- Optou-se por geração global em vez de por série porque séries derivadas e sintéticas dependem de múltiplas séries base e a ingestão é diária, simplificando radicalmente o grafo de dependências sem complexidade de rastreamento.
- Tratamento resiliente: falhas no Redis são logadas e engolidas, preservando o cache estritamente como otimização e devolvendo geração 0 em caso de indisponibilidade.

### 2. Detecção Pura de Lacunas por Cadência (`apps/market/gaps.py`)
- Módulo puro (sem dependência de Django ou I/O de banco) para análise de lacunas temporais.
- Tolerâncias baseadas na frequência da série:
  - Séries diárias (`daily`): intervalo consecutivo superior a **10 dias corridos** (evitando falsos positivos em feriados prolongados como Natal/Ano Novo e Carnaval).
  - Séries mensais (`monthly`): intervalo consecutivo superior a **45 dias corridos**.
- Inferência automática de cadência via mediana dos passos entre datas ($\ge 20$ dias $\to$ `monthly`, senão `daily`).
- Séries vazias ou com apenas 1 ponto retornam lista vazia com segurança.

### 3. Re-busca Automática na Ingestão e Diagnóstico (`market_gaps`)
- Comando `python manage.py market_gaps [--only ...]` para diagnóstico e auditoria de lacunas em todas as séries cadastradas e sob demanda.
- O pipeline de `ingest_market_data` executa `_refill_gaps` para cada série após a ingestão incremental (inclusive quando já atualizada), consultando a fonte no intervalo exato de cada lacuna interna (`start + 1` a `end - 1`).
- Lacunas cujas fontes não possuem dados são logadas como `lacuna sem dado na fonte` e mantidas sem abortar a rotina nem incrementar o contador de falhas da ingestão (código de saída permanece limpo).

### 4. Transparência na API e na SPA sem Interpolação Silenciosa
- Convenção de mercado (§5.3.1): `ffill` (repetição do último valor conhecido) para cálculo contínuo das curvas, **sempre acompanhado** de aviso explícito ao usuário. Interpolação inventada é expressamente proibida.
- Contrato aditivo na API (`meta.gaps`):
  ```json
  "meta": {
    "start": "2024-01-01",
    "end": "2024-06-30",
    "missing": [],
    "gaps": [
      {
        "series": "IMA-S",
        "start": "2023-05-22",
        "end": "2024-02-21"
      }
    ]
  }
  ```
- Componente `PeriodNotice` na SPA exibe cada lacuna identificada com alerta âmbar: `Lacuna nos dados de <series> de <dd/mm/aaaa> a <dd/mm/aaaa> (último valor repetido)`.

### 5. Abertura dos Benchmarks no Último Valor Conhecido (`engine.dataset`)
- Constante `BENCH_LOOKBACK_DAYS = 40`.
- `build_dataset` busca os benchmarks e séries base sintéticas a partir de `start - 40 dias`.
- As séries são concatenadas cronologicamente e sofrem `ffill()` em toda a janela expandida.
- O corte para `[crop_start, crop_end]` só ocorre **após** o forward-fill, garantindo que o benchmark possua o último valor conhecido na data exata de abertura da carteira (evitando NaNs e rebases espúrios). A taxa livre de risco `rf` (CDI) continua restrita ao período efetivo `[start, end]`.

## Evidência e Validação

### 1. Diagnóstico de Lacunas no LAB
Execução do diagnóstico no ambiente real antes e depois da re-busca:
- `bcb:12469`: lacuna detectada de 275 dias (`2023-05-22` a `2024-02-21`). A re-busca confirmou ausência de dados na fonte oficial (BCB SGS 404), mantendo o aviso aditivo.
- `yf:LFTS11.SA`: 4 lacunas detectadas em 2024–2025 (incluindo o buraco de 151 dias de `2024-09-26` a `2025-02-24`). Yahoo Finance retornou `YFPricesMissingError`, logado e mantido como buraco de fonte legítimo.
- `b3:SMLL`: lacunas históricas (2006 a 2016) identificadas e catalogadas.

### 2. Regressão de Paridade Estrita (Task 4)
Comparação executada contra a linha de base congelada `reports/parity/20261028-180012-fase7-abertura/baseline`:

| caso | twr | drawdown | rolling_vol | rolling_sharpe | risk_return | irr | shadow | yearly | kpis | portfolio |
|---|---|---|---|---|---|---|---|---|---|---|
| w1-completo | IPCA + 6%, CDI | IPCA + 6% | CDI, IPCA + 6% | CDI, IPCA + 6% | Volatilidade, Retorno (CAGR), Sharpe | = | CDI, IPCA + 6% | 2016..2026, Total Acum. | = | = |
| w2-12m | IPCA + 6%, CDI | IPCA + 6% | CDI, IPCA + 6% | CDI, IPCA + 6% | Volatilidade, Retorno (CAGR), Sharpe | = | CDI, IPCA + 6% | 2025..2026, Total Acum. | = | = |
| w3-meio | CDI, IPCA + 6% | IPCA + 6% | CDI, IPCA + 6% | CDI, IPCA + 6% | Volatilidade, Retorno (CAGR), Sharpe | = | CDI, IPCA + 6% | 2021..2026, Total Acum. | = | = |
| w4-fim-passado | IPCA + 6%, CDI | IPCA + 6% | CDI, IPCA + 6% | CDI, IPCA + 6% | Volatilidade, Retorno (CAGR), Sharpe | = | CDI, IPCA + 6% | 2016..2024, Total Acum. | = | = |
| w5-inicio-mes | IPCA + 6%, CDI | IPCA + 6% | CDI, IPCA + 6% | CDI, IPCA + 6% | Volatilidade, Retorno (CAGR), Sharpe | = | CDI, IPCA + 6% | 2019..2026, Total Acum. | = | = |
| w1-completo-classe | IPCA + 6%, CDI | IPCA + 6% | CDI, IPCA + 6% | CDI, IPCA + 6% | Volatilidade, Retorno (CAGR), Sharpe | = | CDI, IPCA + 6% | 2016..2026, Total Acum. | = | = |

- **Resultados de paridade:**
  - `irr`: 100% idêntico (`=`) em todos os casos.
  - `kpis`: 100% idêntico (`=`) em todos os casos.
  - `portfolio`: 100% idêntico (`=`) em todos os casos.
  - Coluna e linha `Carteira` em todas as métricas: 100% idêntica (`=`) em todos os casos.
  - Apenas as colunas dos benchmarks (`CDI`, `IPCA + 6%`) mudaram conforme esperado devido ao valor conhecido na data de abertura.

### 3. Nova Linha de Base Congelada
- Gravada em: `reports/parity/20261029-122417-fase7b/baseline`
- Re-verificação contra a nova linha de base resultou em 100% `=` em todas as métricas e janelas.

## Consequências

- **Cache:** Qualquer escrita no repositório de séries invalida atomicamente todas as séries em cache via incremento da geração global. O aquecimento diário pós-ingestão mantém o desempenho otimizado para o usuário final.
- **Robustez:** Lacunas históricas ou buracos de fonte são comunicados com clareza e transparência no frontend, sem ocultação ou distorção de cálculos.
- **Abertura consistente:** Comparações entre a carteira e benchmarks agora utilizam valores reais conhecidos no dia exato de abertura da janela, eliminando desvios artificiais no início dos períodos.

