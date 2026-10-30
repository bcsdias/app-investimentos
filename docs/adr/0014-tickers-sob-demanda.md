# ADR 0014 — Tickers sob demanda, geração de cache DLP e single-flight

- **Status:** Aceita
- **Data:** 2026-10-26
- **Fase:** 3 (API da Carteira)

## Contexto

O app legado permitia comparar a carteira contra qualquer ativo de `data/ativos.csv` (~7.500 linhas), com conversão automática em Reais para ativos internacionais. Além disso, como a Fase 3 adota um endpoint por gráfico (8 chamadas paralelas no carregamento do dashboard), um cache frio da DLP ou de um ticker recém-solicitado dispararia ~8 requisições externas idênticas simultaneamente. Por fim, os mecanismos de lock e invalidação de cache precisavam funcionar tanto em produção (`django-redis`) quanto na suíte de testes (`LocMemCache`).

## Decisão

1. **Cópia empacotada do universo (D8):** `ativos.csv` vive em `backend/apps/market/data/ativos.csv` (configurável via `settings.MARKET_ASSETS_CSV`), tornando o backend autocontido para a imagem Docker da Fase 5.
2. **Identificador público = ticker Yahoo e variante BRL explícita (D7, D10):**
   - `BVMF:` → `<sigla>.SA`; `LON:` e `LSE:` → `<sigla>.L`; demais bolsas internacionais → `<sigla>`.
   - Em caso de colisão de chave Yahoo no CSV, a primeira linha em ordem de precedência por bolsa é preservada.
   - Ativos com `mercado == "INTL"` aceitam a variante explícita `"<yf_ticker> BRL"` (`ToBRL(yf_ticker)`). Nomes do catálogo fixo (`catalog.py`) têm precedência sobre `assets.py`.
3. **Fetch-on-read com histórico completo e fallback resiliente (D11):**
   - Ao pedir uma série `yf:<ticker>` do universo fora do catálogo com cache frio, `series.get_series` verifica `repository.last_date`:
     - `None` → baixa o histórico completo desde `catalog.EPOCH` (`1995-01-01`) via Yahoo e persiste em `MarketSeries`;
     - `last_date < hoje - 3 dias` → faz top-up incremental de `last_date + 1` até hoje.
   - Se o Yahoo falhar (`FetchError`), `series.get_series` loga warning e serve os pontos já gravados no banco (ou `None` se não houver nenhum, caindo em `meta.missing` sem derrubar a análise).
4. **Atualização contínua na ingestão (D9):** `ingest_market_data` descobre todas as chaves `yf:*` já persistidas em `MarketSeries` fora do catálogo (`_on_demand_defs()`) e as atualiza no lote diário com a mesma política de erro.
5. **Single-flight via `cache.add` e invalidação DLP por geração (D12, desvio consciente do spec §6):**
   - Em vez de `cache.lock` e `cache.delete_pattern` (exclusivos do `django-redis`), o single-flight usa `cache.add(key + ":lock", 1, timeout)` com polling curto + re-check do cache, e a invalidação DLP por usuário incrementa um contador de geração `market:dlp:gen:<token_hash>` embutido nas chaves `hist` e `resumo`.
   - O comportamento em produção no Redis é idêntico e toda a concorrência/invalidação é exercitada diretamente nos testes unitários com `LocMemCache`.

## Consequências

### Positivas
- Qualquer ativo do universo (~7.500 tickers) pode ser comparado sob demanda sem pré-popular o banco com milhares de séries não utilizadas.
- O primeiro carregamento paralelo de múltiplos gráficos faz apenas 1 chamada à DLP por usuário e 1 chamada ao Yahoo por ticker novo.
- Compatibilidade total entre `django-redis` e `LocMemCache`.

### Negativas / Mitigações
- Ativos de Londres (`.L`) na variante `"BRL"` continuam convertidos por PTAX USD/BRL para preservar a paridade exata com o legado na Fase 6 (registrado como débito pós-Fase 6).
