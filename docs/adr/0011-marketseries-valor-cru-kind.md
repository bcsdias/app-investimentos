# ADR 0011 — MarketSeries armazena valor cru por fonte + campo kind

- **Status:** Aceita
- **Data:** 2026-08-31
- **Fase:** 2 (Serviços de Mercado)

## Contexto

No sistema legado Streamlit, os dados de mercado eram cacheados de forma ad-hoc, por vezes já acumulados ou transformados, o que causava inconsistências de janela temporal e dificultava a auditoria das séries reais. Além disso, a falta de distinção clara entre séries de taxa percentual (ex.: CDI diário, IPCA mensal) e séries de nível de preço/índice (ex.: cotas de ETF, S&P 500, PU do Tesouro) forçava heurísticas espalhadas pelo código.

## Decisão

1. **`MarketSeries` armazena exclusivamente valores crus** conforme entregues pela fonte externa (taxa % diária para CDI/SELIC, cotação de fechamento para ações/ETFs/índices, PU base para Tesouro Direto, cotação de venda para PTAX).
2. Adicionar o campo `kind` (`"rate"` | `"level"`) ao modelo `core.MarketSeries`, com indexação e controle pelo catálogo (`catalog.py`).
3. **Toda derivação ocorre na leitura** (`transforms.py` e `series.py`): acumulação (`cumprod`), emendas de proxies (*splice* ANBIMA $\\to$ ETF), conversão cambial USD $\\to$ BRL e benchmarks sintéticos.
4. **PostgreSQL (`core.MarketSeries`) é a única fonte da verdade**; Redis é cache derivado temporário, indexado com chave contendo a janela de datas (`market:series:<nome>:<start>:<end>`).

## Consequências

### Positivas
- **Auditabilidade completa:** Os dados gravados no banco são idênticos aos divulgados pelas fontes originais (BCB, B3, Yahoo, Tesouro).
- **Flexibilidade evolutiva:** Melhorias em algoritmos de interpolação, composição de carteiras ou emendas de proxies não requerem reingestão de dados históricos no banco.
- **Eliminação de bugs de cache:** Chaves de cache com a janela temporal (`start:end`) eliminam distorções de recortes de período.

### Negativas / Mitigações
- **Cálculo na leitura:** Cache misses exigem recalcular o `cumprod` ou emenda. *Mitigação:* Redis com TTL de 24h para séries de mercado serve respostas instantâneas na grande maioria das requisições.
