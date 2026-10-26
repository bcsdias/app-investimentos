# ADR 0015 — Planejador de migração com isenção de IR (`apps/migration_ir/`)

- **Status:** Aceita
- **Data:** 2026-10-26
- **Fase:** 3b (Planejador de IR)
- **Spec:** [`../superpowers/specs/2026-10-26-fase-3b-planejador-ir-design.md`](../superpowers/specs/2026-10-26-fase-3b-planejador-ir-design.md)

## Contexto

O planejador legado (`src/ui/pages/3_migracao_ir.py`) tinha alvos fixos (`IMAB11`/`VWRA11`/`BITH11`), aplicava um teto genérico de R$ 19.500 a qualquer classe (FII/ETF/BDR pareciam isentos), inventava preço quando a cotação falhava e usava `price` (moeda original) em vez de `price_fx` (BRL) para ativos estrangeiros. A Fase 3b reescreve o cálculo como API (`/api/v1/migration/{wallet,plan,plan.pdf}`) em vez de portá-lo literalmente — não há paridade a preservar (a Fase 6 só compara as métricas da carteira).

## Decisão

1. **App próprio, núcleo puro:** `apps/migration_ir/` com `wallet.py`, `tax.py` e `planner.py` sem Django; `prices.py` é o único que toca `apps.market`; `api.py` (router fino, registrado em `config/api.py`) converte `PlanError` em `ApiError 422`. O `engine/` fica restrito à paridade ([ADR 0012](0012-engine-copia-literal-fase-2-5.md)).
2. **Alvos livres e associação manual:** 1–10 tickers com pesos somando 100; cada papel da carteira é `sell`, `count_as:<alvo>` ou `ignore` (padrão). O backend não classifica papéis (as listas de palavras-chave do legado somem).
3. **Regra fiscal por classe DLP:** teto mensal (`0 < cap ≤ 20.000`, default 20.000) só para `ACAO`, sobre o **valor de venda** e incluindo `already_sold_this_month` no mês 1. `FII` 20 %, `ETF`/`BDR`/`BDR ETF` 15 %, demais classes `not_estimated`. Tributadas e não estimadas vendem integralmente no mês 1. IR é **estimativa** (sem taxas, sem compensação), com disclaimer fixo na resposta e no PDF.
4. **Nunca inventar números:** alvo sem cotação → `422 price_unavailable`; venda exige cotação; saldo `count_as` sem cotação usa `vlr_investido` (`saldo_source: cost`, warning `cost_basis`) ou o `saldo` informado (`user`); IR de venda sem `pm_fx` é `null` + `not_estimated`.
5. **Moeda:** sempre `price_fx`/`pm_fx`. Cotação de alvo: `price_fx` da carteira → último fechamento (janela de 10 dias) via `apps.market.series`, resolvendo a sigla por `assets.lookup` (BR primeiro, depois `exchange_rank`); internacional usa a variante `"<ticker> BRL"`; `ETF_GB`/GBP leva warning `fx_approx` (débito de câmbio já registrado na [ADR 0014](0014-tickers-sob-demanda.md)).
6. **Guloso e meses:** ações por saldo crescente, unidades inteiras até o teto; papel com preço unitário acima do teto cheio vende 1 unidade sozinho num mês "vazio" (sem vendas de ações do plano e, no mês 1, sem `already_sold`), com `exceeds_cap` e — acima de 20.000 — `exceeds_exemption` + 15 % sobre o lucro das ações do mês. Resíduo fracionário sai junto da última venda do papel **se couber no teto**; senão vira um lote próprio no mês seguinte (nunca estoura o teto por causa da fração). Meses relativos `1..N`, mês 1 sempre presente.
7. **Compras:** valor bruto vendido (+ caixa no mês 1) distribuído por `calcular_compras_rebalanceadas` (cópia literal do legado, sem vender alvos); saldos projetados acumulam mês a mês; `purchases` lista todos os alvos.
8. **Stateless:** nenhum modelo/migration; o plano é recalculado a cada chamada e o PDF renderiza exatamente o envelope de `/plan`.

## Consequências

### Positivas
- Regra fiscal explícita e testada (tabela única em `tax.py`); estouros aparecem como warnings em vez de ficarem escondidos.
- Núcleo testável como funções puras; a API só orquestra.

### Negativas / débitos
- Regras fiscais simplificadas (sem day-trade, sem compensação de prejuízo, BDR tratado como tributado 15 %).
- Nomes de campos de `/resumo → wallet` ainda não confirmados ao vivo (teste `integration` pendente de `DLP_TEST_TOKEN`).
