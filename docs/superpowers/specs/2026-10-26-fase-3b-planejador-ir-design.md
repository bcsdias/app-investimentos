# Design — Fase 3b (Planejador de Migração com Isenção de IR)

> **Data:** 2026-10-26
> **Branch:** `migracao-django`
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md) (§1.1, §4, §Fase 3b)
> **Fase anterior:** 3 — [`2026-10-26-fase-3-api-carteira-design.md`](2026-10-26-fase-3-api-carteira-design.md)
> **Referência DLP:** [`../../referencia-api-dlp-invest.md`](../../referencia-api-dlp-invest.md) (§5.2 `AtivoCarteiraModel`)
> **Legado:** `src/ui/pages/3_migracao_ir.py` (735 linhas) — **não é portado literalmente** (ver §3).
> **Escopo:** endpoints `/api/v1/migration/{wallet,plan,plan.pdf}` em `apps/migration_ir/`. A tela
> React é da Fase 4. A Fase 3c (otimizações DLP) é independente e não faz parte deste design.

---

## 1. Contexto e porquê

O planejador ajuda o usuário a **migrar a carteira atual para uma alocação-alvo** vendendo
posições aos poucos, de modo que as vendas de ações fiquem **dentro da isenção de IR de
R$ 20.000/mês**, e reinvestindo o que foi vendido (mais caixa) em compras que rebalanceiam os
alvos. Saída: um cronograma mês a mês (o que vender, o que comprar) e um PDF.

No legado isso é uma sub-app Streamlit com três defeitos estruturais que esta fase corrige:

1. **Alvos fixos** `IMAB11 / VWRA11 / BITH11` espalhados por cálculo, tabelas e PDF.
2. **Não conhece a regra fiscal**: o "limite" é um teto genérico (padrão R$ 19.500) aplicado a
   qualquer classe escolhida — FII/ETF/BDR vendidos aparecem como se fossem isentos.
3. **Números silenciosamente errados**: preço inventado (100/120/60) quando a cotação falha;
   saldo de ativo estrangeiro calculado com `price` (moeda original) em vez de `price_fx` (BRL);
   papel com preço acima do teto vendido sem aviso, estourando o limite.

A backend já tem o que é preciso: `dlp.buscar_resumo_carteira` (cache Redis 10 min,
single-flight, erros tipados), `apps.market.assets` (universo de ~7.500 tickers) e
`apps.market.series` (fetch-on-read de cotações), além do padrão de API da Fase 3
(`ApiSessionAuth`, `ApiError`, `get_token`, `dlp_errors`).

---

## 2. Decisões travadas (brainstorming 2026-10-26)

Decisões **do Bruno**:

| # | Tema | Decisão |
|---|---|---|
| 1 | Ativos-alvo | **Livres**: o usuário define a alocação-alvo com quaisquer tickers e pesos. Nada fixo no código. |
| 2 | Saldo existente que "conta" como alvo (ex.: CDB somando ao IMAB11) | **Associação manual, explícita**, papel a papel. O backend **não** classifica nem sugere (as listas de palavras-chave do legado somem). |
| 3 | Regra fiscal | O planejador **conhece a regra de isenção**: o teto mensal vale só para vendas de **ações** (`classe == "ACAO"`); demais classes são tributadas e marcadas como tal. |
| 4 | IR das vendas tributadas | **Calcular uma estimativa** por papel e por mês, sempre rotulada como estimativa, com aviso explícito para conferir a apuração oficial na DLP. |
| 5 | Onde o código mora | **Abordagem A** — app próprio `apps/migration_ir/` (ver §3). |
| 6 | Saldo de papel sem cotação (RF/CDB/Tesouro com `price_fx` nulo) marcado `count_as` | `qtd × price_fx` quando há cotação; senão **`vlr_investido`** marcado `saldo_source: "cost"`; o request aceita **`saldo` opcional por papel** (valor atual informado pelo usuário, `saldo_source: "user"`), que tem precedência. Para **`sell`**, cotação continua obrigatória. |

Decisões **do designer** (calls razoáveis, apresentadas no brainstorming sem objeção; registrar em ADR):

| # | Tema | Decisão | Por quê |
|---|---|---|---|
| D1 | O que vender | **Por papel**, não por classe. Cada papel da carteira tem um estado: `sell`, `count_as:<alvo>` ou `ignore` (default). | Com associação manual (#2), o modelo por papel é o natural; "vender todas as ações" vira atalho na SPA. |
| D2 | Cotação de alvo indisponível | **`422 price_unavailable`** com o ticker, nunca preço inventado. | App financeiro: plano errado silencioso é pior que erro. |
| D3 | Papel de ação com preço unitário > teto restante de um mês vazio | Vende **1 unidade sozinha num mês**, e esse mês sai com aviso `exceeds_exemption` + IR estimado sobre as vendas de ações desse mês. | Não trava o plano; torna o estouro visível em vez de escondido. |
| D4 | Moeda | Sempre os campos **`price_fx` / `pm_fx`** (BRL) do `/resumo`. | Corrige o bug do legado com ativos estrangeiros. |
| D5 | Vendas de ações já feitas no mês corrente | Campo opcional `already_sold_this_month` (R$), **descontado do teto do mês 1**. | A isenção é sobre o total de vendas de ações no mês, incluindo as feitas fora do plano. |
| D6 | Vendas tributadas | **Todas no mês 1**, fora do teto, quantidade integral. | Adiar não reduz o imposto; espalhar só alonga o plano. |
| D7 | Teto | `monthly_cap` em R$, **0 < cap ≤ 20.000**, default **20.000**. | Acima de 20 mil perde a isenção por definição; abaixo serve de margem de segurança. |
| D8 | Persistência | **Nenhuma**. O plano é recalculado a cada chamada (stateless). | Igual ao legado; sem modelo/migration nesta fase. |
| D9 | Meses | Relativos (`month: 1..N`), **mês 1 = mês corrente**. Sem datas de calendário. | Evita supor quando o usuário executa. |
| D10 | IR e reinvestimento | O valor **bruto** das vendas é reinvestido; o IR estimado é mostrado à parte (pago via DARF depois). | Mantém o plano simples e o IR claramente separado como estimativa. |
| D11 | Teto do usuário × limite legal | Passar do `monthly_cap` gera só warning **`exceeds_cap`**; IR de 15% sobre ações (e warning **`exceeds_exemption`**) só quando as vendas de ações do mês (plano + `already_sold_this_month` no mês 1) passam de **20.000**. | Separa a margem que o usuário escolheu do fato fiscal. |
| D12 | Mês 1 com `already_sold_this_month` | Teto do mês 1 = `monthly_cap − already_sold` (piso 0). Ação que não cabe no restante vai para o mês 2. A venda de "1 unidade sozinha" (D3) só ocorre quando o preço unitário > **`monthly_cap` cheio**, e nunca no mês 1 se ele já tiver vendas de fora. Se `already_sold > 20.000`, o mês 1 não recebe vendas de ações e leva warning `already_over_exemption`. O IR estimado cobre **só vendas do plano**. | Regra única e previsível; lucro das vendas de fora é desconhecido. |
| D13 | Ticker do alvo → cotação de mercado | Alvo presente na carteira: resolver pelo `market_cod` da DLP. Fora da carteira: sigla no universo de `assets` com prioridade BR (`.SA`) e depois `_EXCHANGE_RANK` da Fase 3. Alvo `ETF_GB`/moeda `GBP`: conversão por USD/BRL (débito já registrado na ADR 0014) + warning `fx_approx`. | Reaproveita o que existe; não corrige câmbio GBP nesta fase. |
| D14 | `ativo` repetido em várias linhas (corretoras) | Soma `qtd`, `vlr_investido`; `pm_fx` ponderado pela qtd; `price_fx` da primeira linha não nula; `classe` da linha de maior `qtd` + warning `mixed_class` se divergirem. | Determinístico e visível. |

---

## 3. Abordagens consideradas

| | Abordagem | Trade-off | Status |
|---|---|---|---|
| **A** | App `apps/migration_ir/` com `planner.py` + `tax.py` (Python puro, sem Django), `prices.py`, `pdf.py`, `api.py` | Domínio isolado, segue o §4 do doc mestre, mantém `engine/` restrito à paridade (ADR 0012). Custo: routers em dois lugares (`apps/api/routers/` e aqui — já há precedente em `apps/accounts/api.py`). | **Escolhida** |
| B | Cálculo em `engine/migration.py` + router em `apps/api/routers/migration.py` | Mesmo padrão da Fase 3, mas mistura lógica **nova** no `engine/` (cópia literal sob regra de paridade) e desvia do §4. | Descartada |
| C | Portar literalmente e generalizar depois | Não há paridade a preservar (a Fase 6 só compara as métricas da carteira); o cálculo muda de propósito → trabalho dobrado. | Descartada |

---

## 4. Estrutura

```
backend/apps/migration_ir/
├── __init__.py
├── apps.py            # name = "apps.migration_ir"; adicionar em INSTALLED_APPS
├── schemas.py         # schemas Ninja (request/response) — §5
├── wallet.py          # normaliza /resumo -> list[Holding] (puro, recebe o payload dict)
├── tax.py             # tabela de classes + estimativa de IR (puro) — §6
├── planner.py         # distribuição de vendas + rebalanceamento (puro) — §7
├── prices.py          # cotação dos alvos: wallet price_fx -> apps.market (único que toca Django/rede)
├── pdf.py             # gerar_pdf_plano(plan) -> bytes (fpdf2) — §8
├── api.py             # router fino: /wallet, /plan, /plan.pdf
└── tests/
```

- `config/api.py`: `api.add_router("/migration/", migration_router)`, autenticado por
  `ApiSessionAuth` como os demais.
- Reuso de `apps.api.analysis.get_token` e `apps.api.analysis.dlp_errors`, e de `ApiError`.
- `planner.py`, `tax.py` e `wallet.py` **não importam Django** (testáveis como funções puras).
- `requirements.txt`: adicionar **`fpdf2`** com versão pinada (hoje ausente no backend).
- Convenção do `backend/CLAUDE.md` "routers finos": `api.py` só valida, chama
  `get_token` → `buscar_resumo_carteira` → `wallet` → `prices` → `planner` (→ `pdf`) e serializa.

---

## 5. Contrato HTTP

Todas as rotas exigem sessão (401 sem sessão) e as `POST` exigem CSRF, como na Fase 3.

### 5.1. `GET /api/v1/migration/wallet`

Lê `/resumo` (via `buscar_resumo_carteira`) e devolve os papéis com saldo > 0:

```json
{
  "data": [
    {"ativo": "PETR4", "classe": "ACAO", "qtd": 300, "price": 38.10, "pm": 30.00,
     "saldo": 11430.00, "tax_treatment": "exempt_20k"},
    {"ativo": "HGLG11", "classe": "FII", "qtd": 50, "price": 160.0, "pm": 150.0,
     "saldo": 8000.00, "tax_treatment": "taxed", "tax_rate": 0.20}
  ],
  "meta": {"total": 19430.00}
}
```

- `price` = `price_fx`, `pm` = `pm_fx` (BRL, D4). `qtd` = `qtd` (campo obrigatório na referência;
  aceitar `quantidade` como fallback, como o legado).
- Cada item traz também `vlr_investido`, `market_cod`, `moeda` e `saldo_source` ∈ `market | cost`.
- `tax_treatment` ∈ `exempt_20k | taxed | not_estimated` (§6); `tax_rate` só quando `taxed`.
- Papel sem `price_fx` (ou ≤ 0) **aparece** com `price: null`, `saldo = vlr_investido`,
  `saldo_source: "cost"`. Pode ser `count_as` (decisão #6), **não** pode ser `sell`.
- Se o mesmo `ativo` vier em mais de uma linha, **agregar** conforme D14.
- Ordenação: por `saldo` decrescente.

### 5.2. `POST /api/v1/migration/plan`

```json
{
  "targets": [
    {"ticker": "IMAB11", "weight": 60},
    {"ticker": "VWRA11", "weight": 35},
    {"ticker": "BITH11", "weight": 5}
  ],
  "holdings": [
    {"ativo": "PETR4", "action": "sell"},
    {"ativo": "HGLG11", "action": "sell"},
    {"ativo": "CDB-XP-2027", "action": "count_as", "target": "IMAB11", "saldo": 12850.00}
  ],
  "monthly_cap": 20000,
  "initial_cash": 0,
  "already_sold_this_month": 0,
  "contributions_only": false
}
```

- **`targets`**: 1–10 itens; `ticker` = sigla B3/negociação **no mesmo formato do `ativo` da DLP**
  (ex.: `IMAB11`, `VWRA11`, `AAPL`); pesos inteiros ou decimais > 0 que **somam 100**
  (tolerância 1e-6); sem tickers repetidos. Um alvo pode ou não existir na carteira.
- **`holdings`**: só os papéis com ação diferente de `ignore` precisam vir (ausente = `ignore`).
  Um alvo presente na carteira entra automaticamente com seu próprio saldo; ele **não** pode ser
  marcado `sell` (rebalanceamento só com compras, como no legado) nem `count_as`.
  `saldo` (R$, > 0) é opcional e só vale para `count_as`: sobrescreve o saldo calculado
  (decisão #6). Em `initial_balances.counted[]` cada item traz `saldo_source` ∈ `market | cost | user`.
- **`contributions_only: true`**: ignora `sell`, exige `initial_cash > 0`, gera um único "mês"
  de compras (equivalente ao modo "Apenas Aportes" do legado).
- Valores monetários ≥ 0; `monthly_cap` conforme D7.

**Resposta** (`{data, meta}`):

```json
{
  "data": {
    "initial_balances": [
      {"ticker": "IMAB11", "direct": 5000.0, "counted": [{"ativo": "CDB-XP-2027", "saldo": 12000.0}], "total": 17000.0}
    ],
    "prices": {"IMAB11": {"price": 85.2, "source": "wallet"}, "BITH11": {"price": 58.1, "source": "market"}},
    "months": [
      {
        "month": 1,
        "sales": [
          {"ativo": "HGLG11", "classe": "FII", "qtd": 50, "valor": 8000.0,
           "tax_treatment": "taxed", "lucro_estimado": 500.0, "ir_estimado": 100.0},
          {"ativo": "PETR4", "classe": "ACAO", "qtd": 300, "valor": 11430.0,
           "tax_treatment": "exempt_20k", "lucro_estimado": 2430.0, "ir_estimado": 0.0}
        ],
        "exempt_sales_total": 11430.0,
        "taxed_sales_total": 8000.0,
        "cash_in": 0.0,
        "purchase_total": 19430.0,
        "purchases": [
          {"ticker": "IMAB11", "valor": 12000.0, "price": 85.2, "qtd_sugerida": 140,
           "saldo_final": 29000.0, "participacao_final": 0.62}
        ],
        "ir_estimado_total": 100.0,
        "warnings": []
      }
    ],
    "totals": {"months": 1, "sold": 19430.0, "bought": 19430.0, "ir_estimado": 100.0, "ir_not_estimated": []}
  },
  "meta": {
    "monthly_cap": 20000.0,
    "disclaimer": "Valores de IR são estimativas (sem taxas, sem compensação de prejuízos). Confira a apuração oficial na DLP."
  }
}
```

- `qtd_sugerida = floor(valor / price)` (como o legado). O `valor` de compra é o rebalanceado
  em R$; a quantidade inteira é sugestão.
- `participacao_final` em fração (0–1); a SPA/PDF formata.
- `warnings` por mês: `exceeds_cap` (D3/D11), `exceeds_exemption` (D11), `already_over_exemption`
  (D12), `not_estimated:<ativo>` (§6). Warnings globais do plano em `meta.warnings`:
  `fx_approx:<ticker>` (D13), `mixed_class:<ativo>` (D14), `cost_basis:<ativo>` (saldo pelo custo).
- `totals.ir_not_estimated`: lista de `ativo` vendidos cujo IR não foi estimado.

### 5.3. `POST /api/v1/migration/plan.pdf`

Mesmo corpo de `/plan`; resposta `application/pdf` com
`Content-Disposition: attachment; filename="plano_migracao_ir_YYYYMMDD.pdf"`. Erros saem no
envelope JSON `ApiError` normal (não PDF).

### 5.4. Erros

| Situação | Status | `code` |
|---|---|---|
| Sem sessão / CSRF | 401 / 403 | (padrão Fase 3) |
| Sem token DLP | 409 | `dlp_token_missing` |
| DLP rejeitou o token | 422 | `dlp_token_rejected` |
| DLP indisponível | 502 | `dlp_unavailable` |
| `/resumo` sem `wallet` ou carteira vazia | 404 | `portfolio_empty` |
| Pesos não somam 100, alvo repetido, 0 ou > 10 alvos | 422 | `invalid_allocation` |
| `ativo` em `holdings` que não está na carteira, `sell` em papel sem preço, `saldo` em ação ≠ `count_as`, alvo marcado `sell`/`count_as`, `count_as` para alvo inexistente | 422 | `invalid_holding` (`details.ativo`) |
| `monthly_cap` fora de (0, 20000] ou valor negativo | 422 | `invalid_parameters` |
| Nada a fazer (nenhum `sell` e `initial_cash == 0`; ou `contributions_only` com caixa 0) | 422 | `nothing_to_plan` |
| Sem cotação para algum alvo (D2) | 422 | `price_unavailable` (`details.tickers`) |

---

## 6. Regra fiscal e estimativa de IR — `tax.py`

Tabela por `classe` DLP (constante em código, fácil de revisar):

| Classe | `tax_treatment` | Alíquota sobre lucro |
|---|---|---|
| `ACAO` | `exempt_20k` | 0% dentro do limite; **15%** sobre o lucro de todas as vendas de ações **do plano** naquele mês se o total do mês (plano + `already_sold_this_month` no mês 1) passar de 20.000 (D11/D12) |
| `FII` | `taxed` | 20% |
| `ETF`, `BDR`, `BDR ETF` | `taxed` | 15% |
| `RF`, `ETF RF`, `CRIPTO`, `STOCK`, `ETF_US`, `ETF_GB`, `REIT`, `FI_INFRA`, `FIP_IE`, qualquer outra | `not_estimated` | — (regras próprias: tabela regressiva / IR na fonte, isenção de R$ 35k de cripto, apuração anual do exterior, isenções de infra) |

- `lucro_estimado = (price_fx − pm_fx) × qtd_vendida`. Prejuízo → `ir_estimado = 0` (sem
  compensação). Sem taxas/emolumentos.
- Papéis `not_estimated` **podem** ser vendidos: vão para o mês 1 (D6, fora do teto) com
  `ir_estimado: null` e warning.
- O **teto de 20 mil é sobre o valor de venda** (não sobre o lucro), considerando
  `already_sold_this_month` no mês 1.
- Disclaimer fixo em `meta.disclaimer` e no rodapé do PDF.

---

## 7. Algoritmo — `planner.py`

Entradas já validadas: holdings normalizados, alvos com preço, parâmetros.

1. **Saldos iniciais por alvo** = saldo direto do alvo na carteira + soma dos saldos `count_as`
   para ele (saldo de cada `count_as`: `saldo` do request → `qtd × price_fx` → `vlr_investido`).
2. **Vendas** (se não `contributions_only`):
   - **Tributadas / não estimadas** (`classe != ACAO`, ação `sell`): quantidade integral, mês 1.
   - **Ações** (`ACAO`, ação `sell`): algoritmo guloso do legado — ordenar por saldo crescente;
     encher cada mês até o teto (mês 1: `monthly_cap − already_sold_this_month`, piso 0) vendendo
     unidades inteiras (`floor(cap_restante / price)`); quando não cabe nem 1 unidade, fecha o mês
     e abre o próximo. Se o preço unitário > `monthly_cap` cheio e o mês está vazio (sem vendas do
     plano e, no mês 1, sem `already_sold`) → D3 (1 unidade sozinha, warnings conforme D11).
     Regras do mês 1 com vendas de fora: D12.
     Quantidades fracionárias de ação: vender a parte inteira pelo guloso e a fração residual junto
     da última venda do papel (builder pode simplificar se a DLP nunca devolver fração em `ACAO`).
3. **Compras por mês**: `valor_compra = vendas_do_mês (brutas) + (initial_cash se mês 1)`,
   distribuído por `calcular_compras_rebalanceadas` **portado literalmente** do legado (pesos-alvo,
   saldos projetados, sem vender alvos — remove iterativamente os alvos acima do peso). Os saldos
   projetados acumulam mês a mês.
4. **IR**: por venda via `tax.py`; total do mês; regra do estouro para `ACAO`.

`calcular_compras_rebalanceadas` é a única peça copiada do legado e deve ter testes que fixam
o comportamento atual (caso clássico 60/35/5 com saldos desbalanceados).

---

## 8. Cotações — `prices.py`

Ordem para cada alvo:
1. `price_fx` do próprio alvo na carteira (`source: "wallet"`).
2. `apps.market.assets`: resolver o ticker Yahoo conforme D13 (carteira → `market_cod`; senão
   sigla com prioridade BR e `_EXCHANGE_RANK`); cotação = último fechamento via
   `apps.market.series.get_series` numa janela curta (ex.: últimos 10 dias).
   Ativo `INTL` → usar a variante `"<ticker> BRL"` (já em BRL via PTAX); `ETF_GB` → warning `fx_approx`. `source: "market"`.
3. Nada → acumula em `price_unavailable` (D2).

Isso **substitui** o `yf.Ticker(...).fast_info` do legado (sem chamada Yahoo ao vivo fora do
padrão fetch-on-read da Fase 3). Alvo que não está nem na carteira nem no universo → `invalid_allocation`.

---

## 9. PDF — `pdf.py`

Porta do layout de `gerar_pdf_plano` (fpdf2, A4, banner azul, seções 1-Parâmetros,
2-Saldos iniciais, 3-Cronograma com subtabelas VENDER | COMPRAR por mês), com mudanças:

- Alvos **dinâmicos** (linhas vêm de `targets`; nada fixo).
- Coluna de IR estimado nas vendas e linha "IR estimado do mês"; warnings do mês em texto.
- Rodapé: disclaimer de IR + "Gerado em <data> | app-investimentos" (remove "Antigravity AI Portfolio").
- Recebe o **mesmo objeto de plano** que `/plan` devolve (sem recalcular nada no PDF).
- Fonte `helvetica` (latin-1) como no legado; textos sem caracteres fora de latin-1.

---

## 10. Testes

- **Puros** (`planner`, `tax`, `wallet`): guloso respeita o teto; `already_sold_this_month`;
  D3 com warning e IR do mês estourado; tributadas no mês 1; `not_estimated`; prejuízo → IR 0;
  `count_as` soma no alvo; `contributions_only`; alvo fora da carteira; agregação de `ativo`
  duplicado (D14); `count_as` sem cotação (custo e `saldo` do usuário); D11 cap < 20k sem IR; D12 mês 1 com vendas de fora; rebalanceamento literal (caso legado 60/35/5).
- **Contrato** (`apps/migration_ir/tests/`, test client): 401 sem sessão, CSRF, cada código de
  erro da §5.4, isolamento (usuário A não vê carteira de B), `plan.pdf` devolve
  `application/pdf` começando com `%PDF`.
- **DLP simulada** com payload `/resumo` de fixture (monkeypatch do `requests.get` em `dlp.py`,
  padrão da Fase 3); cotações de mercado com `MarketSeries` de fixture.
- **`integration`** (desmarcado): `/migration/wallet` com `DLP_TEST_TOKEN` real — confirmar
  nomes dos campos (`quantidade` vs `qtd`, presença de `price_fx`/`pm_fx`, valores de `classe`).
- TDD conforme `backend/CLAUDE.md`.

**Definition of Done:** testes verdes (`.venv/bin/pytest -q`), `manage.py check` limpo, e roteiro
manual no Swagger com o token real do Bruno: `/wallet` → `/plan` com 3 alvos livres → baixar
`/plan.pdf` e conferir visualmente.

---

## 11. Riscos e pendências

| Risco | Tratamento |
|---|---|
| Campos reais de `/resumo → wallet` podem divergir da referência (`quantidade`/`qtd`, `price_fx` nulo). | Teste `integration` no roteiro manual; `wallet.py` concentra a normalização. |
| Regras fiscais simplificadas (sem taxas, sem compensação de prejuízo, sem day-trade, BDR tratado como tributado). | Rótulo de estimativa + disclaimer; classes complexas ficam `not_estimated`. |
| Sigla do alvo vs ticker Yahoo (`IMAB11` → `IMAB11.SA`). | `prices.py` resolve pelo universo de `assets`; teste com alvo BR e INTL. |
| Ação com fração (desdobramento/grupamento). | §7 passo 2; validar no `integration`. |

---

## 12. Documentação e ADRs

- **ADR 0015** — Planejador de IR: alvos livres, associação manual, regra de isenção por classe,
  estimativa de IR, saldo de papel sem cotação (#1–#6, D1–D14).
- Fechamento de fase pelo checklist de [`../../README.md`](../../README.md#checklist-de-atualização-ao-fechar-uma-fase)
  (doc mestre §2.0/§2.2/§4/§9, `backend/README.md`, `backend/CLAUDE.md` se surgir convenção nova).

---

## 13. Deixado para o builder

- Nomes internos de funções/classes, organização dos schemas Ninja, formato exato dos testes.
- Tamanho da janela de cotação em `prices.py` (desde que pegue o último fechamento disponível).
- Detalhes visuais do PDF além do listado na §9.
- Arredondamento de exibição (a API devolve floats crus; arredondar só no PDF).

