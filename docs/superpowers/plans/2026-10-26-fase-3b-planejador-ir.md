# Fase 3b — Planejador de Migração com Isenção de IR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Endpoints `/api/v1/migration/{wallet,plan,plan.pdf}` que montam um cronograma mês a mês de vendas (ações dentro da isenção de R$ 20 mil/mês, demais classes no mês 1 com IR estimado) e compras que rebalanceiam alvos livres, mais o PDF do plano.

**Architecture:** Novo app `apps/migration_ir/`. Núcleo puro sem Django: `wallet.py` (normaliza `/resumo`), `tax.py` (tabela de classes e IR), `planner.py` (validação + algoritmo). `prices.py` é o único que toca `apps.market` (cotações dos alvos). `pdf.py` renderiza o mesmo objeto que `/plan` devolve. `api.py` é router fino: token → `buscar_resumo_carteira` → `wallet` → `planner.validate_plan_input` → `prices` → `planner.build_plan` (→ `pdf`).

**Tech Stack:** Django 5.2, `django-ninja` 1.4, pandas 2.2, `fpdf2` 2.8 (já no `.venv`, falta no `requirements.txt`), `pytest-django`, `LocMemCache`.

**Spec:** `docs/superpowers/specs/2026-10-26-fase-3b-planejador-ir-design.md`

## Global Constraints

Todo task herda implicitamente esta seção.

- **Branch:** `migracao-django`. Rodar tudo a partir de `backend/` com `.venv/bin/python` / `.venv/bin/pytest`.
- **Sem `engine/`**: nada deste plano toca `engine/` (ADR 0012). Nenhum modelo/migration (D8, stateless).
- **Pureza:** `wallet.py`, `tax.py`, `planner.py` **não importam Django** nem `apps.*` (exceto `planner` importar `wallet`/`tax`). `prices.py` levanta `planner.PlanError`; só `api.py` conhece `ApiError`.
- **Moeda:** sempre `price_fx` / `pm_fx` do `/resumo` (D4). `qtd` com fallback `quantidade`.
- **Constantes fiscais (spec §6):** limite de isenção `20_000.0` sobre **valor de venda**; `ACAO` acima do limite → 15 % sobre o lucro das vendas de ações do plano no mês; `FII` → 20 %; `ETF`, `BDR`, `BDR ETF` → 15 %; qualquer outra classe → `not_estimated`. Prejuízo → IR 0 (sem compensação).
- **`monthly_cap`:** `0 < cap <= 20000`, default `20000` (D7). Valores monetários ≥ 0.
- **Alvos:** 1–10, pesos > 0 somando 100 (tolerância `1e-6`), sem repetidos; ticker normalizado com `.strip().upper()`.
- **Disclaimer (texto fixo, `meta.disclaimer` e rodapé do PDF):** `"Valores de IR são estimativas (sem taxas, sem compensação de prejuízos). Confira a apuração oficial na DLP."`
- **Erros (spec §5.4):** `dlp_token_missing` 409 · `dlp_token_rejected` 422 (nunca 401) · `dlp_unavailable` 502 · `portfolio_empty` 404 · `invalid_allocation` / `invalid_holding` (`details.ativo`) / `invalid_parameters` / `nothing_to_plan` / `price_unavailable` (`details.tickers`) 422. Mensagens em português.
- **Envelope:** `{"data": ..., "meta": ...}`; floats crus na API (arredondar só no PDF).
- **PDF:** fonte `helvetica` (latin-1); nada fora de latin-1 chega ao fpdf (substituir por `?`). Rodapé sem "Antigravity AI Portfolio".
- **Testes:** rede sempre mockada (`dlp.requests.get` ou `buscar_resumo_carteira`, `series.get_series`); chamadas reais só sob `@pytest.mark.integration`.
- **Commits:** Conventional Commits + Gitmoji + imperativo em português. **Nunca** `Co-Authored-By` (regra do `backend/CLAUDE.md`). Um commit por task, com a mensagem indicada.
- **TDD obrigatório:** teste → rodar e ver falhar → implementar → ver passar.

## Review Focus

1. **Venda tributada de papel sem `pm_fx`** (DLP devolve `pm_fx` nulo para ativo recém-transferido) → lucro desconhecido; esperado `lucro_estimado: null`, `ir_estimado: null`, warning `not_estimated:<ativo>` e entrada em `totals.ir_not_estimated` — nunca IR calculado com custo 0 (Task 4, `test_taxed_sale_without_pm_is_not_estimated`).
2. **Ticker de alvo digitado em minúsculas ou com espaço** (`" imab11"`) → reconhecido como o `IMAB11` da carteira (saldo direto e preço `wallet`), não como alvo novo sem cotação (Task 3, `test_target_ticker_is_normalized_and_matches_wallet`).
3. **Campos numéricos do `/resumo` como string ou nulos** (`"qtd": "300"`, `"price_fx": null`, `"price_fx": 0`) → normalizados (`0`/nulo = sem cotação), sem `TypeError` (Task 1, `test_normalize_coerces_strings_and_zero_price`).
4. **Ação com quantidade fracionária** (`qtd = 10.5`) → o resíduo `0.5` sai junto da última venda do papel, nunca uma venda extra de qtd < 1 em mês separado (Task 4, `test_fractional_residual_rides_last_sale`).
5. **`already_sold_this_month` ≥ `monthly_cap` mas ≤ 20.000** → mês 1 sem vendas de ações (vão para o mês 2) e **sem** warning `already_over_exemption`; o mês 1 continua existindo com as tributadas/caixa (Task 4, `test_already_sold_fills_month1_cap_without_exemption_warning`).

---

## Estrutura de arquivos

Tudo em `backend/`. `M` = modificar.

```
requirements.txt                          # M — fpdf2~=2.8 (Task 7)
config/settings.py                        # M — "apps.migration_ir" em INSTALLED_APPS (Task 1)
config/api.py                             # M — api.add_router("/migration/", ...) (Task 6)
apps/market/assets.py                     # M — Asset.exchange_rank + lookup() (Task 5)
apps/market/tests/test_assets.py          # M (Task 5)
apps/migration_ir/
├── __init__.py
├── apps.py                               # MigrationIrConfig, name = "apps.migration_ir"
├── wallet.py                             # Holding, normalize_wallet (Task 1)
├── tax.py                                # tax_treatment, estimate_profit, estimate_ir (Task 2)
├── planner.py                            # PlanError, Action, PlanInput, validate_plan_input,
│                                         #   calcular_compras_rebalanceadas, build_plan (Tasks 3–4)
├── prices.py                             # TargetPrice, resolve_target_prices (Task 5)
├── schemas.py                            # schemas Ninja (Task 6)
├── api.py                                # router /wallet, /plan, /plan.pdf (Tasks 6–7)
├── pdf.py                                # render_plan_pdf (Task 7)
└── tests/
    ├── __init__.py
    ├── fixtures.py                       # RESUMO (payload /resumo), helpers
    ├── fixtures/ativos_migration.csv     # universo mínimo p/ prices e API
    ├── test_wallet.py  test_tax.py  test_planner_validate.py  test_planner_build.py
    ├── test_prices.py  test_api.py  test_pdf.py  test_integration.py
```

### Fixture compartilhada (`apps/migration_ir/tests/fixtures.py`, criada na Task 1)

```python
RESUMO = {"wallet": [
    {"ativo": "PETR4", "classe": "ACAO", "qtd": 300, "price_fx": 38.10, "pm_fx": 30.00,
     "vlr_investido": 9000.0, "market_cod": "BVMF:PETR4", "moeda": "BRL"},
    {"ativo": "HGLG11", "classe": "FII", "qtd": 50, "price_fx": 160.0, "pm_fx": 150.0,
     "vlr_investido": 7500.0, "market_cod": "BVMF:HGLG11", "moeda": "BRL"},
    {"ativo": "CDB-XP-2027", "classe": "RF", "qtd": 1, "price_fx": None, "pm_fx": None,
     "vlr_investido": 12000.0, "market_cod": "", "moeda": "BRL"},
    {"ativo": "IMAB11", "classe": "ETF RF", "qtd": 50, "price_fx": 100.0, "pm_fx": 90.0,
     "vlr_investido": 4500.0, "market_cod": "BVMF:IMAB11", "moeda": "BRL"},
    {"ativo": "BTC", "classe": "CRIPTO", "qtd": 0.1, "price_fx": 300000.0, "pm_fx": 200000.0,
     "vlr_investido": 20000.0, "market_cod": "CURRENCY:BTC", "moeda": "BRL"},
]}
```

Saldos: BTC 30000 · CDB 12000 (custo) · PETR4 11430 · HGLG11 8000 · IMAB11 5000 · total 66430.

`fixtures/ativos_migration.csv` (mesmo formato de `apps/market/tests/fixtures/ativos_mini.csv`, `;`, BOM opcional): linhas `IMAB11;BVMF:IMAB11;ETF RF`, `VWRA11;BVMF:VWRA11;ETF`, `BITH11;BVMF:BITH11;ETF`, `PETR4;BVMF:PETR4;ACAO`, `GDX;LON:GDX;ETF_GB`, `GDX;NYSEARCA:GDX;ETF_US`, `AAPL;NASDAQ:AAPL;STOCK` (coluna `razao_social` livre). Fixture pytest `migration_assets_csv` no `tests/conftest.py` do app, igual à `assets_csv` do `conftest.py` raiz, apontando para esse arquivo.

---

### Task 1: App `migration_ir` + `wallet.py`

**Files:**
- Create: `apps/migration_ir/{__init__.py,apps.py,wallet.py}`, `apps/migration_ir/tests/{__init__.py,fixtures.py}`
- Modify: `config/settings.py` (INSTALLED_APPS, depois de `"apps.api"`)
- Test: `apps/migration_ir/tests/test_wallet.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Holding(ativo: str, classe: str, qtd: float, price: float | None, pm: float | None, vlr_investido: float, market_cod: str, moeda: str)` com propriedades `saldo -> float` (`qtd * price` se `price`, senão `vlr_investido`) e `saldo_source -> Literal["market", "cost"]`.
  - `normalize_wallet(payload: dict) -> tuple[list[Holding], list[str]]` — holdings com `saldo > 0`, ordenados por `saldo` decrescente (empate: `ativo`); segundo item = warnings globais (`mixed_class:<ativo>`). Payload sem `wallet` → `([], [])`.

- [ ] **Step 1: Write the failing tests**

```python
def test_normalize_maps_fx_fields_and_sorts_by_saldo():
    hs, warns = normalize_wallet(RESUMO)
    assert [h.ativo for h in hs] == ["BTC", "CDB-XP-2027", "PETR4", "HGLG11", "IMAB11"]
    petr = hs[2]
    assert (petr.price, petr.pm, petr.saldo, petr.saldo_source) == (38.10, 30.0, pytest.approx(11430.0), "market")
    cdb = hs[1]
    assert (cdb.price, cdb.saldo, cdb.saldo_source) == (None, 12000.0, "cost")
    assert warns == []


def test_normalize_coerces_strings_and_zero_price():
    hs, _ = normalize_wallet({"wallet": [
        {"ativo": "X", "classe": "ACAO", "quantidade": "10", "price_fx": "0", "pm_fx": None, "vlr_investido": "50"},
        {"ativo": "Z", "classe": "ACAO", "qtd": 0, "price_fx": 10, "vlr_investido": 0},
    ]})
    assert len(hs) == 1 and hs[0].qtd == 10.0 and hs[0].price is None and hs[0].saldo == 50.0


def test_normalize_aggregates_duplicate_ativo():
    hs, warns = normalize_wallet({"wallet": [
        {"ativo": "PETR4", "classe": "ACAO", "qtd": 100, "price_fx": None, "pm_fx": 20.0, "vlr_investido": 2000},
        {"ativo": "PETR4", "classe": "BDR", "qtd": 300, "price_fx": 40.0, "pm_fx": 40.0, "vlr_investido": 12000},
    ]})
    (h,) = hs
    assert (h.qtd, h.vlr_investido, h.price, h.classe) == (400, 14000, 40.0, "BDR")
    assert h.pm == pytest.approx(35.0)          # (100*20 + 300*40) / 400
    assert warns == ["mixed_class:PETR4"]


def test_normalize_without_wallet_key():
    assert normalize_wallet({}) == ([], [])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/migration_ir -v`
Expected: FAIL — `ModuleNotFoundError: apps.migration_ir.wallet`.

- [ ] **Step 3: Implement**

Scaffold do app (`apps.py` com `name = "apps.migration_ir"`, `default_auto_field = "django.db.models.BigAutoField"`), `INSTALLED_APPS`. Em `wallet.py`: helper `_num(v) -> float | None` (aceita str/int/float, inválido → `None`); `price`/`pm` ≤ 0 → `None`. Agregação D14: soma `qtd`/`vlr_investido`; `pm` ponderado pela `qtd` das linhas com `pm`; `price` da primeira linha não nula; `classe`/`market_cod`/`moeda` da linha de maior `qtd`; warning se `classe` divergir.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest apps/migration_ir -v && .venv/bin/python manage.py check`
Expected: PASS; `no issues`.

- [ ] **Step 5: Commit**

```bash
git add config/settings.py apps/migration_ir
git commit -m "✨ feat(migration_ir): app do planejador e normalização da carteira /resumo"
```

---

### Task 2: `tax.py`

**Files:**
- Create: `apps/migration_ir/tax.py`
- Test: `apps/migration_ir/tests/test_tax.py`

**Interfaces:**
- Produces:
  - `EXEMPTION_LIMIT = 20_000.0`, `ACAO_OVER_LIMIT_RATE = 0.15`
  - `tax_treatment(classe: str) -> tuple[Literal["exempt_20k", "taxed", "not_estimated"], float | None]` — alíquota só para `taxed`; `classe` comparada após `.strip().upper()`.
  - `estimate_profit(price: float, pm: float | None, qtd: float) -> float | None` — `(price − pm) × qtd`; `pm is None` → `None`.
  - `estimate_ir(lucro: float | None, rate: float) -> float | None` — `None` se `lucro is None`; `max(0, lucro) × rate`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("classe,expected", [
    ("ACAO", ("exempt_20k", None)), (" acao ", ("exempt_20k", None)),
    ("FII", ("taxed", 0.20)), ("ETF", ("taxed", 0.15)), ("BDR", ("taxed", 0.15)), ("BDR ETF", ("taxed", 0.15)),
    ("RF", ("not_estimated", None)), ("ETF RF", ("not_estimated", None)), ("CRIPTO", ("not_estimated", None)),
    ("STOCK", ("not_estimated", None)), ("ETF_GB", ("not_estimated", None)), ("QUALQUER", ("not_estimated", None)),
])
def test_tax_treatment_table(classe, expected):
    assert tax_treatment(classe) == expected


def test_profit_and_ir():
    assert estimate_profit(160.0, 150.0, 50) == 500.0
    assert estimate_ir(500.0, 0.20) == 100.0
    assert estimate_ir(-300.0, 0.15) == 0.0
    assert estimate_profit(10.0, None, 5) is None and estimate_ir(None, 0.15) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/migration_ir/tests/test_tax.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — tabela como `dict` constante no topo do módulo (revisável).

- [ ] **Step 4: Run to verify they pass** — mesmo comando, PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/migration_ir/tax.py apps/migration_ir/tests/test_tax.py
git commit -m "✨ feat(migration_ir): regra de isenção por classe e estimativa de IR"
```

---

### Task 3: `planner.py` — validação e rebalanceamento do legado

**Files:**
- Create: `apps/migration_ir/planner.py`
- Test: `apps/migration_ir/tests/test_planner_validate.py`

**Interfaces:**
- Consumes: `Holding`, `normalize_wallet` (Task 1).
- Produces:
  - `class PlanError(Exception)` com `code: str`, `message: str`, `details: dict | None` (`__init__(code, message, details=None)`).
  - `@dataclass(frozen=True) Action(ativo: str, action: Literal["sell", "count_as", "ignore"], target: str | None = None, saldo: float | None = None)`
  - `@dataclass(frozen=True) PlanInput(targets: dict[str, float], wallet: dict[str, Holding], actions: dict[str, Action], monthly_cap: float, initial_cash: float, already_sold: float, contributions_only: bool)` — `targets` na ordem do request, tickers normalizados; `wallet` indexado por `ativo`; `actions` só `sell`/`count_as` (ações `ignore` descartadas; em `contributions_only`, `sell` também descartado).
  - `validate_plan_input(*, targets: list[tuple[str, float]], actions: list[Action], wallet: list[Holding], monthly_cap: float = 20000.0, initial_cash: float = 0.0, already_sold: float = 0.0, contributions_only: bool = False) -> PlanInput`
  - `target_holding(inp: PlanInput, ticker: str) -> Holding | None` — holding cujo `ativo.strip().upper() == ticker`.
  - `calcular_compras_rebalanceadas(total_aporte: float, pesos_alvo: dict[str, float], valores_atuais: dict[str, float]) -> dict[str, float]` — **cópia literal** de `src/ui/pages/3_migracao_ir.py:26-70` (só ajustes de lint/type hints).

Ordem de validação (o primeiro erro vence): `invalid_allocation` → `invalid_parameters` → `invalid_holding` → `nothing_to_plan`. A cotação (`price_unavailable`) é da Task 5.

- [ ] **Step 1: Write the failing tests**

```python
WALLET, _ = normalize_wallet(RESUMO)
T3 = [("IMAB11", 60), ("VWRA11", 35), ("BITH11", 5)]
SELL_PETR = [Action("PETR4", "sell")]


def _v(**kw):
    base = dict(targets=T3, actions=SELL_PETR, wallet=WALLET)
    return validate_plan_input(**{**base, **kw})


def test_rebalance_legacy_60_35_5():
    out = calcular_compras_rebalanceadas(10000, {"A": 60, "B": 35, "C": 5}, {"A": 10000, "B": 0, "C": 2000})
    assert out["A"] == pytest.approx(2631.578947, rel=1e-6)
    assert out["B"] == pytest.approx(7368.421053, rel=1e-6)
    assert out["C"] == 0.0 and sum(out.values()) == pytest.approx(10000)


@pytest.mark.parametrize("kw,code", [
    (dict(targets=[("IMAB11", 60), ("VWRA11", 39)]), "invalid_allocation"),
    (dict(targets=[("IMAB11", 50), ("imab11 ", 50)]), "invalid_allocation"),
    (dict(targets=[]), "invalid_allocation"),
    (dict(targets=[(f"T{i}", 100 / 11) for i in range(11)]), "invalid_allocation"),
    (dict(targets=[("IMAB11", 100), ("VWRA11", 0)]), "invalid_allocation"),
    (dict(monthly_cap=0), "invalid_parameters"),
    (dict(monthly_cap=20000.01), "invalid_parameters"),
    (dict(initial_cash=-1), "invalid_parameters"),
    (dict(already_sold=-1), "invalid_parameters"),
    (dict(actions=[Action("XPTO3", "sell")]), "invalid_holding"),
    (dict(actions=[Action("CDB-XP-2027", "sell")]), "invalid_holding"),
    (dict(actions=[Action("PETR4", "sell", saldo=100)]), "invalid_holding"),
    (dict(actions=[Action("IMAB11", "sell")]), "invalid_holding"),
    (dict(actions=[Action("IMAB11", "count_as", target="VWRA11")]), "invalid_holding"),
    (dict(actions=[Action("CDB-XP-2027", "count_as", target="BOVA11")]), "invalid_holding"),
    (dict(actions=[Action("CDB-XP-2027", "count_as")]), "invalid_holding"),
    (dict(actions=[Action("CDB-XP-2027", "count_as", target="IMAB11", saldo=0)]), "invalid_holding"),
    (dict(actions=[Action("PETR4", "sell"), Action("PETR4", "sell")]), "invalid_holding"),
    (dict(actions=[]), "nothing_to_plan"),
    (dict(contributions_only=True), "nothing_to_plan"),
])
def test_validation_errors(kw, code):
    with pytest.raises(PlanError) as ei:
        _v(**kw)
    assert ei.value.code == code


def test_invalid_holding_details_name_the_ativo():
    with pytest.raises(PlanError) as ei:
        _v(actions=[Action("XPTO3", "sell")])
    assert ei.value.details == {"ativo": "XPTO3"}


def test_target_ticker_is_normalized_and_matches_wallet():
    inp = _v(targets=[(" imab11", 100)])
    assert list(inp.targets) == ["IMAB11"]
    assert target_holding(inp, "IMAB11").ativo == "IMAB11"


def test_contributions_only_drops_sells_keeps_count_as():
    inp = _v(contributions_only=True, initial_cash=1000,
             actions=[Action("PETR4", "sell"), Action("CDB-XP-2027", "count_as", target="IMAB11")])
    assert list(inp.actions) == ["CDB-XP-2027"]


def test_cash_only_plan_is_valid():
    assert _v(actions=[], initial_cash=500).initial_cash == 500
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/migration_ir/tests/test_planner_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: apps.migration_ir.planner`.

- [ ] **Step 3: Implement** `PlanError`, `Action`, `PlanInput`, `validate_plan_input`, `target_holding`, `calcular_compras_rebalanceadas` (ver Interfaces). Mensagens em português, uma por condição (ex.: `"Os pesos dos alvos devem somar 100."`, `"O papel {ativo} não pode ser vendido: não tem cotação."`).

- [ ] **Step 4: Run to verify they pass** — mesmo comando, PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/migration_ir/planner.py apps/migration_ir/tests/test_planner_validate.py
git commit -m "✨ feat(migration_ir): validação do plano e rebalanceamento portado do legado"
```

---

### Task 4: `planner.build_plan` — vendas, compras, IR e warnings

**Files:**
- Modify: `apps/migration_ir/planner.py`
- Test: `apps/migration_ir/tests/test_planner_build.py`

**Interfaces:**
- Consumes: `PlanInput`, `target_holding`, `calcular_compras_rebalanceadas` (Task 3); `tax_treatment`, `estimate_profit`, `estimate_ir`, `EXEMPTION_LIMIT`, `ACAO_OVER_LIMIT_RATE` (Task 2).
- Produces: `build_plan(inp: PlanInput, prices: dict[str, float]) -> tuple[dict, list[str]]` — primeiro item = `{"initial_balances", "months", "totals"}` com os campos exatos do spec §5.2 (sem `prices`, que a API acrescenta); segundo = warnings globais do planner (`cost_basis:<ativo>` para cada `count_as` ou saldo direto de alvo com `saldo_source == "cost"`).

Regras que o spec deixa abertas e este plano fixa:
- `counted[]` traz `{"ativo", "saldo", "saldo_source"}` com `saldo_source ∈ market | cost | user`.
- `purchases` lista **todos** os alvos na ordem do request (inclusive `valor == 0`); `participacao_final = saldo_final / Σ saldo_final dos alvos`; `qtd_sugerida = floor(valor / price)`.
- Em cada mês as vendas tributadas/não estimadas vêm antes das de ações, na ordem das `actions` do request. Ações em ordem de `saldo` crescente (empate: `ativo`).
- "Mês vazio" (D3) = sem vendas de **ações** do plano naquele mês e, se mês 1, `already_sold == 0`.
- O mês 1 sempre existe (pode ficar sem vendas); os meses são contíguos `1..N`.
- Warnings de teto (`exceeds_cap`, `exceeds_exemption`) só em meses com venda de ação do plano; total do mês = vendas de ações do plano + `already_sold` (mês 1). `already_over_exemption` no mês 1 quando `already_sold > 20000`.
- `not_estimated:<ativo>` vai para o mês da venda quando `tax_treatment == "not_estimated"` **ou** o lucro não pôde ser estimado (`pm is None`).
- `contributions_only`: um único mês, `sales == []`, `cash_in = initial_cash`.

Guloso de ações (spec §7.2), para fixar o que a assinatura não determina:

```text
remaining = max(0, cap - already_sold)  # mês 1
para cada ação (saldo crescente), q = qtd:
    enquanto q >= 1:
        n = min(floor(remaining / price), floor(q))
        se n >= 1: vende n no mês atual; remaining -= n*price; q -= n
        senão se mês vazio e price > cap: vende 1 sozinha; q -= 1; fecha o mês
        senão: fecha o mês (próximo mês: remaining = cap)
    se 0 < q < 1: soma q à última venda desse papel (ou cria a venda no mês atual se não houver)
```

Helper de teste: `_h(ativo, classe, qtd, price, pm=None, vlr=0.0) -> Holding` (moeda `"BRL"`, `market_cod` vazio) e `_inp(holdings, actions, **kw)` que chama `validate_plan_input` com alvos `[("A", 100)]` por padrão e `prices={"A": 10.0}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_greedy_fills_cap_then_spills():
    hs = [_h("AAA3", "ACAO", 1500, 10.0, 5.0), _h("BBB3", "ACAO", 1000, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell"), Action("BBB3", "sell")]), {"A": 10.0})
    m1, m2 = plan["months"]
    assert [(s["ativo"], s["qtd"]) for s in m1["sales"]] == [("BBB3", 1000), ("AAA3", 1000)]
    assert m1["exempt_sales_total"] == 20000 and m1["ir_estimado_total"] == 0 and m1["warnings"] == []
    assert [(s["ativo"], s["qtd"]) for s in m2["sales"]] == [("AAA3", 500)]
    assert plan["totals"]["months"] == 2 and plan["totals"]["sold"] == 25000


def test_already_sold_reduces_month1_cap():
    hs = [_h("AAA3", "ACAO", 1000, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")], already_sold=15000), {"A": 10.0})
    assert [m["exempt_sales_total"] for m in plan["months"]] == [5000, 5000]


def test_already_sold_fills_month1_cap_without_exemption_warning():
    hs = [_h("AAA3", "ACAO", 100, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")], already_sold=20000), {"A": 10.0})
    m1, m2 = plan["months"]
    assert m1["sales"] == [] and m1["warnings"] == [] and m2["exempt_sales_total"] == 1000


def test_already_over_exemption_warning():
    hs = [_h("AAA3", "ACAO", 100, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")], already_sold=21000), {"A": 10.0})
    assert plan["months"][0]["warnings"] == ["already_over_exemption"]
    assert plan["months"][0]["sales"] == []


def test_unit_price_above_cap_sells_one_alone_with_ir():
    hs = [_h("CARA3", "ACAO", 2, 25000.0, 20000.0)]
    plan, _ = build_plan(_inp(hs, [Action("CARA3", "sell")]), {"A": 10.0})
    assert len(plan["months"]) == 2
    m1 = plan["months"][0]
    assert m1["sales"][0]["qtd"] == 1 and m1["sales"][0]["ir_estimado"] == pytest.approx(750.0)
    assert set(m1["warnings"]) == {"exceeds_cap", "exceeds_exemption"}


def test_unit_price_above_cap_skips_month1_with_outside_sales():
    hs = [_h("CARA3", "ACAO", 2, 25000.0, 20000.0)]
    plan, _ = build_plan(_inp(hs, [Action("CARA3", "sell")], already_sold=1000), {"A": 10.0})
    assert [len(m["sales"]) for m in plan["months"]] == [0, 1, 1]


def test_user_cap_below_20k_only_warns_exceeds_cap():
    hs = [_h("CARA3", "ACAO", 1, 16000.0, 10000.0)]
    plan, _ = build_plan(_inp(hs, [Action("CARA3", "sell")], monthly_cap=15000), {"A": 10.0})
    m1 = plan["months"][0]
    assert m1["warnings"] == ["exceeds_cap"] and m1["ir_estimado_total"] == 0


def test_fractional_residual_rides_last_sale():
    hs = [_h("AAA3", "ACAO", 10.5, 10.0, 5.0)]
    plan, _ = build_plan(_inp(hs, [Action("AAA3", "sell")]), {"A": 10.0})
    (m1,) = plan["months"]
    assert [s["qtd"] for s in m1["sales"]] == [10.5] and m1["exempt_sales_total"] == 105.0


def test_taxed_and_not_estimated_go_to_month1():
    hs = [_h("AAA3", "ACAO", 3000, 10.0, 5.0), _h("HGLG11", "FII", 50, 160.0, 150.0),
          _h("BOVA11", "ETF", 10, 100.0, 120.0), _h("BTC", "CRIPTO", 0.1, 300000.0, 200000.0)]
    acts = [Action(a, "sell") for a in ("AAA3", "HGLG11", "BOVA11", "BTC")]
    plan, _ = build_plan(_inp(hs, acts), {"A": 10.0})
    m1 = plan["months"][0]
    by = {s["ativo"]: s for s in m1["sales"]}
    assert by["HGLG11"]["ir_estimado"] == 100.0 and by["BOVA11"]["ir_estimado"] == 0.0
    assert by["BTC"]["ir_estimado"] is None and "not_estimated:BTC" in m1["warnings"]
    assert m1["taxed_sales_total"] == pytest.approx(8000 + 1000 + 30000)
    assert [s["ativo"] for s in m1["sales"]][:3] == ["HGLG11", "BOVA11", "BTC"]
    assert plan["totals"]["ir_not_estimated"] == ["BTC"] and plan["totals"]["ir_estimado"] == 100.0


def test_taxed_sale_without_pm_is_not_estimated():
    hs = [_h("HGLG11", "FII", 50, 160.0, None)]
    plan, _ = build_plan(_inp(hs, [Action("HGLG11", "sell")]), {"A": 10.0})
    s = plan["months"][0]["sales"][0]
    assert s["lucro_estimado"] is None and s["ir_estimado"] is None
    assert plan["totals"]["ir_not_estimated"] == ["HGLG11"]


def test_count_as_balances_and_purchases_accumulate():
    hs = [_h("AAA3", "ACAO", 3000, 10.0, 5.0), _h("IMAB11", "ETF RF", 50, 100.0, 90.0),
          _h("CDB1", "RF", 1, None, None, vlr=12000.0), _h("CDB2", "RF", 1, None, None, vlr=1000.0)]
    acts = [Action("AAA3", "sell"), Action("CDB1", "count_as", target="IMAB11"),
            Action("CDB2", "count_as", target="IMAB11", saldo=1500.0)]
    inp = _inp(hs, acts, targets=[("IMAB11", 50), ("VWRA11", 50)])
    plan, warns = build_plan(inp, {"IMAB11": 100.0, "VWRA11": 50.0})
    ib = {b["ticker"]: b for b in plan["initial_balances"]}
    assert ib["IMAB11"]["direct"] == 5000 and ib["IMAB11"]["total"] == 18500
    assert [(c["ativo"], c["saldo_source"]) for c in ib["IMAB11"]["counted"]] == [("CDB1", "cost"), ("CDB2", "user")]
    assert warns == ["cost_basis:CDB1"]
    m1, m2 = plan["months"]
    assert m2["purchases"][0]["saldo_final"] + m2["purchases"][1]["saldo_final"] == pytest.approx(18500 + 30000)
    assert m1["purchase_total"] == 20000 and m2["purchase_total"] == 10000


def test_contributions_only_single_month():
    hs = [_h("IMAB11", "ETF RF", 50, 100.0, 90.0)]
    inp = _inp(hs, [], targets=[("IMAB11", 100)], contributions_only=True, initial_cash=1000)
    plan, _ = build_plan(inp, {"IMAB11": 100.0})
    (m1,) = plan["months"]
    assert m1["sales"] == [] and m1["cash_in"] == 1000
    p = m1["purchases"][0]
    assert (p["ticker"], p["valor"], p["qtd_sugerida"], p["participacao_final"]) == ("IMAB11", 1000.0, 10, 1.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/migration_ir/tests/test_planner_build.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_plan'`.

- [ ] **Step 3: Implement `build_plan`** conforme Interfaces e o pseudocódigo do guloso.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest apps/migration_ir -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/migration_ir/planner.py apps/migration_ir/tests/test_planner_build.py
git commit -m "✨ feat(migration_ir): cronograma de vendas no teto de isenção com compras rebalanceadas"
```

---

### Task 5: `assets.lookup` + `prices.py`

**Files:**
- Modify: `apps/market/assets.py` (campo `exchange_rank` em `Asset`, função `lookup`)
- Modify: `apps/market/tests/test_assets.py`
- Create: `apps/migration_ir/prices.py`, `apps/migration_ir/tests/conftest.py`, `apps/migration_ir/tests/fixtures/ativos_migration.csv`
- Test: `apps/migration_ir/tests/test_prices.py`

**Interfaces:**
- Consumes: `PlanError`, `PlanInput`, `target_holding` (Task 3); `apps.market.series.get_series(name, start, end)`.
- Produces:
  - `Asset.exchange_rank: int = _DEFAULT_EXCHANGE_RANK` (último campo, com default; `load_universe` preenche).
  - `assets.lookup(sigla: str, market_cod: str | None = None) -> Asset | None` — com `market_cod`, tenta `universe[_yf_ticker(sigla, market_cod)]`; senão (ou se ausente) a sigla com prioridade `mercado == "BR"`, depois menor `exchange_rank`, depois ordem do CSV.
  - `@dataclass(frozen=True) TargetPrice(price: float, source: Literal["wallet", "market"])`
  - `resolve_target_prices(inp: PlanInput, today: dt.date | None = None) -> tuple[dict[str, TargetPrice], list[str]]` — segundo item = warnings `fx_approx:<ticker>`. Levanta `PlanError("invalid_allocation", ..., {"tickers": [...]})` para alvo fora da carteira e do universo; depois `PlanError("price_unavailable", ..., {"tickers": [...]})` para os sem cotação.

Regras: (1) holding do alvo com `price` → `wallet`. (2) `lookup(ticker, holding.market_cod if holding else None)`; série = `asset.yf_ticker` se `BR`, senão `f"{asset.yf_ticker} BRL"`; `get_series(nome, today - 10 dias, today)`; preço = último valor não nulo; `KeyError`/`None`/vazia → indisponível. `classe == "ETF_GB"` (ou `holding.moeda == "GBP"`) → `fx_approx`. Alvo na carteira sem cotação e fora do universo → `price_unavailable` (não `invalid_allocation`).

- [ ] **Step 1: Write the failing tests**

`test_assets.py` (usa `assets_csv`):

```python
def test_lookup_prefers_br_then_exchange_rank(assets_csv):
    assert assets.lookup("GDX").yf_ticker == "GDX"            # NYSEARCA (rank 1) vence LON (rank 2)
    assert assets.lookup("GDX", "LON:GDX").yf_ticker == "GDX.L"
    assert assets.lookup("PETR4").yf_ticker == "PETR4.SA"
    assert assets.lookup("NADA") is None
```

`test_prices.py` (usa `migration_assets_csv`; `get_series` falso via `monkeypatch.setattr("apps.migration_ir.prices.series.get_series", fake)` devolvendo `pd.Series([49.0, 50.0])` para `"VWRA11.SA"`, `[20.0]` para `"BITH11.SA"`, `[7.0]` para `"GDX.L BRL"` e `None` para o resto; registrar os nomes pedidos):

```python
def test_wallet_price_first_then_market(fake_series):
    inp = validate_plan_input(targets=[("IMAB11", 60), ("VWRA11", 35), ("BITH11", 5)],
                              actions=[], wallet=WALLET, initial_cash=100)
    prices, warns = resolve_target_prices(inp, today=dt.date(2026, 10, 26))
    assert prices["IMAB11"] == TargetPrice(100.0, "wallet")
    assert prices["VWRA11"] == TargetPrice(50.0, "market") and prices["BITH11"].price == 20.0
    assert "IMAB11.SA" not in fake_series.calls and warns == []


def test_etf_gb_uses_brl_variant_and_warns(fake_series): ...
    # alvo GDX com wallet contendo GDX market_cod "LON:GDX" sem price_fx → 7.0, warns == ["fx_approx:GDX"]

def test_unknown_target_is_invalid_allocation(fake_series): ...   # alvo "ZZZZ11" → code invalid_allocation, details tickers ["ZZZZ11"]

def test_missing_quote_is_price_unavailable(fake_series): ...     # alvo "AAPL" (série None) → price_unavailable, tickers ["AAPL"]
```

(Os três últimos seguem o mesmo esqueleto do primeiro; asserções nos comentários.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/market/tests/test_assets.py apps/migration_ir/tests/test_prices.py -v`
Expected: FAIL — `AttributeError: module 'apps.market.assets' has no attribute 'lookup'` / `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `exchange_rank`, `lookup`, `prices.py`, fixture CSV e `migration_assets_csv`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest apps/market apps/migration_ir -q`
Expected: PASS (inclusive os testes antigos de `assets`/busca).

- [ ] **Step 5: Commit**

```bash
git add apps/market/assets.py apps/market/tests/test_assets.py apps/migration_ir
git commit -m "✨ feat(migration_ir): cotação dos alvos pela carteira ou pelo universo de tickers"
```

---

### Task 6: Schemas + `GET /wallet` e `POST /plan`

**Files:**
- Create: `apps/migration_ir/schemas.py`, `apps/migration_ir/api.py`
- Modify: `config/api.py` (`api.add_router("/migration/", migration_router)` após `/assets/`)
- Test: `apps/migration_ir/tests/test_api.py`

**Interfaces:**
- Consumes: `get_token`, `dlp_errors` (`apps.api.analysis`), `ApiError`, `dlp.buscar_resumo_carteira`, `normalize_wallet`, `validate_plan_input`, `resolve_target_prices`, `build_plan`, `tax_treatment`.
- Produces:
  - Schemas: `TargetIn(ticker: str, weight: float)`, `HoldingActionIn(ativo: str, action: Literal["sell","count_as","ignore"], target: str | None = None, saldo: float | None = None)`, `PlanRequest(targets: list[TargetIn], holdings: list[HoldingActionIn] = [], monthly_cap: float = 20000, initial_cash: float = 0, already_sold_this_month: float = 0, contributions_only: bool = False)`; saídas `WalletOut` e `PlanOut` com os campos exatos do spec §5.1/§5.2 (aninhados; opcionais `| None`).
  - Em `api.py`: `router = Router(tags=["migration"])`; `compute_plan(user, payload: PlanRequest) -> dict` (envelope `{data, meta}` completo, reutilizado pela Task 7); context manager `plan_errors()` que converte `PlanError` em `ApiError(422, e.code, e.message, e.details)`.
  - `meta` do `/plan`: `{"monthly_cap", "disclaimer", "warnings"}` com `warnings = wallet + prices + planner`. `data.prices[t] = {"price", "source"}`.
  - `/wallet`: item = `{ativo, classe, qtd, price, pm, saldo, saldo_source, vlr_investido, market_cod, moeda, tax_treatment, tax_rate?}`; `meta.total = Σ saldo`.

Fixture de teste: `user_with_token_client` e `user_with_token` já existem em `apps/api/tests/conftest.py`; **mover** as duas para o `conftest.py` raiz (junto de `auth_client`) para o app novo usá-las, sem mudar o comportamento. `monkeypatch.setattr("apps.migration_ir.api.dlp.buscar_resumo_carteira", lambda tok: ...)`; `get_series` falso como na Task 5.

- [ ] **Step 1: Write the failing tests**

```python
PLAN_BODY = {
    "targets": [{"ticker": "IMAB11", "weight": 60}, {"ticker": "VWRA11", "weight": 35}, {"ticker": "BITH11", "weight": 5}],
    "holdings": [{"ativo": "PETR4", "action": "sell"}, {"ativo": "HGLG11", "action": "sell"},
                 {"ativo": "CDB-XP-2027", "action": "count_as", "target": "IMAB11", "saldo": 12850.0}],
}


def test_wallet_lists_holdings(client_resumo):
    r = client_resumo.get("/api/v1/migration/wallet")
    body = r.json()
    assert r.status_code == 200 and body["meta"]["total"] == pytest.approx(66430)
    first = {i["ativo"]: i for i in body["data"]}
    assert first["HGLG11"]["tax_treatment"] == "taxed" and first["HGLG11"]["tax_rate"] == 0.20
    assert (first["CDB-XP-2027"]["price"], first["CDB-XP-2027"]["saldo_source"]) == (None, "cost")
    assert [i["ativo"] for i in body["data"]][0] == "BTC"      # saldo decrescente


def test_plan_end_to_end(client_resumo):
    r = client_resumo.post("/api/v1/migration/plan", PLAN_BODY, content_type="application/json")
    assert r.status_code == 200
    data, meta = r.json()["data"], r.json()["meta"]
    (m1,) = data["months"]
    assert m1["taxed_sales_total"] == 8000 and m1["exempt_sales_total"] == pytest.approx(11430)
    assert m1["ir_estimado_total"] == pytest.approx(100.0)
    buy = {p["ticker"]: p for p in m1["purchases"]}
    assert buy["IMAB11"]["valor"] == pytest.approx(4518) and buy["VWRA11"]["valor"] == pytest.approx(13048)
    assert buy["BITH11"]["valor"] == pytest.approx(1864) and buy["VWRA11"]["qtd_sugerida"] == 260
    assert buy["IMAB11"]["participacao_final"] == pytest.approx(0.6)
    assert data["prices"]["IMAB11"] == {"price": 100.0, "source": "wallet"}
    assert data["initial_balances"][0]["total"] == pytest.approx(17850)
    assert meta["disclaimer"].startswith("Valores de IR são estimativas")


@pytest.mark.parametrize("patch,status,code", [
    ({"targets": [{"ticker": "IMAB11", "weight": 50}]}, 422, "invalid_allocation"),
    ({"holdings": [{"ativo": "XPTO3", "action": "sell"}]}, 422, "invalid_holding"),
    ({"monthly_cap": 25000}, 422, "invalid_parameters"),
    ({"holdings": []}, 422, "nothing_to_plan"),
    ({"targets": [{"ticker": "AAPL", "weight": 100}], "holdings": [{"ativo": "PETR4", "action": "sell"}]},
     422, "price_unavailable"),
])
def test_plan_business_errors(client_resumo, patch, status, code): ...
    # POST PLAN_BODY | patch → status e r.json()["code"] == code

def test_dlp_errors_map(...): ...
    # sem UserToken → 409 dlp_token_missing; buscar_resumo_carteira levanta DLPAuthError → 422 dlp_token_rejected;
    # DLPUnavailableError → 502 dlp_unavailable; payload {"wallet": []} → 404 portfolio_empty

def test_requires_session_and_csrf(client, db, user_with_token): ...
    # GET /wallet sem login → 401; Client(enforce_csrf_checks=True) logado, POST /plan → 403

def test_users_are_isolated(...): ...
    # user (tok-teste) e other_user (tok-ana) com resumos diferentes por token → cada /wallet vê só o seu
```

(`client_resumo` = fixture do app que monkeypatcha `buscar_resumo_carteira` → `RESUMO`, `get_series` falso e `migration_assets_csv`, devolvendo `user_with_token_client`.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/migration_ir/tests/test_api.py -v`
Expected: FAIL — `404` em `/api/v1/migration/wallet` (router não registrado).

- [ ] **Step 3: Implement** schemas, `api.py` (router fino; `portfolio_empty` quando `normalize_wallet` devolve lista vazia ou o payload é `None`), registro em `config/api.py`, mover fixtures para o `conftest.py` raiz.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest -q`
Expected: PASS na suíte inteira (as fixtures movidas não quebram `apps/api`).

- [ ] **Step 5: Commit**

```bash
git add conftest.py apps/api/tests/conftest.py apps/migration_ir config/api.py
git commit -m "✨ feat(migration_ir): endpoints /migration/wallet e /migration/plan"
```

---

### Task 7: `pdf.py` + `POST /plan.pdf`

**Files:**
- Create: `apps/migration_ir/pdf.py`
- Modify: `apps/migration_ir/api.py`, `requirements.txt` (nova seção `# --- Fase 3b: planejador de IR ---` com `fpdf2~=2.8`)
- Test: `apps/migration_ir/tests/test_pdf.py`, `apps/migration_ir/tests/test_api.py`

**Interfaces:**
- Consumes: `compute_plan` (Task 6); layout de referência `src/ui/pages/3_migracao_ir.py:368-735` (`gerar_pdf_plano`).
- Produces: `render_plan_pdf(envelope: dict, params: PlanRequest, *, generated_at: dt.datetime, compress: bool = True) -> bytes` — `envelope` é exatamente o retorno de `compute_plan`; nada é recalculado.

Layout (spec §9): banner azul `(26, 54, 93)`; 1-Parâmetros (modo, teto, caixa, `already_sold_this_month`, pesos `TICKER: N%`); 2-Saldos iniciais por alvo (direto + contados + total); 3-Cronograma: por mês subtabelas VENDER (ativo, qtd, valor, IR est.) | COMPRAR (ticker, valor, qtd sugerida, part. final), linha "IR estimado do mês" e warnings do mês em texto. Rodapé: disclaimer + `Gerado em <dd/mm/aaaa HH:MM> | app-investimentos` + `Página n/{nb}`. Valores formatados `R$ 1.234,56` só aqui.

- [ ] **Step 1: Write the failing tests**

```python
def test_pdf_has_dynamic_targets_ir_and_footer(plan_envelope):
    raw = render_plan_pdf(plan_envelope, PlanRequest(**PLAN_BODY), generated_at=dt.datetime(2026, 10, 26, 9, 0), compress=False)
    assert raw.startswith(b"%PDF")
    for needle in (b"VWRA11", b"BITH11", b"IR estimado", b"Confira a apura", b"app-investimentos"):
        assert needle in raw
    assert b"Antigravity" not in raw


def test_pdf_survives_non_latin1_text(plan_envelope):
    plan_envelope["meta"]["warnings"].append("fx_approx:ÜB€R→")
    assert render_plan_pdf(plan_envelope, PlanRequest(**PLAN_BODY), generated_at=dt.datetime(2026, 10, 26)).startswith(b"%PDF")


def test_plan_pdf_endpoint(client_resumo):
    r = client_resumo.post("/api/v1/migration/plan.pdf", PLAN_BODY, content_type="application/json")
    assert r.status_code == 200 and r["Content-Type"] == "application/pdf"
    assert re.fullmatch(r'attachment; filename="plano_migracao_ir_\d{8}\.pdf"', r["Content-Disposition"])
    assert r.content.startswith(b"%PDF")


def test_plan_pdf_errors_are_json(client_resumo):
    r = client_resumo.post("/api/v1/migration/plan.pdf", {**PLAN_BODY, "monthly_cap": 0}, content_type="application/json")
    assert r.status_code == 422 and r.json()["code"] == "invalid_parameters"
```

(`plan_envelope` = fixture que chama `compute_plan` via `client_resumo` → `POST /plan` e devolve `r.json()`.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest apps/migration_ir/tests/test_pdf.py apps/migration_ir/tests/test_api.py -k pdf -v`
Expected: FAIL — `ModuleNotFoundError: apps.migration_ir.pdf` / 404.

- [ ] **Step 3: Implement** `render_plan_pdf` (helper `_latin1(s)` = `s.encode("latin-1", "replace").decode("latin-1")` em todo texto; `pdf.compress = compress`) e a rota `POST /plan.pdf` (`HttpResponse(bytes, content_type="application/pdf")` + `Content-Disposition` com `dt.date.today():%Y%m%d`).

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/pytest apps/migration_ir -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt apps/migration_ir
git commit -m "✨ feat(migration_ir): PDF do plano de migração com IR estimado (fpdf2)"
```

---

### Task 8: Integração real, ADR 0015 e fechamento da fase

**Files:**
- Create: `apps/migration_ir/tests/test_integration.py`, `../docs/adr/0015-planejador-de-ir.md`
- Modify: `../docs/adr/README.md` (índice), `../docs/consolidacao_arquitetura_e_migracao.md`, `../docs/README.md`, `README.md`, `CLAUDE.md` (backend, se houver convenção nova)

- [ ] **Step 1: Write the integration test**

```python
pytestmark = pytest.mark.integration

def test_real_migration_wallet(db, user):
    token = os.environ.get("DLP_TEST_TOKEN") or pytest.skip("DLP_TEST_TOKEN não definido no ambiente")
    ut = UserToken(user=user); ut.set_token(token); ut.save()
    c = Client(); c.force_login(user)
    r = c.get("/api/v1/migration/wallet")
    assert r.status_code == 200 and r.json()["data"]
    raw = dlp.buscar_resumo_carteira(token)["wallet"][0]
    print("Campos /resumo wallet:", sorted(raw))
    print("Classes:", sorted({i["classe"] for i in r.json()["data"]}))
```

- [ ] **Step 2: Run with the real token**

Run: `DLP_TEST_TOKEN=... .venv/bin/pytest apps/migration_ir/tests/test_integration.py -m integration -s -v`
Expected: PASS; conferir na saída que existem `qtd` (ou `quantidade`), `price_fx`, `pm_fx` e que as classes batem com a tabela de `tax.py`. Divergência → ajustar `wallet.py`/`tax.py` com teste novo antes de seguir.

- [ ] **Step 3: Definition of Done**

Run: `.venv/bin/pytest -q && .venv/bin/python manage.py check`
Expected: verdes; `no issues`. Roteiro manual no Swagger (`/api/v1/docs`) com o token do Bruno: `/migration/wallet` → `/migration/plan` com 3 alvos livres → baixar `/migration/plan.pdf` e conferir visualmente (o Bruno executa/aprova).

- [ ] **Step 4: ADR 0015 + checklist de documentação** ([`../docs/README.md`](../../README.md#checklist-de-atualização-ao-fechar-uma-fase))

- ADR 0015 "Planejador de IR": alvos livres, associação manual, regra de isenção por classe, estimativa de IR, saldo sem cotação (decisões #1–#6 e D1–D14 do spec §2) + as regras fixadas na Task 4 (mês vazio, ordem das vendas, `purchases` com todos os alvos, `pm` nulo → `not_estimated`).
- Doc mestre: §Fase 3b concluída com data; §9 `[x]`; §4 com `apps/migration_ir/` como **existente**; §2.x com os endpoints `/migration/*`.
- `backend/README.md`: estado, estrutura (`apps/migration_ir/`), contagem de testes, `fpdf2`.
- `backend/CLAUDE.md`: registrar que routers de domínio podem morar no próprio app (`apps/<app>/api.py`) e que núcleos de cálculo novos ficam puros (sem Django) fora de `engine/`.

- [ ] **Step 5: Commit**

```bash
git add apps/migration_ir/tests/test_integration.py ../docs README.md CLAUDE.md
git commit -m "📝 docs: fecha a Fase 3b — ADR 0015, doc mestre e READMEs"
```
