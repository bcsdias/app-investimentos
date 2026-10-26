# Design — Fase 3c (Otimizações DLP sem Impacto no Engine)

> **Data:** 2026-10-26
> **Branch:** `migracao-django`
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md) (§Fase 3c, §Fase 7)
> **Referência DLP:** [`../../referencia-api-dlp-invest.md`](../../referencia-api-dlp-invest.md) (§1, §4.1, §5.1 `HistoricModel`)
> **Escopo:** otimizações na camada DLP e em `/portfolio/filters` que **não alteram números nem
> tocam `engine/`** (preservam 100% da paridade da Fase 6 — [ADR 0012](../../adr/0012-engine-copia-literal-fase-2-5.md)).
> Os itens que tocam `engine/` ou alteram cálculos foram movidos para a **Fase 7** (pós-paridade).

---

## 1. Contexto e porquê

Na Fase 3, cada combinação de filtro `(ativo, classe, corretora)` em `dlp.buscar_historico` gera
uma chave de cache distinta (`dlp:hist:<token_hash>:<gen>:<params_hash>`) e dispara uma chamada
HTTP `GET /historico` própria à DLP (até 60s de timeout). Porém:

1. `GET /api/v1/portfolio/filters` já chama `buscar_historico(token)` **sem filtros** ao abrir a
   tela, trazendo o histórico completo.
2. Pela especificação OpenAPI da DLP (`HistoricModel`), `ativo`, `classe` e `corretora` são
   **campos obrigatórios** em todas as linhas de `/historico`.

Logo, filtrar em memória sobre o DataFrame completo evita chamadas extras de até 60s a cada troca
de filtro na SPA (Fase 4) e reduz o uso de Redis para **1 DataFrame por usuário**.

Além disso:
- `portfolio._extract_distinct` converte e faz `.strip()` em todas as células do histórico diário
  antes de deduplicar, quando pode deduplicar primeiro via `pd.unique`.
- `dlp._BASE` ainda aponta para o domínio legado `https://users.dlombelloplanilhas.com` Fixo no
  código, enquanto o domínio oficial atual é `https://users.dlpinvest.com.br`.

---

## 2. Objetivos, Não-Objetivos e Anti-Objetivos

- **Objetivos:**
  1. Cachear um único DataFrame de `/historico` por `(token_hash, generation)` no Redis e aplicar
     os filtros `ativo`, `classe` e `corretora` em memória no Pandas.
  2. Otimizar `_extract_distinct` em `apps/api/routers/portfolio.py` aplicando `pd.unique` antes
     da normalização de strings.
  3. Parametrizar `DLP_BASE_URL` em `config/settings.py` (via `.env`) com default
     `https://users.dlpinvest.com.br`.
  4. Validar com teste `integration` (token real) que o DataFrame filtrado em memória é
     equivalente ao retornado por `GET /historico?ativo=...&classe=...`.
- **Não-Objetivos (movidos para a Fase 7):**
  - Alterar `engine/dataset.py` ou `engine/twr.py` (pré-parse de `date`, remoção de `groupby`
    duplicado, corte de `end_date` em `portfolio_df` e recálculo de `fluxo` após `gap`).
- **Anti-Objetivos:**
  - Mudar a assinatura pública de `dlp.buscar_historico(token, ativo=None, classe=None, corretora=None)`
    ou qualquer contrato HTTP existente da Fase 3.
  - Maken/alterar dados cacheados in-place de forma que uma chamada filtrada contamine chamadas
    subsequentes em memória.

---

## 3. Abordagens consideradas

| | Abordagem | Trade-off | Status |
|---|---|---|---|
| **A** | Filtrar dentro de `dlp.buscar_historico`, mantendo assinatura e chave de cache única `dlp:hist:<token_hash>:<gen>` | Zero mudança em `engine/dataset.py` e nos routers; single-flight e invalidação por geração funcionam para todas as chamadas. | **Escolhida** |
| B | Filtrar em `engine/dataset.py::fetch_user_portfolio` | Toca `engine/` sem necessidade e exige que outros consumidores de `buscar_historico` lembrem de filtrar. | Descartada |
| C | Manter cache por filtro além do cache cheio | Duplica memória no Redis sem ganho perceptível (filtrar ~5–20k linhas no Pandas leva < 2 ms). | Descartada |

---

## 4. Especificação técnica

### 4.1. `apps/market/services/sources/dlp.py`

1. **URL base parametrizada:**
   - Em `config/settings.py`: `DLP_BASE_URL = os.getenv("DLP_BASE_URL", "https://users.dlpinvest.com.br").rstrip("/")`.
   - Em `dlp.py`: ler `settings.DLP_BASE_URL` (em tempo de chamada em `_get`, facilitando override em testes).

2. **Cache único + filtro em memória em `buscar_historico`:**
   - Chave Redis passa a ser `mc.make_key("dlp", "hist", _token_hash(token), _generation(token))`
     (sem hash de `params`).
   - Busca sempre `_get("/historico", token, {})` dentro de `mc.single_flight(key)`.
   - Se o payload vier vazio (`df.empty`), não cacheia e retorna `None` (comportamento atual).
   - Caso contrário, normaliza a coluna de data (`_norm_date_col`) e grava o DataFrame **completo**
     no Redis (`TTL_DLP = 600s`).
   - Aplica helper puro `_filter_historico(df, ativo=ativo, classe=classe, corretora=corretora)`:
     - Para cada filtro informado (não `None` e `str(val).strip() != ""`):
       - Localiza a coluna de forma case-insensitive (`ativo`, `classe`, `corretora`).
       - Se a coluna existir, filtra comparando `df[col].astype(str).str.strip() == str(val).strip()`.
     - Se o resultado filtrado ficar vazio, retorna `None` (mesmo contrato de quando a DLP
       devolvia `{"historico": []}` para um filtro sem correspondência).
     - Sempre retorna `.copy()` (ou novo DataFrame resetado no índice) para evitar `SettingWithCopyWarning`
       em `fetch_user_portfolio`.

### 4.2. `apps/api/routers/portfolio.py`

Em `_extract_distinct(df, candidates, label)`:
```python
raw_unique = pd.unique(df[col].dropna())
vals = (str(v).strip() for v in raw_unique)
return sorted({v for v in vals if v})
```
Mantém exatamente o mesmo retorno (lista ordenada de strings não-vazias sem duplicatas).

---

## 5. Testes e Verificação

- **Unitários (`apps/market/tests/test_sources_dlp.py`):**
  - Chamada a `buscar_historico(token)` seguida de `buscar_historico(token, classe="ACAO")` e
    `buscar_historico(token, ativo="PETR4")` faz **apenas 1 chamada HTTP** (`requests.get` mockado
    com `params == {}`) e devolve os subconjuntos corretos.
  - Filtro cujo valor não bate com nenhuma linha retorna `None`.
  - `invalidate_user_cache(token)` invalida a chave única e força novo fetch na chamada seguinte.
  - `settings.DLP_BASE_URL` é respeitado por `_get`.
- **Unitários (`apps/api/tests/test_portfolio_filters_refresh.py`):**
  - `_extract_distinct` com valores repetidos, espaços nas bordas, `None`/`NaN` e strings vazias
    continua devolvendo a lista única ordenada.
- **Integração (`@pytest.mark.integration`):**
  - Com `DLP_TEST_TOKEN`, busca o histórico completo, escolhe uma `classe` e um `ativo` presentes
    e compara o resultado de `buscar_historico(token, classe=c)` (filtrado em memória) com a
    chamada direta `_get("/historico", token, {"classe": c})` — mesmas linhas e colunas.

