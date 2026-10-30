# Design — Fase 6 (Validação de paridade e remoção do legado Streamlit)

> **Data:** 2026-10-27
> **Branch:** `migracao-django`
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md) (§2.3, §5 Fase 6, §6.6, §8.1, §9 item 11)
> **Fases anteriores:** 0 a 5 — specs em `docs/superpowers/specs/`
> **ADRs relevantes:** [0012](../../adr/0012-engine-copia-literal-fase-2-5.md) (engine cópia literal), [0014](../../adr/0014-tickers-sob-demanda.md) (tickers sob demanda / `.L` via PTAX USD), [0016](../../adr/0016-empacotamento-no-lab.md) (empacotamento no LAB, Supabase descartado)

---

## 1. Intenção e porquê

O app é financeiro: antes de apagar o legado Streamlit (`src/`) é preciso **provar** que o
app novo (engine em `backend/engine/` + API `/api/v1/portfolio/*`) entrega os mesmos números
que o legado entregava para a mesma carteira e o mesmo período.

A Fase 6 também é a **linha de base congelada** antes da Fase 7, que vai mudar números de
propósito (recortes `end_date`/`gap` em `engine/dataset.py`). Depois da Fase 7 não há mais
como comparar com o legado.

**Sucesso:** um relatório de paridade em que (a) a camada de engine bate em 1e-6 relativo
em todas as métricas, para todos os casos; (b) a camada ponta a ponta bate, ou cada
divergência está explicada e aceita pelo usuário; e então `src/` e as libs legadas são
removidas do projeto.

## 2. Decisões tomadas com o usuário

| # | Tema | Decisão |
|---|---|---|
| 1 | **Streamlit não sobe mais** | O Supabase (onde o legado guardava o token) está inutilizável. A comparação **não** usa a UI: um script importa o `FinancialReport` do legado (`src/engine/financial_report.py`) e o chama sem tela, passando o token diretamente. |
| 2 | **Duas camadas** | **Camada 1 (engine):** os dois lados recebem *exatamente* os mesmos dados de entrada; tolerância rígida. **Camada 2 (ponta a ponta):** cada lado com sua própria fonte de dados de mercado; tolerância mais folgada e divergências investigadas caso a caso. |
| 3 | **Token** | Token DLP real do usuário. Entra **somente por variável de ambiente** no momento da execução; nunca em arquivo versionado, log, relatório ou documento. |
| 4 | **Carteiras compostas** | O legado aceitava benchmarks compostos (`{'nome': ..., 'composicao': {...}}`, p.ex. `IMID BRL 50 + (IPCA+6%) 50`); o app novo não oferece (API, catálogo e SPA). **Perda aceita** pelo usuário: fica fora da paridade e não bloqueia a remoção do legado; reavaliar no futuro. Registrar na ADR da fase. |

## 3. Fatos levantados no código (base para o design)

- O Supabase só é usado em `src/data/user_store.py` (armazenamento do token). O cálculo
  (`FinancialReport.fetch_user_portfolio` → `build_dataset` → `plot_*(return_fig=True)`) não
  depende dele.
- O Upstash é opcional: sem `UPSTASH_REDIS_REST_URL`/`TOKEN` e sem `st.secrets`,
  `src/data/cache.py` usa só a memória.
- As dependências do legado (`streamlit`, `upstash_redis`, `supabase`, `selenium`,
  `webdriver_manager`, `bcb`, `yfinance`, `matplotlib`, `dotenv`) **estão instaladas** em
  `backend/.venv`, mas não constam de `backend/requirements.txt` (vieram do
  `requirements.txt` da raiz).
- Os métodos `plot_*` do legado com `return_fig=True` devolvem `(fig, dados)` — esses `dados`
  são o oráculo de cada métrica:
  `plot_twr_evolution` → df base 100, `plot_drawdown`, `plot_rolling_volatility(252)`,
  `plot_rolling_sharpe(252)`, `plot_risk_return_scatter`, `plot_irr_evolution`,
  `simulate_shadow_portfolios`.
- `generate_summary_table` **não devolve números**: só exporta CSV já formatado em `%` com 2
  casas. Não serve de oráculo direto.
- `backend/engine/{twr,irr,metrics}.py` são cópia literal de `src/engine/`, com um único
  desvio documentado: `pct_change(fill_method=None)` em `metrics.py` (ADR 0012).
- O fluxo novo é `engine.dataset.fetch_user_portfolio` → `engine.dataset.build_dataset`
  (benchmarks via `apps.market.series`, sintéticos via `apps.market.transforms`) →
  `engine.compute.compute_*`. O `/summary` deriva seus KPIs desses mesmos `compute_*`
  (`backend/apps/api/routers/portfolio.py`).
- O legado obtém mercado ao vivo por `src/data/sources/market_data.py`
  (`buscar_historico`, `buscar_dados_benchmark` [yfinance], `buscar_dados_bcb`,
  `buscar_dados_b3` [CSV baixado por Selenium], `buscar_dados_tesouro`, `buscar_dolar_bcb`)
  e combina tudo em `processar_benchmarks` (conversões `* BRL` via PTAX, `IPCA + 6%`,
  carteiras compostas). O novo lê do Postgres (ingestão da Fase 5).

## 4. Não-objetivos e anti-objetivos

**Não-objetivos:**
- Corrigir qualquer número. Os bugs de recorte `end_date`/`gap` (Fase 7) e o câmbio GBP de
  ativos `.L` (ADR 0014) devem aparecer **iguais** dos dois lados. Paridade prova
  equivalência, não correção.
- Tocar em `backend/engine/` ou em `apps/market/` para "fazer bater". Se a camada 1
  divergir, isso é um bug a reportar ao usuário antes de qualquer correção.
- Comparar o Planejador de IR (`3_migracao_ir.py` × `apps/migration_ir/`) — já tem testes
  próprios desde a Fase 3b.
- Comparar a UI visualmente.
- Carteiras compostas como benchmark (decisão 4).

**Anti-objetivos (seria fracasso mesmo "funcionando"):**
- O token, valores em R$ da carteira ou o histórico DLP do usuário parar em git, log
  persistido ou documento.
- Afrouxar a tolerância da camada 1 até passar, ou "ajustar" o harness para esconder uma
  diferença real.
- Remover `src/` antes de o usuário aprovar o relatório.

## 5. O que construir

### 5.1. Harness `scripts/parity_check.py`

Script rodado **no host** com `backend/.venv`, com `src/` (raiz do repo) e `backend/` no
`sys.path`, Django configurado (`DJANGO_SETTINGS_MODULE`) e acesso ao Postgres/Redis do LAB.

- Token em `DLP_TOKEN` (env). Diferente do que o mestre §8.1 sugere (`--token`), para não
  vazar em histórico do shell/`ps`; atualizar §8.1.
- `matplotlib` em backend `Agg`; figuras fechadas após extrair os dados.
- Executa a matriz de casos (§5.3) nas duas camadas e grava o relatório (§5.5).
- Saída com código ≠ 0 se a camada 1 falhar em qualquer caso.

### 5.2. Métricas comparadas

Por caso, comparar série a série (mesmas colunas, mesmo índice de datas, valores):

| Métrica | Legado (oráculo) | Novo |
|---|---|---|
| TWR base 100 (`df_combined`) | `report.df_combined` / `plot_twr_evolution` | `DatasetResult.df_combined` / `compute_twr_evolution` |
| Drawdown | `plot_drawdown` | `compute_drawdown` |
| Volatilidade móvel 252 | `plot_rolling_volatility(252)` | `compute_rolling_metrics` |
| Sharpe móvel 252 | `plot_rolling_sharpe(252)` | `compute_rolling_metrics` |
| Risco × retorno (CAGR, vol, Sharpe) | `plot_risk_return_scatter` | `compute_risk_return` |
| Evolução do XIRR | `plot_irr_evolution` | `compute_irr_evolution` |
| Shadow portfolio | `simulate_shadow_portfolios` | `compute_shadow` |
| Tabela anual | ver abaixo | `compute_yearly_summary` |
| KPIs do `/summary` | derivados dos oráculos acima pela mesma regra do router (`pat_*`, `aportes`, `ganho`, último XIRR, min drawdown, linha `Carteira` do risco×retorno) | resposta de `/summary` (camada 2) / mesma derivação sobre `compute_*` (camada 1) |

**Tabela anual do legado:** como `generate_summary_table` só exporta texto formatado, o
harness reproduz **literalmente** as linhas numéricas de
`src/engine/financial_report.py:404-414` (antes do `.map` de formatação) sobre o
`df_combined` do legado, com comentário apontando a origem. *(Decisão minha.)*

Também comparar `portfolio_df` (colunas `date`, `vlr_mercado`, `vlr_investido`,
`proventos`, `fluxo`) na camada 1, por ser a entrada de XIRR e Shadow.

### 5.3. Matriz de casos

- **Token:** o do usuário (um só).
- **Janelas (3):**
  1. histórico completo: `start` = primeira data do histórico DLP, `end` = última, **igual
     nos dois lados**. O legado também sempre passava `start`/`end`
     (`src/ui/pages/4_configuracoes.py:124-133`), e a API não omite `start` (o default de
     `validate_request` é um ano atrás, `backend/apps/api/analysis.py:81`). O ajuste de `gap`
     roda nos dois lados, com gap ≈ 0;
  2. últimos 12 meses (`end` = último dia útil com dado, `start` = `end` − 1 ano);
  3. recorte no meio da carteira: `start` ≈ data mediana do histórico DLP, `end` = hoje —
     exercita o ajuste de `gap`.
  As datas concretas são calculadas pelo harness a partir do histórico e gravadas no
  relatório.
- **Filtros:** sem filtro de `ativo`/`classe` em todos os casos; mais um caso com um filtro
  de `classe` existente na carteira (escolhido pelo harness a partir de `/filters`), só na
  janela 1. *(Decisão minha: cobre o caminho `_filter_historico` da Fase 3c.)*
- **Benchmarks:** um conjunto fixo que cubra **cada tipo de origem/transformação**, usando
  nomes presentes nos dois catálogos (`src/data/benchmarks_config.py` e
  `apps/market/catalog`): BCB (`CDI`), sintético de taxa (`IPCA + 6%`), Yahoo em BRL (um
  ativo `* BRL`, p.ex. `S&P 500 BRL`), B3 (`IFIX`) e Tesouro (um título disponível nos
  dois). Sem carteiras compostas (decisão 4). O builder fixa a lista
  final; se um tipo não existir nos dois lados, registrar no relatório e seguir.

### 5.4. As duas camadas

**Camada 1 — engine, dados idênticos (tolerância 1e-6 relativo, com piso absoluto 1e-9
para valores ≈ 0; NaN deve casar com NaN):**
- O histórico DLP é baixado **uma vez** por caso e congelado em memória.
- No legado, substituir (monkeypatch) as funções de busca de mais baixo nível de
  `src.data.sources.market_data` — `buscar_historico` (também o nome importado em
  `src.engine.financial_report`), `buscar_dados_benchmark`, `buscar_dados_bcb`,
  `buscar_dados_b3`, `buscar_dados_tesouro`, `buscar_dolar_bcb` — por versões que
  devolvem os dados congelados. Assim a lógica de `processar_benchmarks` e de
  `build_dataset` do legado (conversões BRL, `IPCA + X%`, compostas) **roda de verdade** e
  entra na comparação.
- Os dados de mercado congelados vêm do **Postgres** (repositório da ingestão), no formato
  bruto que cada função legada devolvia (ver contrato no mestre §6.4). No lado novo,
  `apps.market.series` lê do mesmo Postgres; `dlp.buscar_historico` é substituído para
  devolver o mesmo histórico congelado.
- **Isolamento do LAB:** o harness **não** limpa o Redis compartilhado nem grava no Postgres.
  Dentro do próprio processo ele troca o backend de cache do app (desligado ou em memória) e
  substitui `apps.market.series._ensure_fresh` por um no-op, para não haver busca sob
  demanda nem `upsert_points`. O app em execução não é afetado.
- **Desvio do ADR 0012 (`pct_change(fill_method=None)`):** qualquer diferença na camada 1
  conta como FALHA. Se a diferença ficar restrita a vol/Sharpe móveis e **sumir** ao repetir o
  lado legado com `pct_change` forçado a `fill_method=None`, o relatório a marca como
  "DESVIO CONHECIDO (ADR 0012)" com a magnitude, para o usuário decidir. Nunca é silenciada.
- Se para alguma série não for possível montar exatamente o formato bruto que o legado
  esperava, ela sai da camada 1 e fica só na camada 2, com o motivo no relatório.

**Camada 2 — ponta a ponta, cada um com sua fonte (tolerância de aprovação 1e-4 relativo;
acima disso entra na lista de investigação, sem reprovar automaticamente):**
- Legado: `FinancialReport` com o token, buscando mercado ao vivo como sempre buscou
  (Upstash desligado → cache em memória). Usar `processar_benchmarks` original.
- Novo: a **API real em execução** (containers da Fase 5), via HTTP:
  `POST /api/v1/auth/login` com usuário/senha em `PARITY_USER`/`PARITY_PASSWORD` (env), o
  token já cadastrado na conta, e chamadas `POST /api/v1/portfolio/*` com o
  `AnalysisRequest` equivalente ao caso. URL base em `PARITY_API_URL`. *(Decisão minha:
  testa o que de fato o usuário usa.)*
- Selenium da B3 no legado: se o download falhar no host (ARM / `webdriver_manager`), usar
  os CSVs já existentes em `data/raw` se o legado os encontrar; senão, retirar os índices
  B3 da camada 2 e registrar a ressalva.
- Cada divergência é classificada pela **origem provável**: calendário/datas (dias a mais ou
  a menos), revisão de cotação na fonte, conversão de câmbio, ou cálculo. Relatório mostra a
  primeira data divergente e a maior diferença por série.

### 5.5. Relatório

- Gravado em `reports/parity/<timestamp>/` (já ignorado pelo git).
- `summary.md`: por caso × métrica × camada → OK/FALHA/INVESTIGAR, maior diferença relativa,
  nº de pontos fora da tolerância, primeira data divergente. **Sem valores em R$.**
- Arquivos de detalhe (CSV/parquet com as séries dos dois lados) para inspeção local; podem
  conter valores da carteira e **nunca** são versionados.
- **Linha de base para a Fase 7:** com a camada 1 provando novo = legado, a Fase 7 compara o
  engine novo **com ele mesmo** (antes × depois), sem precisar de `src/`. Para isso o harness
  salva nesse diretório as **entradas congeladas** de cada caso (histórico DLP + séries de
  mercado brutas usadas na camada 1) e as saídas do engine novo, num formato que dê para
  recarregar só com `backend/` (sem importar `src`). *(Decisão minha: local, nunca
  versionado, por conter dados pessoais.)*

### 5.6. Pendência da Fase 3c

Rodar o teste `integration` de equivalência do filtro DLP (mestre §2.3,
`backend/apps/api/tests/test_integration.py`) com o token real e registrar o resultado no
relatório.

### 5.7. Remoção do legado (só depois do "ok" do usuário no relatório)

Em commits separados do harness:
- Remover `src/`, `redis.py` e o `requirements.txt` da raiz.
- Remover os scripts que importam `src` (`scripts/sync_to_redis.py`,
  `scripts/update_b3_cache.py`, `scripts/verify_paths.py`, `scripts/verify_fix.py`) e o
  próprio `scripts/parity_check.py` (que deixa de funcionar sem `src/`).
- Remover `tests/` da raiz se os testes já estiverem portados em `backend/engine/tests`
  (verificar antes de remover).
- Remover os demais resíduos listados como "Fase 6" na spec da Fase 5 (`ativos.csv`
  duplicado na raiz, scratch do Gemini), conferindo que nada no `backend/`/`frontend/` os usa.
- Desinstalar de `backend/.venv` as libs só do legado (`streamlit`, `altair`, `supabase`,
  `upstash-redis`, `webdriver-manager` e afins), confirmando que a suíte do backend
  (`pytest -q`) e `manage.py check` continuam verdes.

### 5.8. Documentação

- ADR nova registrando o método de paridade (duas camadas, tolerâncias), o resultado
  resumido (sem R$) e as divergências aceitas.
- Atualizar o mestre: §2.0 (Fase 6 concluída), §2.1/§2.2, §5 Fase 6, §8.1 (comando com
  `DLP_TOKEN`), §9 item 11.

## 6. Restrições

- Nenhuma mudança em `backend/engine/` nem em `apps/market/` nesta fase.
- Token e credenciais só em variáveis de ambiente; o harness não os imprime nem os grava.
- Rodar contra o Postgres/Redis do LAB sem escrever dados de mercado novos durante a camada 1
  (leitura apenas).

## 7. Deixado para o builder

- Estrutura interna do harness (módulos, funções, CLI flags além das env vars).
- Mecanismo exato do monkeypatch e da leitura bruta do Postgres para a camada 1.
- Lista final de benchmarks dentro dos critérios do §5.3.
- Formato dos arquivos de detalhe do relatório.

## 8. Decisões minhas (para o usuário revisar)

1. Tabela anual do legado reproduzida literalmente no harness (§5.2).
2. Caso extra com filtro de `classe` (§5.3).
3. Camada 2 do lado novo via API HTTP real dos containers, com login por env vars (§5.4).
4. Tolerâncias: 1e-6 relativo (camada 1) e 1e-4 relativo como limiar de investigação
   (camada 2).
5. Linha de base da Fase 7 = entradas congeladas + saídas do engine novo, guardadas só
   localmente em `reports/parity/` (§5.5).
6. Remoção do legado em commits separados, só após aprovação do relatório (§5.7).
7. Janela "histórico completo" = primeira→última data do DLP nos dois lados (§5.3).
8. Isolamento do Redis/Postgres do LAB dentro do processo do harness (§5.4).
9. Desvio do ADR 0012 tratado como FALHA, exceto se comprovadamente causado por
   `fill_method` (marcado, não silenciado) (§5.4).
