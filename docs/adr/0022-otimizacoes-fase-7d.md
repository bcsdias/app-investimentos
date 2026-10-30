# ADR 0022 — Otimizações da Fase 7d (Memo de Análise, Limpeza do Engine e Code-Splitting do Frontend)

- **Status:** Aceita
- **Data:** 2026-10-30
- **Fase:** 7d (Otimizações sem Mudar Números)

## Contexto

Após a conclusão das correções analíticas das Fases 7a, 7b e 7c (metodologia de retorno da DLP, integridade de mercado e backlog de bugs), a tela analítica da carteira na SPA disparava em paralelo 8 requisições POST para o backend (`summary`, `twr-evolution`, `drawdown`, `rolling-metrics`, `risk-return`, `shadow`, `yearly-summary`, `irr-evolution`).

Cada um desses 8 endpoints executava independentemente o pipeline completo de `run_analysis`:
1. Validação dos parâmetros e autorização por token DLP.
2. Leitura e recorte do histórico do usuário na DLP (`fetch_user_portfolio`).
3. Construção do dataset unificado de mercado (`build_dataset`), envolvendo recuperação e merge das séries de benchmark, detecção de lacunas (`find_gaps`) e verificação de integridade.

Como resultado, o backend realizava 8 vezes o mesmo trabalho computacional a cada carregamento de página. Adicionalmente, medições detalhadas identificaram gargalos pontuais no engine e no empacotamento da SPA:

### Fatos medidos antes da otimização (janela `w1-completo` de 2.950 linhas):
- `run_engine` completo: ~260 ms no ambiente local / ~177 ms na linha de base congelada.
- Detecção de lacunas (`find_gaps` + `stale_tail`): ~33 ms (devido a 15.502 conversões escalares via `_to_date`).
- `calculate_twr`: reagrupava DataFrames que já haviam sido agrupados por data.
- Latência agregada das 8 chamadas no LAB: ~2.350 ms no carregamento (com o cache DLP já aquecido).
- Bundle JavaScript de entrada da SPA (`assets/index-*.js`): 1.416.824 bytes (~1,42 MB uncompressed, chunk único contendo toda a biblioteca ECharts e todas as páginas).

O objetivo central desta fase foi tornar a aplicação expressivamente mais leve e responsiva, mantendo **estritamente zero diferença** numérica contra a linha de base de paridade da Fase 7c e gabarito 100% verde.

---

## Decisões

### 1. Memo do `AnalysisContext` no Redis compartilhado entre os 8 endpoints
- O resultado de `build_dataset` e do contexto analítico passa a ser memoizado via `django.core.cache.cache` com TTL de 300 segundos (`ANALYSIS_TTL = 300`).
- **Composição da chave de cache:**
  `mc.make_key("analysis", user.pk, dlp.user_cache_version(token), mc.series_generation(), _today().isoformat(), _request_digest(valid, req))`
  - `user.pk`: isolamento absoluto entre usuários. Um usuário nunca tem acesso ao contexto de outro.
  - `dlp.user_cache_version(token)`: chave hash do token combinada à geração de cache do usuário. Quando o usuário clica em "Atualizar Dados" (`POST /portfolio/refresh`), a geração é incrementada e o memo é invalidado na hora.
  - `mc.series_generation()`: quando a ingestão de mercado insere dados novos, a geração de séries sobe e invalida os contextos que usam benchmarks desatualizados.
  - `_today().isoformat()`: garante que a virada de dia invalide o contexto, mantendo atualizada a tolerância de cauda parada (`stale_tail`).
  - `_request_digest(...)`: hash SHA-256 (16 caracteres) cobrindo apenas os campos determinantes do dataset (`start`, `end`, `ativo`, `classe`, `benchmarks` ordenados e especificações de índices sintéticos). Parâmetros cosméticos ou específicos de métrica (como `window` de `RollingRequest`) **não** alteram o digest, permitindo que todos os 8 endpoints compartilhem o mesmo cálculo.
- **Single-flight e Tolerância a Falhas:**
  - O cálculo é protegido por `mc.single_flight(key)` com lock atômico no Redis, evitando que as 8 requisições simultâneas disparem 8 computações em paralelo no cold-start da tela.
  - O cache é tratado como otimização: qualquer falha de leitura ou gravação no Redis (`get`/`set`) é logada como `logger.warning` e o cálculo prossegue normalmente, sem propagar erro 500. Erros de validação e regras de negócio (`ApiError`, 404, 422) nunca são memoizados.
- **Armazenamento via Pickle (não `set_df`):**
  - O memo utiliza `cache.set` / `cache.get` nativos do Django (com o backend redis/pickle), sem passar por `mc.set_df`. O formato JSON de tabela do `set_df` arredonda ponto flutuante em 10 casas decimais (`double_precision=10`), o que geraria desvios numéricos sutis contra o critério de tolerância zero.

### 2. Engine sem retrabalho de agrupamento (`engine/flows.py` e `engine/twr.py`)
- Criada a função `flows.is_grouped(df: pd.DataFrame) -> bool`, que valida se o DataFrame possui coluna `date` em formato `datetime64`, monotonicamente crescente e sem duplicatas, com todas as `FLOW_COLUMNS` presentes e numéricas.
- `calculate_twr` detecta se a entrada já atende a esse formato. Se `is_grouped(df)` for verdadeiro, ignora a chamada a `group_portfolio`, reutilizando o DataFrame diretamente e poupando agrupamentos repetidos.

### 3. Date tipado desde a busca da DLP (`apps/market/services/sources/dlp.py`)
- Em `dlp.buscar_historico`, no caminho frio (quando os dados vêm da API da DLP), a coluna `date` é imediatamente convertida para `datetime64[ns]` após a normalização. Assim, tanto o caminho frio quanto o quente (leitura do cache Redis com `orient="table"`) entregam os mesmos tipos sem exigir conversões posteriores no engine.

### 4. Detecção de lacunas vetorizada (`apps/market/gaps.py`)
- Implementada a função `_clean_dates(dates: Iterable) -> pd.DatetimeIndex`, que normaliza, deduplica e ordena datas de forma vetorizada via Pandas/NumPy.
- `find_gaps` substitui a iteração escalar por `np.diff(idx.values).astype("timedelta64[D]").astype(int)`, identificando intervalos maiores que a tolerância sem converter elemento a elemento.

### 5. Frontend com rotas sob demanda e ECharts modular
- **Rotas sob demanda (`React.lazy` + `Suspense`):** As rotas de `/` (`PortfolioPage`) e `/migracao-ir` (`MigrationPage`) passam a ser carregadas assincronamente apenas quando acessadas, com fallback em `Skeleton`. A rota `/login` permanece eager para inicialização rápida.
- **Modularização do ECharts (`echarts/core`):** Criado `frontend/src/components/portfolio/charts/echarts.js`, que registra unicamente os componentes utilizados na aplicação (`LineChart`, `ScatterChart`, `GridComponent`, `TooltipComponent`, `LegendComponent`, `DataZoomComponent`, `MarkLineComponent`, `TitleComponent` e `CanvasRenderer`).
- `BaseChart` migra de `echarts-for-react` para `echarts-for-react/lib/core`, eliminando o empacotamento estático do ECharts completo no chunk de entrada.

---

## Opções Rejeitadas

1. **Chute quente no algoritmo XIRR:**
   - Foi avaliada a possibilidade de utilizar a TIR do mês anterior como chute inicial do Newton-Raphson para acelerar os 122 cálculos de `compute_irr_evolution`.
   - **Rejeitado:** Alterar o chute inicial altera a trajetória de convergência e os últimos bits de precisão do float, produzindo divergências minúsculas contra a linha de base congelada e violando a meta de zero diferença numérica.
2. **Pré-parse de data na serialização de cache para ganho de velocidade:**
   - Foi testado antecipar a conversão de data antes de `mc.set_df`.
   - **Rejeitado como ganho de leitura:** O `orient="table"` do `read_json` já reconstrói tipos datetime nativamente; a alteração foi mantida apenas para coerência de tipos entre caminhos frios e quentes.

---

## Consequências

### Desempenho Medido

| Métrica | Antes | Depois | Redução |
|---|---|---|---|
| **Latência das 8 chamadas no LAB (total)** | 2.349 ms | 522 ms | **−78%** |
| `summary` | 285 ms | 88 ms | −69% |
| `twr-evolution` | 357 ms | 44 ms | −88% |
| `drawdown` | 267 ms | 40 ms | −85% |
| `rolling-metrics` | 297 ms | 82 ms | −72% |
| `risk-return` | 234 ms | 13 ms | −94% |
| `shadow` | 388 ms | 163 ms | −58% |
| `yearly-summary` | 231 ms | 13 ms | −94% |
| `irr-evolution` | 288 ms | 79 ms | −73% |
| **Engine `w1-completo` (medidor)** | 176,8 ms | 172,6 ms | −2,4% |
| `build_dataset` | 81,0 ms | 75,3 ms | −7,0% |
| `fetch_user_portfolio` | 12,4 ms | 10,2 ms | −17,7% |
| **Bundle SPA: Chunk de entrada (`index-*.js`)** | 1.416.824 bytes (~1,42 MB) | 290.105 bytes (~290 kB) | **−79,5%** |
| Menções a `echarts` no chunk de entrada | 4 | 0 | **100% removido** |
| Chunk `MigrationPage` | — | 25.681 bytes (~25 kB) | Sem ECharts |
| Chunk `PortfolioPage` (com gráficos) | — | 610.369 bytes (~610 kB) | −57% vs original |

### Paridade e Confiabilidade
- **Zero diferença numérica:** A suíte de regressão contra a linha de base (`parity.regress`) permaneceu rigorosamente com sinal `=` em todas as 10 métricas ao longo de todas as 6 janelas de caso.
- Testes automatizados do backend: 523 testes passando.
- Testes automatizados do frontend: 107 testes passando.
- Gabarito e oráculo DLP 100% verdes.

### Atenções para Desenvolvedores
- Novos gráficos adicionados no futuro devem registrar seus componentes e tipos de série em `frontend/src/components/portfolio/charts/echarts.js`.
- O memo do `AnalysisContext` deve continuar utilizando o cache nativo (pickle) para evitar perdas de precisão em ponto flutuante.
