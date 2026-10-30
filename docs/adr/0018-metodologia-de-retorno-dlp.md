# ADR 0018 — Metodologia de cálculo de retorno alinhada à planilha DLP

- **Status:** Aceita (supera a cópia literal temporária do [ADR 0012](0012-engine-copia-literal-fase-2-5.md))
- **Data:** 2026-10-28 (revisada em 2026-10-28 após o 1º teste e em 2026-10-30 na Fase 7c — ver §5 e §7 da Decisão e "TIR exibida")
- **Fase:** 7a
- **Spec:** [`../superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md`](../superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md)
- **Plano:** [`../superpowers/plans/2026-10-28-fase-7a-engine.md`](../superpowers/plans/2026-10-28-fase-7a-engine.md)

## Contexto

Durante a migração do Streamlit para Django, o [ADR 0012](0012-engine-copia-literal-fase-2-5.md) determinou a cópia literal dos algoritmos do engine legado (`twr.py`, `irr.py`, `metrics.py`) para viabilizar a validação de paridade da Fase 6 ([ADR 0017](0017-paridade-legado.md)). A paridade foi concluída com sucesso e o código legado (`src/`) foi aposentado.

Contudo, o engine legado continha falhas conceituais graves herdadas no cálculo de rentabilidade da carteira:
1. **Bug do lucro realizado no fluxo de caixa:** O fluxo financeiro era calculado como $\Delta \text{vlr\_investido} - \text{proventos}$, ignorando os lucros realizados nas operações de venda (`lucro_trades` / `lucro`). Em meses com liquidação ou rebalanceamento com lucro, o valor investido sofria redução pelo custo de aquisição enquanto o patrimônio refletia o montante total com lucro. Sem descontar o lucro realizado, o fluxo externo era superestimado (ou entradas eram interpretadas como novos aportes), distorcendo gravemente a rentabilidade ponderada pelo tempo (TWR) e a taxa interna de retorno (XIRR).
2. **Tratamento temporal de aportes e resgates (HPR):** No cálculo do retorno de período de retenção (HPR), saídas de recursos no encerramento do mês eram penalizadas se tratadas na base inicial. A metodologia de referência da planilha DLP considera aportes no **início** do mês e resgates/retiradas no **fim** do mês. A ausência dessa regra fazia com que, no histórico de 10 anos da carteira real, o TWR acumulado legado despencasse para ~41%, quando o oráculo oficial registrava 195,95% (desvios mensais de até 39 pp).
3. **Instabilidade numérica e convenção no XIRR:** O cálculo de TIR em `irr.py` utilizava método de Newton-Raphson puro sem ordenação prévia de datas, sem normalização de escala (sujeito a estouro numérico/overflow), cutoff fixo e sem fallback em caso de divergência.
4. **Taxa Livre de Risco ($r_f$) arbitrária no Sharpe:** O Sharpe ratio móvel e a relação risco × retorno utilizavam taxa constante (ou zero), sem confrontar o retorno excedente da carteira contra a taxa real de CDI observada ao longo do período analisado.
5. **Recortes de janela (`start_date` / `end_date`) e ajuste de `gap`:** Em filtros com `end_date` no passado, `portfolio_df` não era recortado na API, e em filtros com `start_date` intermediário, o ajuste de `gap` corrigia `vlr_investido` mas deixava o primeiro fluxo distorcido, afetando `irr-evolution`, `/summary` e a carteira `/shadow` (que ainda duplicava a contagem do saldo inicial como aporte).

## Decisão

Implementar a metodologia canônica alinhada à planilha oficial DLP nos módulos do pacote `backend/engine/`:

### 1. Fórmula do Fluxo Financeiro Externo (`engine.flows`)
Definir o fluxo de caixa externo líquido como a variação líquida de capital de terceiros/novo aporte:
$$\text{fluxo}_t = \Delta \text{vlr\_investido}_t - \text{lucro}_t - \text{proventos}_t$$
onde $\text{lucro}_t$ corresponde aos lucros realizados nas vendas e $\text{proventos}_t$ aos dividendos/JCP recebidos no período. Na primeira linha do histórico, $\Delta \text{vlr\_investido}_0 = \text{vlr\_investido}_0$. Em recortes com `start_date`, o primeiro fluxo é sobrescrito pelo patrimônio de abertura (§5).

### 2. Regra de Alocação Temporal no TWR Mensal (HPR)
No cálculo do retorno mensal da carteira em `engine.twr.calculate_twr`:
- **Aporte líquido ($\text{fluxo} > 0$):** considerado disponível desde o **início** do mês:
  $$\text{HPR}_t = \frac{\text{pat\_fim}_t}{\text{pat\_ini}_t + \text{fluxo}_t}$$
- **Resgate / Retirada líquida ($\text{fluxo} < 0$):** considerado ocorrido no **fim** do mês (participando do rendimento do mês inteiro):
  $$\text{HPR}_t = \frac{\text{pat\_fim}_t - \text{fluxo}_t}{\text{pat\_ini}_t} = \frac{\text{pat\_fim}_t + |\text{fluxo}_t|}{\text{pat\_ini}_t}$$
- **Fluxo nulo ($\text{fluxo} = 0$):**
  $$\text{HPR}_t = \frac{\text{pat\_fim}_t}{\text{pat\_ini}_t}$$

### 3. XIRR Robusto com Convenção Actual/365 e Fallback (`engine.irr`)
- Convenção anualizada: $(1 + r)^{(d_i - d_0)/365.0}$.
- Ordenação estrita das datas cronológicas e agregação de múltiplos fluxos na mesma data.
- **Invariância à escala:** normalização prévia dos fluxos por $\max(|F_i|)$ antes do cálculo das raízes, evitando overflow de ponto flutuante.
- **Método híbrido com garantia de convergência:** tentativa inicial via Newton-Raphson com convergência relativa à escala dos fluxos; caso não convirja, fallback por intervalo com troca de sinal usando `scipy.optimize.brentq` (Brent), começando em $[-0{,}999999;\ 10^3]$ e expandindo o teto. Sem troca de sinal → `None` (API `null`).
- **Dependência nova:** `scipy~=1.18` em `backend/requirements.txt` (a spec §5.3 permitia `brentq` "se scipy entrar"). A linha entrou no commit de docs `b1b5849`, e não no commit do XIRR `9bc902b`, cuja mensagem fala em "bisseção"; o comportamento correto é o descrito aqui.

### 4. Sharpe Ratio contra a Série Real do CDI (`engine.metrics`)
- Substituição da taxa fixa ou nula pela série histórica de CDI efetivamente observada no período.
- Retornos excedentes diários calculados contra a taxa diária do CDI correspondente, tanto para a carteira quanto para os benchmarks de mercado.
- Alinhamento pleno com a prática regulatória e de mercado brasileira.

### 5. Recorte de Janela, Abertura e Recálculo Pós-Gap (`engine.dataset`)
- Recorte de `portfolio_df` (`grp`) respeitando `date <= end_date`.
- **Abertura no fechamento anterior:** cada linha do `/historico` é um **fechamento mensal**. O período parte do **último fechamento `<= start_date`** (patrimônio de abertura, GIPS), e essa linha entra em `portfolio_df` como base do TWR (1,0) e 1º fluxo do XIRR. Ex.: "01/01/2026 a 28/10/2026" abre em 31/12/2025 e inclui o retorno de janeiro, como o filtro "jan/26–set/26" da DLP. `start_date` igual a um fechamento abre nele mesmo; `start_date` antes do 1º dado abre no 1º dado; nenhum fechamento dentro de `[start_date, end_date]` → `(None, None)`.
- `build_dataset` não corta essa linha de abertura ao recortar o `df_combined` (o corte começa no menor entre `start_date` e a 1ª data da carteira). O "Período analisado" da SPA passa a mostrar a data de abertura.
- Com abertura antes do 1º dado, `gap = vlr_mercado_0 - vlr_investido_0` é somado a `vlr_investido` e `fluxo[0] = vlr_mercado_0` (aporte inicial fictício = patrimônio de abertura).
- O fim do período é o último mês **fechado** que a DLP entrega: com `end_date` = hoje (28/10), a série vai até 30/09.

### 6. Métricas de Retorno Total e Carteira Shadow (`engine.compute`)
- Eliminação da dupla contagem do saldo inicial no `/summary` e na simulação da carteira `/shadow`: o saldo em `start_date` é tratado estritamente como posição inicial, e apenas os fluxos externos subsequentes são aportados/descontados na evolução das cotas dos benchmarks.

### 7. TIR Exibida e Convenção ANBIMA para Períodos Menores que 12 Meses (`engine.irr`, Fase 7c)
Conforme a convenção ANBIMA e práticas prudenciais de mercado, não se divulga rentabilidade anualizada para períodos inferiores a 12 meses (ou 252 dias úteis / 365 dias corridos), pois a exponenciação matemática de horizontes curtos distorce artificialmente as taxas (ex.: primeiro mês negativo amplificado para nominais de −32% a.a. nos primeiros passos da evolução).
- **Cálculo da TIR acumulada do período:** para horizontes com menos de 12 meses desde a abertura, o engine calcula a TIR efetiva do período (sem anualização: $(1 + r_{\text{anual}})^{\Delta t / 365} - 1$).
- **Contrato aditivo na API (`apps.api.schemas`):**
  - `xirr_period` (float | null): rentabilidade percentual efetiva do período analisado.
  - `xirr_is_annual` (bool): indica se a taxa retornada em `xirr` é anualizada (`true` para $\ge 12$ meses, `false` para $< 12$ meses).
  - `annual_from` (string ISO ou null): data exata em que a evolução temporal atinge 12 meses e passa a ser anualizada.
- **Visualização na SPA:**
  - Card de KPIs (`SummaryCards`): exibe "TIR no período" quando `xirr_is_annual === false`.
  - Gráfico de evolução da TIR (`IrrEvolutionChart`): traça marcação visual pontilhada ("a.a. a partir daqui") na data indicada por `annual_from`.

---

## Evidências

A validação foi conduzida sem expor valores monetários em R$ da carteira pessoal:

### 1. Oráculo da Planilha DLP (`backend/engine/tests/test_oracle_dlp.py`)
Execução do teste de integração contra o export histórico oficial da planilha DLP (123 meses, julho/2016 a setembro/2026):
- **Fluxo mensal:** 123/123 meses idênticos ao oráculo da planilha (diferença máxima $\le 0{,}01$, decorrente de arredondamento de centavos).
- **TWR mensal:** 122/122 meses coincidentes com erro máximo absoluto $< 0{,}00005$ ($0{,}005\text{ pp}$), bem abaixo da tolerância estrita de $0{,}01\text{ pp}$.
- **TWR acumulado:** **195,95% vs 195,95%** (diferença de $0{,}0032\text{ pp}$, perfeitamente dentro da margem de $0{,}05\text{ pp}$). No legado, o acumulado atingia apenas 40,87%.
- **Conferência da TIR (XIRR), janela com a mesma data de corte nos dois lados** (substitui a reconciliação inicial, que comparava a série fechada em set/2026 com o dashboard vivo de 28/10):
  - Filtro **jan/26–set/26** no dashboard da DLP (2026-10-28): XTIR **16,72 % a.a.**, TWR acumulado **12,09 %**.
  - Engine com a abertura corrigida, mesma janela: XTIR **16,72 % a.a.**, TWR **12,09 %** (`test_dlp_window_twr_includes_first_month`, `test_dlp_window_xirr_matches_dashboard`, tolerância 0,05 pp).
  - Engine antes da correção da abertura: **9,18 % a.a.** e **5,66 %**, exatamente a DLP sem o mês de janeiro (`1,1209 / 1,0609 − 1 = 5,66 %`).

### 2. Regressão contra Linha de Base (`scripts/parity/regress.py`)
A execução da ferramenta de regressão contra a linha de base anterior da Fase 6 comprovou:
- Em `twr`, `drawdown`, `rolling_vol` e `yearly`, **apenas a série `Carteira` mudou**; nenhuma coluna de benchmark foi afetada.
- Em `yearly`, os benchmarks `CDI`, `IPCA + 6%`, `S&P 500 BRL` e `IFIX` mantiveram paridade absoluta ($=$).
- As alterações em `rolling_sharpe` e `risk_return` decorreram exclusivamente da adoção da série real do CDI como $r_f$.
- Linha de base congelada com 5 casos e 10 métricas em `reports/parity/20261028-163150-fase7/baseline` (verificação subsequente com 100% de paridade `=`).

### 3. Revisão pós-teste (abertura do período)
No 1º teste do usuário no LAB (filtro 01/01/2026–28/10/2026 contra jan–set/2026 na DLP), o TWR e a TIR divergiram (5,66 % × 12,09 %; 9,18 % × 16,72 % a.a.). Causa: o recorte usava o 1º fechamento **dentro** do período como abertura e descartava o retorno do 1º mês. Escapou porque o oráculo só cobria a janela completa e todas as janelas da matriz de paridade começavam numa data de fechamento (o teste de `gap` da Task 2 usava `start_date` = fechamento).
- Correção em `engine.dataset` (§5 da Decisão), com testes unitários (`test_start_date_mid_month_opens_at_previous_close`, `test_build_dataset_keeps_opening_close_before_start_date`) e o oráculo de janela.
- Matriz de paridade ganhou `w5-inicio-mes` (`start` no dia seguinte a um fechamento). Regressão contra `20261028-163150-fase7`: os 5 casos existentes `=` em todas as métricas (todos começam num fechamento); `w5-inicio-mes` novo.
- Linha de base vigente: `reports/parity/20261028-180012-fase7-abertura/baseline` (6 casos, re-verificada com tudo `=`).

### 4. TIR exibida (Fase 7c, Task 3b)
No teste do usuário na Fase 7b, notou-se que o gráfico de "Evolução Histórica da TIR" apresentava uma queda inicial drástica para −32% a.a. no primeiro mês de retenção.
- A aplicação da convenção ANBIMA (§7 da Decisão) elimina a taxa anualizada espúria para períodos inferiores a 12 meses, apresentando a TIR acumulada do período.
- A série histórica mantém integridade temporal, sinalizando a transição para taxa anualizada (`a.a.`) a partir do 12º mês via `annual_from`, com campos aditivos `xirr_period` e `xirr_is_annual`.

---

## Consequências

- **Mudança nos números reportados:** Os valores de rentabilidade acumulada, TIR, Sharpe e carteira shadow exibidos pela SPA e pela API para a carteira do usuário são alterados, tornando-se matematicamente corretos e alinhados à planilha DLP.
- **Convenção de TIR:** Janelas inferiores a 12 meses passam a exibir a TIR do período acumulada em vez de anualizada, evitando distorções em períodos curtos sem quebrar o contrato da API v1.
- **Linha de base atualizada:** A linha de base de paridade passa a ser `reports/parity/20261028-180012-fase7-abertura/baseline` (local, não versionada), que substitui `20261028-163150-fase7`.
- **Períodos recortados:** todo período com `start_date` fora de um fechamento (inclusive os atalhos 1A/2A/5A da SPA) passa a incluir o 1º mês; o "1A" cobre 12 meses, não 11.
- **Encerramento da Fase 7a:** Conclui as correções matemáticas e de confiabilidade do engine de cálculo financeiro.
