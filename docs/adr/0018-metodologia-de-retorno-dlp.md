# ADR 0018 — Metodologia de cálculo de retorno alinhada à planilha DLP

- **Status:** Aceita (supera a cópia literal temporária do [ADR 0012](0012-engine-copia-literal-fase-2-5.md))
- **Data:** 2026-10-28
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
onde $\text{lucro}_t$ corresponde aos lucros realizados nas vendas e $\text{proventos}_t$ aos dividendos/JCP recebidos no período. O primeiro fluxo é definido como $\text{vlr\_investido}_0 - \text{lucro}_0 - \text{proventos}_0$.

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
- **Método híbrido com garantia de convergência:** tentativa inicial via Newton-Raphson com estimativa inicial baseada no fluxo total e prazo ponderado; caso não convirja em 50 iterações ou resulte em erro matemático, ativa fallback automático para busca por bisseção com expansão adaptativa de limites (bracket), cobrindo taxas reais no intervalo $(-1.0, +\infty)$.

### 4. Sharpe Ratio contra a Série Real do CDI (`engine.metrics`)
- Substituição da taxa fixa ou nula pela série histórica de CDI efetivamente observada no período.
- Retornos excedentes diários calculados contra a taxa diária do CDI correspondente, tanto para a carteira quanto para os benchmarks de mercado.
- Alinhamento pleno com a prática regulatória e de mercado brasileira.

### 5. Recorte Estrito de Janela e Recálculo Pós-Gap (`engine.dataset`)
- Recorte imediato de `portfolio_df` (`grp`) respeitando `date <= end_date`.
- Ao aplicar corte inicial (`start_date`), o `gap = pat_ini - vlr_investido[0]` é adicionado a `vlr_investido`, e o fluxo inicial (`fluxo[0]`) é recalculado como $\text{vlr\_investido}_0 - \text{proventos}_0 - \text{lucro}_0$, eliminando saltos artificiais no histórico de TIR e no saldo inicial.

### 6. Métricas de Retorno Total e Carteira Shadow (`engine.compute`)
- Eliminação da dupla contagem do saldo inicial no `/summary` e na simulação da carteira `/shadow`: o saldo em `start_date` é tratado estritamente como posição inicial, e apenas os fluxos externos subsequentes são aportados/descontados na evolução das cotas dos benchmarks.

---

## Evidências

A validação foi conduzida sem expor valores monetários em R$ da carteira pessoal:

### 1. Oráculo da Planilha DLP (`backend/engine/tests/test_oracle_dlp.py`)
Execução do teste de integração contra o export histórico oficial da planilha DLP (123 meses, julho/2016 a setembro/2026):
- **Fluxo mensal:** 123/123 meses idênticos ao oráculo da planilha (diferença máxima $\le 0{,}01$, decorrente de arredondamento de centavos).
- **TWR mensal:** 122/122 meses coincidentes com erro máximo absoluto $< 0{,}00005$ ($0{,}005\text{ pp}$), bem abaixo da tolerância estrita de $0{,}01\text{ pp}$.
- **TWR acumulado:** **195,95% vs 195,95%** (diferença de $0{,}0032\text{ pp}$, perfeitamente dentro da margem de $0{,}05\text{ pp}$). No legado, o acumulado atingia apenas 40,87%.
- **Conferência da TIR (XIRR):**
  - TIR anual calculada pelo engine sobre a série mensal fechada até set/2026: **12,01% a.a.** (0,95% a.m.).
  - TIR da planilha no snapshot em tempo real de 28/10/2026: **11,10% a.a.** (0,90% a.m.) / **11,15% a.a.** com máxima precisão.
  - A diferença aparente de 0,91 pp decorre de oscilação negativa de mercado de **-2,57%** ocorrida em outubro/2026. Ao projetar os fluxos com a cotação patrimonial de 28/10, a TIR anual do engine recalcula para **11,30% a.a.**, reduzindo a diferença residual a meros **0,15 pp a.a.** (explicada pela consolidação de fluxos mensais vs extrato diário).

### 2. Regressão contra Linha de Base (`scripts/parity/regress.py`)
A execução da ferramenta de regressão contra a linha de base anterior da Fase 6 comprovou:
- Em `twr`, `drawdown`, `rolling_vol` e `yearly`, **apenas a série `Carteira` mudou**; nenhuma coluna de benchmark foi afetada.
- Em `yearly`, os benchmarks `CDI`, `IPCA + 6%`, `S&P 500 BRL` e `IFIX` mantiveram paridade absoluta ($=$).
- As alterações em `rolling_sharpe` e `risk_return` decorreram exclusivamente da adoção da série real do CDI como $r_f$.
- Nova linha de base congelada com 5 casos e 10 métricas em `reports/parity/20261028-163150-fase7/baseline` (verificação subsequente com 100% de paridade `$=$`).

---

## Consequências

- **Mudança nos números reportados:** Os valores de rentabilidade acumulada, TIR, Sharpe e carteira shadow exibidos pela SPA e pela API para a carteira do usuário são alterados, tornando-se matematicamente corretos e alinhados à planilha DLP.
- **Linha de base atualizada:** A linha de base de paridade passa a ser formalmente `reports/parity/20261028-163150-fase7/baseline` (local, não versionada).
- **Encerramento da Fase 7a:** Conclui as correções matemáticas e de confiabilidade do engine de cálculo financeiro.
