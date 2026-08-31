# ADR 0012 — Cópia literal do engine de cálculo puro na Fase 2.5 e revisão pós-Fase 6

- **Status:** Aceita
- **Data:** 2026-08-31
- **Fase:** 2.5 (Refatoração do Engine)

## Contexto

O núcleo de cálculo analítico financeiro (`twr.py`, `irr.py`, `metrics.py`) em `src/engine/` foi construído com NumPy/pandas puros e possui testes unitários pré-existentes. A **Fase 6** do projeto realizará testes estritos de paridade numérica comparando o backend novo com a aplicação legada Streamlit. 

Refatorar ou reescrever a lógica interna dos algoritmos numéricos durante a migração na Fase 2.5 invalidaria a comparação de paridade direta. No entanto, o código atual possui oportunidades claras de melhoria e débitos técnicos conhecidos.

## Decisão

1. **Copiar `twr.py`, `irr.py` e `metrics.py` sem alteração de lógica** para `backend/engine/`.
2. **Única exceção consciente:** explicitar `fill_method=None` em chamadas `pct_change()` em `metrics.py` para alinhar com pandas 2.2 e evitar divergências de regressão.
3. **Extrair funções puras `compute_*` em `compute.py`** a partir dos métodos de plotagem de `financial_report.py`, mantendo a lógica de cálculo estritamente separada da renderização de gráficos.
4. **Isolar Matplotlib em `report_legacy.py`** exclusivamente para exportações CLI e conferência visual, garantindo que a API REST (Fase 3) nunca importe Matplotlib.
5. **Postergar a refatoração profunda dos algoritmos para após a Fase 6**, quando a suíte de paridade estiver ativa e o código legado tiver sido aposentado.

## Pontos Concretos para Revisão Pós-Fase 6

- **`irr.py`:**
  - Implementação atual utiliza Newton-Raphson sem métodos de fallback (ex.: Bisseção ou Brent);
  - Cutoff fixo de tolerância `abs(npv) < 0.1` e chute inicial estático `0.1`;
  - Base de dias `365.0` fixo (não considera anos bissextos na contagem);
  - Clamp rígido `r <= -1.0 → -0.99` que pode oscilar;
  - Não ordena explicitamente as datas dos fluxos.
  - *Ação pós-Fase 6:* Avaliar migração para `scipy.optimize.brentq` ou `numpy_financial.xirr`.
- **`twr.py`:**
  - `np.where(denominador != 0, ...)` sem margem epsilon contra instabilidade de ponto flutuante;
  - `fluxo = vlr_investido.diff().fillna(iloc[0])` assume a primeira linha sempre como aporte limpo.
- **`metrics.py`:**
  - Constantes mágicas (`252`, `rf_constant=0.10`) embutidas no código em vez de parametrizadas por configuração.
- **Pendência de Paridade:**
  - O arquivo legado `src/engine/metrics.py` precisará receber o mesmo `fill_method=None` antes da rodada final de paridade da Fase 6.

## Consequências

### Positivas
- **Paridade garantida:** A validação numérica da Fase 6 permanece uma comparação direta "maçã com maçã".
- **Desacoplamento limpo:** O backend ganha uma camada `compute.py` 100% pura, pronta para servir respostas JSON estruturadas para o frontend React/ECharts da Fase 4.
- **API limpa:** Matplotlib fica confinado a um módulo auxiliar só-CLI, reduzindo pegada de memória e dependências no runtime da API.

### Negativas / Mitigações
- Débitos técnicos nos algoritmos numéricos permanecem temporariamente no código, mitigados pelo registro formal neste ADR para tratamento pós-Fase 6.
