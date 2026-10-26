# Fase 4 — Frontend React SPA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir a aplicação cliente completa em React 18 SPA (`frontend/`), substituindo o protótipo legado em Streamlit, integrada à API Django Ninja (`/api/v1/`) com autenticação por sessão, gestão de token DLP, dashboard analítico da carteira (Performance & Risco) e planejador de migração com isenção de IR e download de PDF.

**Architecture:** Single Page Application construída com Vite, React 18 e Tailwind CSS. Gerenciamento de estado dividido entre React Context (sessão de auth e filtros ativos de período/benchmarks) e TanStack Query (camada de dados assíncronos com cache, deduplicação e cancelamento automático via `AbortController` de 8 queries concorrentes). Gráficos interativos renderizados com Apache ECharts (`echarts-for-react`) em tema escuro nativo. Comunicação via cliente Axios centralizado com `withCredentials: true`, injeção automática de `X-CSRFToken` e intercepção de erros 409/422 da DLP.

**Tech Stack:** React 18, Vite 5, Tailwind CSS 3, Lucide React, Apache ECharts 5, `echarts-for-react`, `@tanstack/react-query` 5, Axios, `react-router-dom` 6, Vitest, `@testing-library/react`, `@testing-library/jest-dom`, `jsdom`.

**Spec:** [`docs/superpowers/specs/2026-10-26-fase-4-frontend-react-design.md`](../specs/2026-10-26-fase-4-frontend-react-design.md)

---

## Global Constraints

Todo task herda implicitamente esta seção:

- **Branch:** `migracao-django`. Código e testes do frontend residem estritamente em `frontend/`.
- **Node e npm:** Node v22.x e npm v10.x já disponíveis no ambiente do host.
- **Backend intocado:** O backend em `backend/` está concluído e validado (329+ testes verdes); nenhum arquivo em `backend/` deve ser alterado nesta fase.
- **Comunicação com a API:** Base URL `/api/v1`, requisições com `withCredentials: true`. Em desenvolvimento, `vite.config.js` faz proxy de `/api/v1` e `/admin` para `http://localhost:8000`.
- **CSRF:** Todas as mutações (`POST`, `PUT`, `PATCH`, `DELETE`) devem enviar o cabeçalho `X-CSRFToken` lido do cookie `csrftoken` (gerado após `GET /api/v1/auth/csrf`).
- **Tratamento de Sessão e DLP:**
  - HTTP 401 redireciona para `/login` (exceto na própria página `/login`).
  - HTTP 409 (`dlp_token_missing`) e HTTP 422 (`dlp_token_rejected`) disparam a abertura do `DlpTokenModal` via evento/estado global, **sem deslogar o usuário**.
  - HTTP 404 (`portfolio_empty`) exibe componente `EmptyState` instrucional, sem quebrar a UI.
- **Contrato de Filtros:** Payload analítico (`AnalysisRequest`) usa estritamente `start` e `end` (strings `YYYY-MM-DD` ou `null`), `ativo`, `classe`, `benchmarks` (lista de `yf_ticker`) e `synthetics` (lista de `{base, operador, valor}`). O campo `corretora` **não** é enviado.
- **Gráficos ECharts:** Séries temporais no formato nativo `[["YYYY-MM-DD", valor | null], ...]`. Tratar valores `null` sem erros de renderização ou quebra de tooltip.
- **TDD e Qualidade:** Todo task deve ter testes automatizados com Vitest + Testing Library. Commits atômicos no formato Conventional Commits em português ao final de cada task.

---

## Review Focus

Cenários de borda e modos de falha implícitos no spec que um usuário real encontrará:

1. **Token DLP ausente ou rejeitado ao carregar a carteira** (API retorna 409 `dlp_token_missing` ou 422 `dlp_token_rejected`) -> A UI intercepta o erro, exibe alerta explicativo e abre o modal de token sem deslogar o usuário (Task 2 e Task 3).
2. **Mudança rápida de período/filtros antes do término do carregamento** -> TanStack Query repassa o `signal` do `AbortController` cancelando as requisições em voo anteriores, impedindo que dados obsoletos substituam o novo estado (Task 6).
3. **Métricas móveis com valores nulos no início do histórico** (primeiros 252 pontos são `null` em Sharpe e Volatilidade) -> ECharts plota a linha a partir do primeiro ponto não nulo sem quebrar o cálculo de min/max dos eixos nem o tooltip (Task 5 e Task 7).
4. **Alocação-alvo com soma dos pesos diferente de 100% no Planejador de IR** -> Validação visual instantânea no formulário bloqueando a simulação e destacando a diferença percentual (Task 8).
5. **Carteira sem movimentações na DLP (HTTP 404 `portfolio_empty`)** -> `HoldingsTable` e gráficos exibem um `EmptyState` amigável com orientações em vez de tela branca ou toast genérico de erro (Task 8).

---

## Estrutura de Arquivos

Tudo dentro de `frontend/`:

```
frontend/
├── package.json
├── vite.config.js
├── vitest.config.js
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── index.css
│   ├── test/
│   │   ├── setup.js
│   │   └── mockData.js
│   ├── api/
│   │   ├── client.js
│   │   ├── auth.js
│   │   ├── dlp.js
│   │   ├── portfolio.js
│   │   ├── benchmarks.js
│   │   ├── assets.js
│   │   └── migration.js
│   ├── context/
│   │   ├── AuthContext.jsx
│   │   └── PortfolioFilterContext.jsx
│   ├── hooks/
│   │   ├── usePortfolioQueries.js
│   │   └── useMigrationPlan.js
│   ├── components/
│   │   ├── common/
│   │   │   ├── Button.jsx
│   │   │   ├── Card.jsx
│   │   │   ├── Modal.jsx
│   │   │   ├── Badge.jsx
│   │   │   ├── Skeleton.jsx
│   │   │   ├── Tabs.jsx
│   │   │   ├── EmptyState.jsx
│   │   │   └── RequireAuth.jsx
│   │   ├── layout/
│   │   │   ├── Header.jsx
│   │   │   └── Layout.jsx
│   │   ├── dlp/
│   │   │   └── DlpTokenModal.jsx
│   │   ├── portfolio/
│   │   │   ├── FiltersBar.jsx
│   │   │   ├── BenchmarkSelector.jsx
│   │   │   ├── SummaryCards.jsx
│   │   │   ├── charts/
│   │   │   │   ├── BaseChart.jsx
│   │   │   │   ├── TwrChart.jsx
│   │   │   │   ├── DrawdownChart.jsx
│   │   │   │   ├── RollingSharpeChart.jsx
│   │   │   │   ├── ShadowChart.jsx
│   │   │   │   └── RiskReturnChart.jsx
│   │   │   └── tables/
│   │   │       └── YearlyTable.jsx
│   │   └── migration/
│   │       ├── HoldingsTable.jsx
│   │       ├── TargetAllocationForm.jsx
│   │       ├── MonthlyPurchasesTable.jsx
│   │       └── ExportPdfButton.jsx
│   └── pages/
│       ├── LoginPage.jsx
│       ├── PortfolioPage.jsx
│       └── MigrationPage.jsx
```

---

## Task 1: Bootstrap do Projeto Vite, Tailwind CSS e Ambiente de Testes

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.js`, `frontend/vitest.config.js`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/index.html`, `frontend/src/index.css`, `frontend/src/test/setup.js`, `frontend/src/test/smoke.test.jsx`

**Interfaces:**
- Produces: Ambiente de execução Vite com React 18, Tailwind CSS (Zinc/Emerald) e suíte de testes Vitest com `@testing-library/react` e `jsdom`.

- [ ] **Step 1: Criar `frontend/package.json` com todas as dependências pinadas**
  Dependências: `react` (^18.3.1), `react-dom` (^18.3.1), `react-router-dom` (^6.28.0), `axios` (^1.7.9), `@tanstack/react-query` (^5.62.0), `echarts` (^5.5.1), `echarts-for-react` (^3.0.2), `lucide-react` (^0.468.0).
  DevDependencies: `vite` (^6.0.3), `@vitejs/plugin-react` (^4.3.4), `tailwindcss` (^3.4.16), `postcss` (^8.4.49), `autoprefixer` (^10.4.20), `vitest` (^2.1.8), `@testing-library/react` (^16.1.0), `@testing-library/jest-dom` (^6.6.3), `@testing-library/user-event` (^14.5.2), `jsdom` (^25.0.1).
  Scripts: `"dev": "vite"`, `"build": "vite build"`, `"preview": "vite preview"`, `"test": "vitest run"`, `"test:watch": "vitest"`.

- [ ] **Step 2: Configurar `vite.config.js`, `vitest.config.js`, `tailwind.config.js`, `postcss.config.js` e `index.html`**
  - `vite.config.js`: plugin `@vitejs/plugin-react`, server proxy de `/api/v1` e `/admin` para `http://localhost:8000`.
  - `vitest.config.js`: `globals: true`, `environment: 'jsdom'`, `setupFiles: './src/test/setup.js'`.
  - `tailwind.config.js`: `darkMode: 'class'`, content em `./index.html` e `./src/**/*.{js,jsx}`, cores tema escuro (`zinc-900`, `zinc-800`, `emerald-500`, etc.).
  - `src/index.css`: `@tailwind base; @tailwind components; @tailwind utilities;` + reset de scrollbar.
  - `src/test/setup.js`: `import '@testing-library/jest-dom';` + mocks para `matchMedia` e `ResizeObserver`.

- [ ] **Step 3: Escrever o teste de fumaça (smoke test) `frontend/src/test/smoke.test.jsx`**

```jsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';

describe('Frontend Environment Smoke Test', () => {
  it('renders a simple element into jsdom with tailwind class', () => {
    render(<div className="bg-zinc-900 text-white">App Investimentos</div>);
    const element = screen.getByText('App Investimentos');
    expect(element).toBeInTheDocument();
    expect(element).toHaveClass('bg-zinc-900');
  });
});
```

- [ ] **Step 4: Instalar dependências e rodar teste de fumaça e build**
  Executar: `cd frontend && npm install && npm test`
  Resultado esperado: `1 passed`
  Executar: `npm run build`
  Resultado esperado: Build Vite concluído gerando pasta `dist/` sem erros de sintaxe.

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): inicializar projeto Vite com Tailwind CSS e Vitest"
```

---

## Task 2: Cliente HTTP Axios, Proteção CSRF, AuthContext e Interceptores DLP

**Files:**
- Create: `frontend/src/api/client.js`, `frontend/src/api/auth.js`, `frontend/src/api/dlp.js`, `frontend/src/context/AuthContext.jsx`, `frontend/src/components/common/RequireAuth.jsx`
- Test: `frontend/src/test/api-auth.test.jsx`

**Interfaces:**
- Produces:
  - `api` (`AxiosInstance`): intercepta request para injetar `X-CSRFToken` e intercepta response para tratar 401 (redirect `/login`) e 409/422 (disparar evento customizado `open-dlp-modal`).
  - `authApi`: `getCsrf()`, `login({ username, password })`, `logout()`, `getMe()`.
  - `dlpApi`: `getTokenStatus()`, `saveToken(token)`, `deleteToken()`.
  - `AuthContext`: provê `{ user, isAuthenticated, isLoading, login, logout, checkAuth }`.
  - `<RequireAuth>`: componente wrapper que renderiza `<Outlet />` se autenticado ou redireciona para `/login`.

- [ ] **Step 1: Escrever teste automatizado para o cliente Axios, interceptores e AuthContext**
  Arquivo: `frontend/src/test/api-auth.test.jsx`.
  Testar:
  - Injeção de `X-CSRFToken` ao chamar método `post` quando o cookie `csrftoken` existe.
  - Interceptação de erro 409 (`dlp_token_missing`) dispara o evento window `open-dlp-modal` com detalhe `reason: "missing"`.
  - Interceptação de erro 422 (`dlp_token_rejected`) dispara o evento window `open-dlp-modal` com detalhe `reason: "rejected"`.
  - `AuthContext` atualiza `user` ao chamar `login()` com sucesso e limpa ao chamar `logout()`.

- [ ] **Step 2: Rodar teste para verificar que falha**
  Executar: `cd frontend && npm test src/test/api-auth.test.jsx`
  Resultado esperado: FAIL ("cannot find module api/client")

- [ ] **Step 3: Implementar `client.js`, `auth.js`, `dlp.js`, `AuthContext.jsx` e `RequireAuth.jsx`**
  - `client.js`: Criar instância com `baseURL: '/api/v1'`, `withCredentials: true`. Interceptor de request busca cookie `csrftoken` e insere `X-CSRFToken` para métodos de escrita. Interceptor de response redireciona 401 para `/login` (quando fora de `/login`) e dispara evento customizado para 409/422.
  - `auth.js`: Funções REST encapsulando `GET /auth/csrf`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`.
  - `dlp.js`: Funções REST para `GET /dlp/token-status`, `POST /dlp/token`, `DELETE /dlp/token`.
  - `AuthContext.jsx`: Hook `useAuth()` e provider com carregamento assíncrono inicial chamando `getMe()` e `getCsrf()`.
  - `RequireAuth.jsx`: Verifica `isLoading` (renderiza loader) e se `!isAuthenticated` redireciona para `<Navigate to="/login" replace />`.

- [ ] **Step 4: Rodar teste para verificar que passa**
  Executar: `cd frontend && npm test src/test/api-auth.test.jsx`
  Resultado esperado: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/ frontend/src/context/AuthContext.jsx frontend/src/components/common/RequireAuth.jsx frontend/src/test/api-auth.test.jsx
git commit -m "feat(frontend): implementar cliente Axios com CSRF, AuthContext e interceptor DLP"
```

---

## Task 3: Primitivas de UI, Topbar (Header), Layout e Modal de Token DLP

**Files:**
- Create: `frontend/src/components/common/{Button,Card,Modal,Badge,Skeleton,Tabs,EmptyState}.jsx`, `frontend/src/components/layout/Header.jsx`, `frontend/src/components/layout/Layout.jsx`, `frontend/src/components/dlp/DlpTokenModal.jsx`
- Test: `frontend/src/test/header-dlp.test.jsx`

**Interfaces:**
- Consumes: `useAuth()`, `dlpApi`, `api.portfolio.refreshPortfolio`.
- Produces:
  - `<Header />`: Topbar com branding, indicador de status DLP (`configured: true/false`), botão de sincronização ("Atualizar Dados"), perfil e botão de logout.
  - `<DlpTokenModal isOpen onClose />`: Modal interativo para salvar ou remover token DLP com máscara de exibição e feedback de status.
  - Primitivas reutilizáveis: `Button`, `Card`, `Modal`, `Badge`, `Skeleton`, `Tabs`, `EmptyState`.

- [ ] **Step 1: Escrever teste automatizado para Header e DlpTokenModal**
  Arquivo: `frontend/src/test/header-dlp.test.jsx`.
  Testar:
  - Header exibe badge verde quando `configured: true` e amarelo de alerta quando `configured: false`.
  - Clique no badge de status ou recebimento do evento `open-dlp-modal` abre o modal de token.
  - Submissão com token preenchido chama `dlpApi.saveToken` e atualiza o estado para `configured: true`.
  - Botão "Atualizar Dados" chama `POST /api/v1/portfolio/refresh` e aciona invalidação de queries do TanStack Query.

- [ ] **Step 2: Rodar teste para verificar que falha**
  Executar: `cd frontend && npm test src/test/header-dlp.test.jsx`
  Resultado esperado: FAIL ("cannot find module components/layout/Header")

- [ ] **Step 3: Implementar componentes visuais primitivos, Header e DlpTokenModal**
  - Implementar primitivas em `components/common/` com design system Tailwind (tema escuro, focus rings acessíveis).
  - Implementar `DlpTokenModal.jsx`: formulário com input protegido (`type="password"` com botão de revelar), validação de string não vazia, chamada a `POST /dlp/token` e botão opcional de remover token via `DELETE /dlp/token`.
  - Implementar `Header.jsx`: consome `useQuery` de `/dlp/token-status` para exibir badge com `updated_at` formatado, botão com ícone `RefreshCw` da Lucide animado durante o carregamento de `POST /portfolio/refresh`, links de navegação para `/` e `/migracao-ir`.
  - Implementar `Layout.jsx`: envoltório padrão da aplicação contendo `<Header />` e container responsivo principal (`max-w-7xl mx-auto px-4 py-6`).

- [ ] **Step 4: Rodar teste para verificar que passa**
  Executar: `cd frontend && npm test src/test/header-dlp.test.jsx`
  Resultado esperado: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/
git commit -m "feat(frontend): implementar componentes base, Header, Layout e DlpTokenModal"
```

---

## Task 4: Estado dos Filtros da Carteira e Barra de Controles (`PortfolioFilterContext`, `FiltersBar` e `BenchmarkSelector`)

**Files:**
- Create: `frontend/src/api/benchmarks.js`, `frontend/src/api/assets.js`, `frontend/src/api/portfolio.js`, `frontend/src/context/PortfolioFilterContext.jsx`, `frontend/src/components/portfolio/FiltersBar.jsx`, `frontend/src/components/portfolio/BenchmarkSelector.jsx`
- Test: `frontend/src/test/filters-bar.test.jsx`

**Interfaces:**
- Consumes: `GET /api/v1/benchmarks/catalog`, `GET /api/v1/assets/search`, `GET /api/v1/portfolio/filters`.
- Produces:
  - `PortfolioFilterContext`: provê `{ period, setPeriod, start, end, setCustomDates, benchmarks, addBenchmark, removeBenchmark, synthetics, addSynthetic, removeSynthetic, ativo, setAtivo, classe, setClasse, resetFilters }`.
  - `<FiltersBar />`: Barra horizontal com botões de período (`1A`, `2A`, `5A`, `Tudo`, `Personalizado`), seletores de `ativo` e `classe`, e o seletor de benchmarks.
  - `<BenchmarkSelector />`: Dropdown multiselect para índices do catálogo, busca de tickers (`assets/search`) com opção "Versão em BRL", e modal de criação de índice sintético (`{base, operador, valor}`).

- [ ] **Step 1: Escrever teste automatizado para PortfolioFilterContext e FiltersBar**
  Arquivo: `frontend/src/test/filters-bar.test.jsx`.
  Testar:
  - Selecionar `1A` calcula `start` de 365 dias atrás e `end` atual (formato ISO `YYYY-MM-DD`).
  - Selecionar `Tudo` define `start: null` e `end: null`.
  - Adicionar benchmark do catálogo insere a string na lista `benchmarks`.
  - Buscar ativo e selecionar com BRL marcado adiciona `${yf_ticker} BRL` (ex: `AAPL BRL`).
  - Adicionar sintético insere objeto `{base: "CDI", operador: "+", valor: 2}` na lista `synthetics`.

- [ ] **Step 2: Rodar teste para verificar que falha**
  Executar: `cd frontend && npm test src/test/filters-bar.test.jsx`
  Resultado esperado: FAIL ("cannot find module context/PortfolioFilterContext")

- [ ] **Step 3: Implementar chamadas de API, Contexto de Filtros e Componentes de Controle**
  - `api/benchmarks.js`: `getCatalog()` chamando `GET /benchmarks/catalog`.
  - `api/assets.js`: `searchAssets(query)` chamando `GET /assets/search?q=${encodeURIComponent(query)}`.
  - `api/portfolio.js`: `getFilters()` chamando `GET /portfolio/filters`.
  - `PortfolioFilterContext.jsx`: cálculo automático de datas para presets `1A`, `2A`, `5A`, estado de `benchmarks` (default `['CDI', '^BVSP']` ou carregado do catálogo), `synthetics`, `ativo`, `classe`.
  - `BenchmarkSelector.jsx`: popover com abas "Catálogo", "Buscar Ticker" (com debounce de 300ms) e "Sintético" (validação de número positivo e limites: `valor <= 50` para `+` e `valor <= 500` para `%`).
  - `FiltersBar.jsx`: botões estilizados com visual pill ativo/inativo, inputs de data quando em modo "Personalizado", e badges para filtros ativos com botão de remoção rápida.

- [ ] **Step 4: Rodar teste para verificar que passa**
  Executar: `cd frontend && npm test src/test/filters-bar.test.jsx`
  Resultado esperado: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/benchmarks.js frontend/src/api/assets.js frontend/src/api/portfolio.js frontend/src/context/PortfolioFilterContext.jsx frontend/src/components/portfolio/
git commit -m "feat(frontend): implementar contexto de filtros, FiltersBar e BenchmarkSelector"
```

---

## Task 5: Cards de Resumo e Integração com ECharts (`SummaryCards` e `BaseChart`)

**Files:**
- Create: `frontend/src/components/portfolio/SummaryCards.jsx`, `frontend/src/components/portfolio/charts/BaseChart.jsx`
- Test: `frontend/src/test/summary-cards-chart.test.jsx`

**Interfaces:**
- Consumes: `POST /api/v1/portfolio/summary`, `echarts-for-react`.
- Produces:
  - `<SummaryCards data={summaryData} isLoading={isLoading} isError={isError} />`: Exibe 5 cards (Patrimônio Fim, Total Aportado, Ganho de Capital, Rentabilidade TWR e XIRR Anualizada).
  - `<BaseChart option={option} height={height} isLoading={isLoading} />`: Wrapper seguro sobre `ReactECharts` com suporte a redimensionamento responsivo, tema escuro zinc, tooltips customizados e visual de loading.

- [ ] **Step 1: Escrever teste automatizado para SummaryCards e BaseChart**
  Arquivo: `frontend/src/test/summary-cards-chart.test.jsx`.
  Testar:
  - `SummaryCards` exibe skeletons quando `isLoading: true`.
  - `SummaryCards` formata valores monetários em BRL (`R$ 150.000,00`) e percentuais com sinal (`+15,40%` em verde, `-2,10%` em vermelho).
  - `BaseChart` repassa as opções para o ECharts aplicando o tema escuro e habilitando o resize observer.

- [ ] **Step 2: Rodar teste para verificar que falha**
  Executar: `cd frontend && npm test src/test/summary-cards-chart.test.jsx`
  Resultado esperado: FAIL ("cannot find module components/portfolio/SummaryCards")

- [ ] **Step 3: Implementar `SummaryCards.jsx` e `BaseChart.jsx`**
  - `SummaryCards.jsx`: Grade responsiva (1 coluna no mobile, 5 no desktop). Mapeia campos do backend: `pat_fim`, `aportes`, `ganho`, `twr` (multiplicado por 100 para percentual), `xirr` (multiplicado por 100).
  - `BaseChart.jsx`: Utiliza `ReactECharts` importado de `echarts-for-react`. Configura cores padrão (paleta: esmeralda `#10b981`, azul `#3b82f6`, âmbar `#f59e0b`, roxo `#8b5cf6`, vermelho `#ef4444`), fonte Inter/sans-serif, fundo transparente, tooltip com `trigger: 'axis'` e `backgroundColor: '#18181b'`, `borderColor: '#27272a'`.

- [ ] **Step 4: Rodar teste para verificar que passa**
  Executar: `cd frontend && npm test src/test/summary-cards-chart.test.jsx`
  Resultado esperado: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/portfolio/SummaryCards.jsx frontend/src/components/portfolio/charts/BaseChart.jsx frontend/src/test/summary-cards-chart.test.jsx
git commit -m "feat(frontend): implementar SummaryCards de métricas e BaseChart ECharts"
```

---

## Task 6: Hook de Queries Analíticas e Gráficos da Aba Performance (`TwrChart`, `YearlyTable`, `ShadowChart`, `IrrEvolutionChart`)

**Files:**
- Create: `frontend/src/hooks/usePortfolioQueries.js`, `frontend/src/components/portfolio/charts/TwrChart.jsx`, `frontend/src/components/portfolio/tables/YearlyTable.jsx`, `frontend/src/components/portfolio/charts/ShadowChart.jsx`, `frontend/src/components/portfolio/charts/IrrEvolutionChart.jsx`
- Test: `frontend/src/test/performance-charts.test.jsx`

**Interfaces:**
- Consumes: `usePortfolioFilters()`, `@tanstack/react-query`, endpoints `/portfolio/{summary, twr-evolution, shadow, yearly-summary, irr-evolution}`.
- Produces:
  - `usePortfolioQueries()`: dispara simultaneamente as queries analíticas com o payload `{start, end, benchmarks, synthetics, ativo, classe}`, retornando o mapa de queries `{ summary, twr, drawdown, rolling, riskReturn, shadow, yearly, irr }`.
  - `<TwrChart data={twrData} isLoading={...} />`: Gráfico com cota base 100 de carteira vs benchmarks com slider de zoom no rodapé (`dataZoom`).
  - `<YearlyTable data={yearlyData} isLoading={...} />`: Tabela de rentabilidade anual com linhas para anos e colunas para Carteira e benchmarks selecionados.
  - `<ShadowChart data={shadowData} isLoading={...} />`: Gráfico de evolução patrimonial comparada.
  - `<IrrEvolutionChart data={irrData} isLoading={...} />`: Curva histórica da taxa interna de retorno.

- [ ] **Step 1: Escrever teste automatizado para `usePortfolioQueries` e componentes de Performance**
  Arquivo: `frontend/src/test/performance-charts.test.jsx`.
  Testar:
  - `usePortfolioQueries` propaga o `AbortController.signal` para as chamadas Axios.
  - `TwrChart` monta a configuração ECharts com série `time` e formata tooltip com base 100.
  - `YearlyTable` renderiza cabeçalho com anos e células com percentuais coloridos (verde positivo, vermelho negativo).
  - Valores nulos em séries temporais são repassados ao ECharts sem quebrar.

- [ ] **Step 2: Rodar teste para verificar que falha**
  Executar: `cd frontend && npm test src/test/performance-charts.test.jsx`
  Resultado esperado: FAIL ("cannot find module hooks/usePortfolioQueries")

- [ ] **Step 3: Implementar hook `usePortfolioQueries` e os 4 componentes de Performance**
  - `usePortfolioQueries.js`: Usa `useQueries` do TanStack Query. Define `staleTime: 5 * 60 * 1000` e repassa `signal`. Cria funções em `api/portfolio.js` para cada endpoint analítico (`fetchSummary`, `fetchTwrEvolution`, `fetchShadow`, `fetchYearlySummary`, `fetchIrrEvolution`, `fetchDrawdown`, `fetchRollingMetrics`, `fetchRiskReturn`).
  - `TwrChart.jsx`: Configura gráfico de linha com `xAxis.type = 'time'`, `dataZoom` inferior, tooltip formatando valores em índice (ex: `142.50 (+42.50%)`).
  - `YearlyTable.jsx`: Tabela estilizada em Tailwind com cabeçalho fixo, linhas zebradas sutis e badges coloridos para retorno anual.
  - `ShadowChart.jsx`: Gráfico de área/linha monetária com tooltip formatando em R$.
  - `IrrEvolutionChart.jsx`: Gráfico de linha temporal com percentual anualizado no eixo Y.

- [ ] **Step 4: Rodar teste para verificar que passa**
  Executar: `cd frontend && npm test src/test/performance-charts.test.jsx`
  Resultado esperado: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/usePortfolioQueries.js frontend/src/api/portfolio.js frontend/src/components/portfolio/charts/TwrChart.jsx frontend/src/components/portfolio/tables/YearlyTable.jsx frontend/src/components/portfolio/charts/ShadowChart.jsx frontend/src/components/portfolio/charts/IrrEvolutionChart.jsx frontend/src/test/performance-charts.test.jsx
git commit -m "feat(frontend): implementar usePortfolioQueries e gráficos da aba Performance"
```

---

## Task 7: Gráficos da Aba Risco e Página do Dashboard (`DrawdownChart`, `RollingSharpeChart`, `RiskReturnChart`, `PortfolioPage`)

**Files:**
- Create: `frontend/src/components/portfolio/charts/DrawdownChart.jsx`, `frontend/src/components/portfolio/charts/RollingSharpeChart.jsx`, `frontend/src/components/portfolio/charts/RiskReturnChart.jsx`, `frontend/src/pages/PortfolioPage.jsx`
- Test: `frontend/src/test/risk-charts-page.test.jsx`

**Interfaces:**
- Consumes: `usePortfolioQueries()`, endpoints `/portfolio/{drawdown, rolling-metrics, risk-return}`.
- Produces:
  - `<DrawdownChart data={...} isLoading={...} />`: Gráfico de perdas percentuais com área avermelhada subaquática (eixo Y variando de -100% a 0%).
  - `<RollingSharpeChart data={...} isLoading={...} />`: Gráfico com dois eixos Y (Sharpe Ratio no Y1, Volatilidade anualizada no Y2).
  - `<RiskReturnChart data={...} isLoading={...} />`: Scatter plot Risco (Volatilidade anualizada %) no eixo X e Retorno (CAGR %) no eixo Y com quadrantes de dispersão.
  - `<PortfolioPage />`: Página principal agregando `FiltersBar`, `SummaryCards`, abas `Performance` e `Risco`.

- [ ] **Step 1: Escrever teste automatizado para os gráficos de Risco e a PortfolioPage**
  Arquivo: `frontend/src/test/risk-charts-page.test.jsx`.
  Testar:
  - `DrawdownChart` formata o eixo Y em percentual negativo (`-15.2%`).
  - `RollingSharpeChart` suporta pontos iniciais `null` sem renderizar valores quebrados.
  - `RiskReturnChart` monta série do tipo `scatter` associando nome do ativo/benchmark ao ponto.
  - `PortfolioPage` renderiza os componentes da aba Performance por padrão e troca para a aba Risco ao clicar sem disparar recargas na rede se os dados já estiverem em cache.

- [ ] **Step 2: Rodar teste para verificar que falha**
  Executar: `cd frontend && npm test src/test/risk-charts-page.test.jsx`
  Resultado esperado: FAIL ("cannot find module components/portfolio/charts/DrawdownChart")

- [ ] **Step 3: Implementar gráficos de Risco e a PortfolioPage**
  - `DrawdownChart.jsx`: ECharts com `areaStyle` em gradiente vermelho translúcido (`rgba(239, 68, 68, 0.25)`), linha `#ef4444`, `max: 0`.
  - `RollingSharpeChart.jsx`: ECharts dual-axis (eixo esquerdo para Sharpe Ratio com linha de referência zero, eixo direito para Volatilidade em %).
  - `RiskReturnChart.jsx`: Scatter plot com tooltips contendo `{name, vol, cagr}` formatados e linhas de média para criar quadrantes visuais.
  - `PortfolioPage.jsx`: Layout estruturado com `<FiltersBar />`, `<SummaryCards />`, seletor de abas `<Tabs>` (Performance e Risco). Ambos os grupos de gráficos são mantidos no DOM ou ativados sem recarregar estado, consumindo o hook `usePortfolioQueries()`.

- [ ] **Step 4: Rodar teste para verificar que passa**
  Executar: `cd frontend && npm test src/test/risk-charts-page.test.jsx`
  Resultado esperado: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/portfolio/charts/DrawdownChart.jsx frontend/src/components/portfolio/charts/RollingSharpeChart.jsx frontend/src/components/portfolio/charts/RiskReturnChart.jsx frontend/src/pages/PortfolioPage.jsx frontend/src/test/risk-charts-page.test.jsx
git commit -m "feat(frontend): implementar gráficos de Risco e Dashboard PortfolioPage"
```

---

## Task 8: Módulo de Migração de IR e Exportação em PDF

**Files:**
- Create: `frontend/src/api/migration.js`, `frontend/src/hooks/useMigrationPlan.js`, `frontend/src/components/migration/HoldingsTable.jsx`, `frontend/src/components/migration/TargetAllocationForm.jsx`, `frontend/src/components/migration/MonthlyPurchasesTable.jsx`, `frontend/src/components/migration/ExportPdfButton.jsx`, `frontend/src/pages/MigrationPage.jsx`
- Test: `frontend/src/test/migration-ir.test.jsx`

**Interfaces:**
- Consumes: `GET /api/v1/migration/wallet`, `POST /api/v1/migration/plan`, `POST /api/v1/migration/plan.pdf`.
- Produces:
  - `migrationApi`: `getWallet()`, `calculatePlan(payload)`, `downloadPdf(payload)`.
  - `useMigrationPlan`: hook de estado gerenciando a lista de alvos, configurações de papéis e submissão do plano.
  - `<HoldingsTable />`: Tabela interativa da custódia com badges fiscais (`Isento R$ 20k`, `Tributado`, `Não estimado`), seletor de ação (`sell`, `count_as:<alvo>`, `ignore`), input para saldo manual em renda fixa e persistência de rascunho em `sessionStorage`.
  - `<TargetAllocationForm />`: Formulário dinâmico de alvos com soma percentual de pesos (validação visual = 100%) e busca de tickers via `assets/search`.
  - `<MonthlyPurchasesTable />`: Cronograma mês a mês detalhado exibindo vendas de ações no limite de R$ 20.000/mês, outras vendas e compras calculadas.
  - `<ExportPdfButton />`: Dispara o download de `POST /migration/plan.pdf` como blob `application/pdf`.
  - `<MigrationPage />`: Tela consolidada do planejador.

- [ ] **Step 1: Escrever teste automatizado para o módulo de Migração de IR**
  Arquivo: `frontend/src/test/migration-ir.test.jsx`.
  Testar:
  - `HoldingsTable` renderiza badges fiscais corretos para `ACAO` (`exempt_20k`) e `FII` (`taxed`).
  - `HoldingsTable` permite selecionar ação e insere saldo manual caso `saldo_source: "cost"`.
  - Se `getWallet` retornar 404 (`portfolio_empty`), renderiza o componente `EmptyState` com mensagem orientativa.
  - `TargetAllocationForm` desabilita botão de simular se a soma dos pesos for diferente de 100%.
  - `MonthlyPurchasesTable` exibe vendas agrupadas por mês e alertas fiscais (`exceeds_cap`, `exceeds_exemption`).
  - `ExportPdfButton` recebe blob e cria link de download simulado no DOM.

- [ ] **Step 2: Rodar teste para verificar que falha**
  Executar: `cd frontend && npm test src/test/migration-ir.test.jsx`
  Resultado esperado: FAIL ("cannot find module components/migration/HoldingsTable")

- [ ] **Step 3: Implementar API de migração, hook `useMigrationPlan`, componentes e tela `MigrationPage`**
  - `api/migration.js`: chamadas REST para `/wallet`, `/plan` e `/plan.pdf` (configurado com `responseType: 'blob'`).
  - `HoldingsTable.jsx`: tabela com colunas Ativo, Classe, Qtd, Preço Médio, Cotação, Saldo, Ação. Rascunhos de edições são lidos e sincronizados com `sessionStorage`.
  - `TargetAllocationForm.jsx`: lista dinâmica com botões de adicionar/remover alvos, input com autocomplete para ticker, input de peso (com badge verde quando soma for 100% e vermelho caso contrário).
  - `MonthlyPurchasesTable.jsx`: cartões de cada mês com detalhamento das vendas de ações, vendas de outras classes, total reinvestido em cada ativo-alvo e estimativa de DARF.
  - `ExportPdfButton.jsx`: manipula `window.URL.createObjectURL(blob)`, cria elemento `<a>`, aciona `.click()` e revoga o objeto URL.
  - `MigrationPage.jsx`: integra os blocos em um fluxo com 3 etapas claras: 1) Revisão da Custódia, 2) Definição de Metas, 3) Cronograma e Exportação.

- [ ] **Step 4: Rodar teste para verificar que passa**
  Executar: `cd frontend && npm test src/test/migration-ir.test.jsx`
  Resultado esperado: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api/migration.js frontend/src/hooks/useMigrationPlan.js frontend/src/components/migration/ frontend/src/pages/MigrationPage.jsx frontend/src/test/migration-ir.test.jsx
git commit -m "feat(frontend): implementar módulo completo de Migração de IR e exportação PDF"
```

---

## Task 9: Tela de Login, Roteamento Principal e Testes de Integração de Telas

**Files:**
- Create: `frontend/src/pages/LoginPage.jsx`, `frontend/src/App.jsx`, `frontend/src/main.jsx`, `frontend/src/test/app-routing.test.jsx`
- Modify: `frontend/package.json`
- Test: `frontend/src/test/app-routing.test.jsx`

**Interfaces:**
- Consumes: Todos os componentes, páginas e rotas criadas nas tasks anteriores.
- Produces:
  - `<LoginPage />`: Formulário de autenticação moderno com username, senha, indicador de loading e tratamento de erro 401 (`Credenciais inválidas`).
  - `<App />`: Roteador raiz com `BrowserRouter`, `QueryClientProvider`, `AuthProvider`, `PortfolioFilterProvider`, `<RequireAuth>` em rotas protegidas e rota `/login`.
  - `main.jsx`: Ponto de entrada React 18 montando no `#root`.

- [ ] **Step 1: Escrever teste de integração de roteamento e fluxos de tela**
  Arquivo: `frontend/src/test/app-routing.test.jsx`.
  Testar:
  - Navegar para `/` sem autenticação redireciona para `/login`.
  - Submeter formulário de login com sucesso redireciona para `/` e renderiza o `<Header />` e a `<PortfolioPage />`.
  - Alternar links no Header navega entre `/` (Carteira) e `/migracao-ir` (Planejador) sem recarregar a página.
  - Clicar no botão de logout limpa a sessão e retorna para `/login`.

- [ ] **Step 2: Rodar teste para verificar que falha**
  Executar: `cd frontend && npm test src/test/app-routing.test.jsx`
  Resultado esperado: FAIL ("cannot find module pages/LoginPage")

- [ ] **Step 3: Implementar `LoginPage.jsx`, `App.jsx` e `main.jsx`**
  - `LoginPage.jsx`: Card centralizado com visual escuro elegante, logo da plataforma, inputs estilizados com ícones da Lucide (`User`, `Lock`), botão de login com spinner durante requisição e banner de erro em caso de falha.
  - `App.jsx`: Configura `QueryClient` com defaults resilientes (`refetchOnWindowFocus: false`, `retry: 1`). Define rotas com `react-router-dom`:
    - Rota pública: `/login`.
    - Rotas protegidas (sob `<Layout />` e `<RequireAuth />`): `/` (`PortfolioPage`) e `/migracao-ir` (`MigrationPage`).
    - Rota curinga `*` redirecionando para `/`.
  - `main.jsx`: `ReactDOM.createRoot` montando `<App />` com import de `index.css`.

- [ ] **Step 4: Rodar todos os testes da aplicação e validar build de produção**
  Executar: `cd frontend && npm test`
  Resultado esperado: Todos os arquivos de teste passando (`X passed, 0 failed`).
  Executar: `npm run build`
  Resultado esperado: Build Vite finalizado gerando os assets estáticos em `frontend/dist/` sem warnings impeditivos.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/LoginPage.jsx frontend/src/App.jsx frontend/src/main.jsx frontend/src/test/app-routing.test.jsx
git commit -m "feat(frontend): implementar tela de login, roteamento e validação final da SPA"
```

---

## Self-Review e Verificação de Consistência

1. **Spec Coverage:**
   - Autenticação e sessão com CSRF -> Task 2 e Task 9.
   - Status DLP e modal de token -> Task 2 e Task 3.
   - Botão de atualizar carteira com invalidação de queries -> Task 3.
   - Barra de filtros com período, ativo, classe e benchmarks/sintéticos/tickers -> Task 4.
   - Cards de resumo patrimonial -> Task 5.
   - Gráficos de Performance (TWR base 100, Resumo Anual, Shadow, TIR) -> Task 6.
   - Gráficos de Risco (Drawdown, Rolling Sharpe, Dispersão Risco x Retorno) -> Task 7.
   - Migração de IR (Custódia com badges fiscais, alocação-alvo, cálculo e PDF) -> Task 8.
   - Resiliência contra 409/422 DLP e 404 portfolio empty -> Task 2, Task 3 e Task 8.
2. **Type/Contract Consistency:**
   - Nomes de campos alinhados ao backend: `start`, `end`, `ativo`, `classe`, `benchmarks`, `synthetics` (sem `corretora`).
   - Tickers com identificador `yf_ticker` e `${yf_ticker} BRL`.
   - Injeção de `X-CSRFToken` consistente em todas as mutações.
3. **Review Focus:** Todos os 5 cenários críticos mapeados possuem testes dedicados descritos nas respectivas tasks.
4. **Proporção:** O plano decompõe as 9 tasks em passos verificáveis com comandos claros de execução e commits atômicos.

