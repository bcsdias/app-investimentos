# Design — Fase 4 (Frontend React SPA)

> **Data:** 2026-10-26  
> **Branch:** `migracao-django`  
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md) (§1.1, §4, §5 Fase 4)  
> **Fases anteriores:** 0 + 1 (2026-08-28), 2 + 2.5 (2026-08-31), 3 + 3b + 3c (2026-10-26) — specs em `docs/superpowers/specs/`  
> **Legado:** `src/ui/` (Streamlit multipágina — `app.py`, `pages/1..4`, `components/`)  
> **Escopo:** Criação do frontend moderno `frontend/` (Vite + React 18 + Tailwind CSS + ECharts + TanStack Query) consumindo a API Django Ninja (`/api/v1/`).

---

## 1. Contexto e Porquê

Com as Fases 1 a 3c concluídas e validadas por 329 testes unitários/contrato e testes de integração com serviços reais, o backend Django Ninja expõe toda a inteligência analítica e fiscal da plataforma em `/api/v1/`:
- Autenticação por sessão (`/auth/`) com proteção CSRF.
- Gestão de credenciais DLP (`/dlp/`).
- 8 endpoints analíticos de carteira (`/portfolio/`) com formato serializado nativo para gráficos de séries temporais.
- Catálogo de benchmarks (`/benchmarks/`) e busca de tickers sob demanda (`/assets/search`).
- Planejador de Migração de IR e geração de PDF (`/migration/`).

O frontend atual ainda é o protótipo legado em Streamlit (`src/ui`), que roda na porta 8501, acoplado a dependências pesadas de servidor, com recarregamento completo da página a cada interação e sem a agilidade e reatividade exigidas por um produto financeiro moderno.

A **Fase 4** cria a aplicação cliente definitiva em **React 18 SPA**, desacoplada, servida via Vite em desenvolvimento e pronta para empacotamento estático no Nginx (Fase 5), proporcionando uma experiência de uso fluida, gráficos interativos de alta performance e suporte nativo ao fluxo de trabalho do investidor.

---

## 2. Decisões Travadas (Brainstorming 2026-10-26)

Decisões tomadas com o usuário:

| # | Tema | Decisão |
|---|---|---|
| 1 | **Organização de Telas** | **Duas visões principais + Modal de Configurações**: Dashboard da Carteira (com abas ágeis *Performance* e *Risco*), Módulo Dedicado de *Migração de IR*, e *Modal/Drawer de Token DLP* no topo. Nada de página isolada para salvar token. |
| 2 | **Estratégia de Disparo dos Gráficos** | **Disparo simultâneo de todas as métricas em segundo plano** no carregamento inicial / troca de filtros. Como o backend já possui bloqueio *single-flight* no Redis (ADR 0013 / ADR 0015), as 8 requisições paralelas não sobrecarregam a DLP e a troca entre abas fica 100% instantânea para o usuário. |
| 3 | **Arquitetura de Estado e Dados** | **Abordagem 1: TanStack Query + React Context**. O TanStack Query gerencia o ciclo de vida das 8 queries analíticas, cancelando automaticamente requisições obsoletas via `AbortController` quando o usuário altera filtros antes do retorno, e o React Context gerencia a sessão e os filtros globais. |

Decisões de engenharia / chamadas do designer:

| # | Tema | Decisão | Por quê |
|---|---|---|---|
| D1 | **Tema e Estilo** | **Tema escuro nativo (Dark Mode)** usando paleta do Tailwind CSS (Zinc/Slate) com acentos esmeralda/índigo. | Padrão dos dashboards financeiros e do protótipo legado (`src/ui/components/theme.py`). |
| D2 | **Biblioteca de Gráficos** | **Apache ECharts** (`echarts-for-react` + `echarts`). | Já definido nas fases anteriores; o backend (`apps.api.serializers`) devolve exatamente a estrutura esperada pelo ECharts (`name`, `data: [["YYYY-MM-DD", val], ...]`). Suporta zoom, pan e tooltip rico sem degradação com milhares de pontos. |
| D3 | **Tratamento de Erros da DLP** | **Intercepção centralizada de erros da DLP**. Se a API retornar HTTP 409 (`dlp_token_missing`) ou HTTP 422 (`dlp_token_rejected`), a UI abre automaticamente o modal de configuração de token com feedback explicativo, **sem** deslogar o usuário. | 401 desloga (sessão expirada); 409/422 são erros de credencial externa. |
| D4 | **Comunicação HTTP e CSRF** | Cliente Axios configurado com `withCredentials: true`. No bootstrap da sessão, chama `GET /api/v1/auth/csrf` para obter o token e o injeta no cabeçalho `X-CSRFToken` em todas as mutações (`POST`, `DELETE`). | Exigência da arquitetura de segurança do Django Ninja (`ApiSessionAuth`). |
| D5 | **Roteamento SPA** | `react-router-dom` v6 com rotas `/login`, `/` (Carteira) e `/migracao-ir` (Planejador), envelopadas por `<RequireAuth>`. | Simplicidade e URLs limpas compartilháveis. |
| D6 | **Estado dos Filtros na URL** | Filtros de período básico (`period=1A`) refletidos opcionalmente na query string para manter histórico de navegação limpo. | Permite recarregar a página sem perder o recorte temporal. |

---

## 3. Abordagens Consideradas

| | Abordagem | Trade-off | Status |
|---|---|---|---|
| **1** | **TanStack Query + React Context** | Estado assíncrono isolado com cache em memória, cancelamento nativo com `AbortController`, retries controlados e deduplicação de requisições. Custo: adiciona `@tanstack/react-query`. | **Escolhida** |
| 2 | **React Context Puro + Axios Direto** | Nenhuma dependência externa além de Axios, mas exige escrever controle manual de concorrência, cancelamento de requisições obsoletas e múltiplos estados de loading. | Descartada |
| 3 | **Redux Toolkit / RTK Query** | Excessivamente opinado e com boilerplate desnecessário para uma SPA de 2 telas analíticas. | Descartada |

---

## 4. Estrutura de Diretórios (`frontend/`)

```
frontend/
├── package.json
├── vite.config.js              # Proxy /api/v1 e /admin -> backend local
├── tailwind.config.js          # Tema escuro e paleta financeira
├── postcss.config.js
├── index.html
├── src/
│   ├── main.jsx                # Ponto de entrada, QueryClientProvider, RouterProvider
│   ├── App.jsx                 # Configuração de rotas e providers de contexto
│   ├── index.css               # Diretivas do Tailwind e estilos globais de scrollbar/inputs
│   │
│   ├── api/
│   │   ├── client.js           # Instância Axios (withCredentials, CSRF interceptor, baseURL)
│   │   ├── auth.js             # Funções /auth/ (csrf, login, logout, me)
│   │   ├── dlp.js              # Funções /dlp/ (token, token-status)
│   │   ├── portfolio.js        # Funções /portfolio/ (filters, refresh, summary, gráficos)
│   │   ├── benchmarks.js       # Funções /benchmarks/ (catalog)
│   │   ├── assets.js           # Funções /assets/ (search)
│   │   └── migration.js        # Funções /migration/ (wallet, plan, plan.pdf)
│   │
│   ├── context/
│   │   ├── AuthContext.jsx     # Contexto de autenticação, usuário logado e verificação de sessão
│   │   └── PortfolioFilterContext.jsx # Período, benchmarks ativos, sintéticos e filtros de ativos
│   │
│   ├── hooks/
│   │   ├── usePortfolioQueries.js # Hook mestre que dispara e gerencia as queries do TanStack Query
│   │   └── useMigrationPlan.js    # Hook de simulação e geração de PDF da migração
│   │
│   ├── components/
│   │   ├── common/             # Primitivas visuais reutilizáveis
│   │   │   ├── Button.jsx
│   │   │   ├── Card.jsx
│   │   │   ├── Modal.jsx
│   │   │   ├── Skeleton.jsx
│   │   │   ├── Badge.jsx
│   │   │   └── Tabs.jsx
│   │   │
│   │   ├── layout/
│   │   │   ├── Header.jsx      # Topbar com status DLP, refresh, perfil e navegação
│   │   │   └── Layout.jsx      # Envoltório de página
│   │   │
│   │   ├── dlp/
│   │   │   └── DlpTokenModal.jsx # Modal para salvar/remover token com status ao vivo
│   │   │
│   │   ├── portfolio/
│   │   │   ├── FiltersBar.jsx           # Presets de período, seletor de datas, filtros de ativos
│   │   │   ├── BenchmarkSelector.jsx    # Catálogo + busca de tickers + criador de sintéticos
│   │   │   ├── SummaryCards.jsx         # Cards de KPI (Patrimônio, Aportes, TWR, TIR, etc.)
│   │   │   ├── charts/
│   │   │   │   ├── BaseChart.jsx        # Wrapper de ECharts com tema, resize automático e loading
│   │   │   │   ├── TwrChart.jsx         # Gráfico de evolução da cota (base 100)
│   │   │   │   ├── DrawdownChart.jsx    # Gráfico de perdas subaquáticas (%)
│   │   │   │   ├── RollingSharpeChart.jsx # Sharpe e volatilidade anualizada móvel
│   │   │   │   ├── ShadowChart.jsx      # Simulação de aportes nos benchmarks
│   │   │   │   └── RiskReturnChart.jsx  # Dispersão Risco (Vol) x Retorno (CAGR)
│   │   │   └── tables/
│   │   │       └── YearlyTable.jsx      # Matriz de rentabilidade anual por ativo/benchmark
│   │   │
│   │   └── migration/
│   │       ├── HoldingsTable.jsx        # Tabela da carteira atual com ações (sell/count_as/ignore)
│   │       ├── TargetAllocationForm.jsx # Configuração de ativos-alvo e percentuais
│   │       ├── MonthlyPurchasesTable.jsx# Cronograma detalhado mês a mês
│   │       └── ExportPdfButton.jsx      # Disparo do download de PDF com estado de carregamento
│   │
│   └── pages/
│       ├── LoginPage.jsx       # Tela de login simples
│       ├── PortfolioPage.jsx   # Dashboard da carteira com abas Performance e Risco
│       └── MigrationPage.jsx   # Tela do planejador de migração de IR
└── tests/                      # Testes de componentes e integração via Vitest + Testing Library
```

---

## 5. Fluxos e Telas Detalhadas

### 5.1. Autenticação e Sessão
1. **Entrada na SPA:** `<App>` inicializa o `AuthContext`.
2. Executa `GET /api/v1/auth/me` e `GET /api/v1/auth/csrf`.
3. Se autenticado, carrega a rota solicitada. Se `401 Unauthorized`, redireciona para `/login`.
4. Na tela `/login`, formulário simples com campos `username` e `password`.
5. Ao submeter com sucesso (`POST /api/v1/auth/login`), armazena o usuário no estado e redireciona para `/`.
6. No logout (`POST /api/v1/auth/logout`), limpa o cache do TanStack Query e redireciona para `/login`.

### 5.2. Header e Gestão do Token DLP
- Exibe o logo e o título da plataforma.
- **Badge de Status DLP:**
  - Consulta `GET /api/v1/dlp/token-status`.
  - Se `configured: true`, exibe indicador verde com data de `updated_at`.
  - Se `configured: false`, exibe alerta amarelo ("Token DLP não configurado").
  - Clique no badge abre o **`DlpTokenModal`**.
- **Botão "Atualizar Dados":**
  - Dispara `POST /api/v1/portfolio/refresh`, que invalida o cache de 10 minutos da DLP no backend.
  - Ao retornar sucesso, aciona `queryClient.invalidateQueries({ queryKey: ['portfolio'] })`, disparando a recarga instantânea de todos os gráficos.
- **Navegação:** Links simples para alternar entre `Carteira` e `Planejador de IR`.

### 5.3. Dashboard da Carteira (`/`)

#### 5.3.1. Barra de Filtros (`FiltersBar`)
- **Seletor de Período:** Botões rápidos `1A`, `2A`, `5A`, `Tudo` e opção `Personalizado` (com inputs `start` e `end` no formato `YYYY-MM-DD`).
- **Filtros Opcionais de Custódia:** Dropdowns alimentados por `GET /api/v1/portfolio/filters` para filtrar por `ativo` ou `classe` (alinhado a `PortfolioFiltersOut`).
- **Seletor de Benchmarks e Comparadores (`BenchmarkSelector`):**
  - Dropdown com checkboxes para os índices padrão do catálogo (`CDI`, `IBOV`, `S&P 500`, etc. obtidos em `GET /api/v1/benchmarks/catalog`).
  - Botão "+ Comparador":
    - Aba 1: *Pesquisar Ativo* — busca com debounce em `GET /api/v1/assets/search?q=...` sobre os ~7.500 tickers do universo. O identificador armazenado e enviado à API é sempre o `yf_ticker` retornado (ex.: `PETR4.SA`, `AAPL`). Se a opção "Versão em BRL" estiver marcada (para ativos com `brl_variant: true`), envia `${asset.yf_ticker} BRL`.
    - Aba 2: *Criar Sintético* — formulário estruturado `{base: "CDI" | "IPCA", operador: "+" | "%", valor: number}`.

#### 5.3.2. Cards de Resumo (`SummaryCards`)
- Consome `POST /api/v1/portfolio/summary`.
- Exibe 5 cards com métricas formatadas:
  1. **Patrimônio Atual:** Valor em R$ e variação patrimonial total.
  2. **Total Aportado:** Soma líquida de aportes históricos no período.
  3. **Ganho de Capital:** Lucro/Prejuízo nominal em R$.
  4. **Rentabilidade TWR:** Percentual ponderado pelo tempo.
  5. **TIR (XIRR):** Taxa interna anualizada real da carteira.
- Exibe skeletons enquanto a requisição estiver carregando.

#### 5.3.3. Abas Analíticas
- **Aba Performance:**
  - *Evolução TWR (`TwrChart`):* Séries temporais em base 100 da carteira vs benchmarks selecionados. ECharts com zoom interativo por arrasto no rodapé.
  - *Tabela Ano a Ano (`YearlyTable`):* Grade comparativa contendo o retorno anual percentual da carteira e de cada benchmark, com cores dinâmicas (verde para positivo, vermelho para negativo).
  - *Shadow Portfolio (`ShadowChart`):* Gráfico de patrimônio simulado em R$, demonstrando o resultado caso os aportes tivessem ido para os benchmarks.
  - *Evolução da TIR (`IrrEvolutionChart`):* Curva histórica da TIR ao longo do tempo.
- **Aba Risco:**
  - *Drawdown Subaquático (`DrawdownChart`):* Gráfico de área negativa demonstrando as quedas percentuais a partir do topo histórico.
  - *Sharpe e Volatilidade Móvel (`RollingSharpeChart`):* Janela de 252 dias úteis demonstrando a variação do risco e da eficiência.
  - *Dispersão Risco × Retorno (`RiskReturnChart`):* Gráfico de dispersão (scatter plot) plotando Volatilidade Anualizada no eixo X e CAGR no eixo Y.

### 5.4. Módulo de Migração de IR (`/migracao-ir`)
1. **Carregamento da Carteira:**
   - Chama `GET /api/v1/migration/wallet`.
   - Se a API retornar `404 portfolio_empty` (usuário sem histórico na DLP), exibe um `EmptyState` explicativo orientando o cadastro de movimentações.
   - Renderiza a `HoldingsTable` com colunas: Ativo, Classe, Quantidade, Preço Médio, Cotação Atual, Saldo e Tratamento Fiscal (com badges visuais para `Isento (R$ 20k)`, `Tributado` e `Não estimado`).
   - Cada linha permite selecionar a ação: `Vender`, `Contar como <alvo>` ou `Ignorar`.
   - Permite informar saldo manual para ativos de renda fixa que não possuem cotação em bolsa (`saldo_source: "cost"`).
   - O rascunho de edições do usuário (ações por papel e saldos manuais) é salvo temporariamente no `sessionStorage` para não ser perdido caso o usuário navegue entre abas.
2. **Formulário de Alocação-Alvo (`TargetAllocationForm`):**
   - Usuário adiciona os tickers alvo (com busca em `assets/search`) e define os pesos percentuais desejados.
   - Validação visual em tempo real: a soma dos pesos deve totalizar exatamente 100%.
   - Campo para informar o teto mensal (padrão R$ 20.000) e vendas já realizadas no mês corrente fora do plano.
3. **Simulação do Plano (`MonthlyPurchasesTable`):**
   - Botão "Calcular Plano" dispara `POST /api/v1/migration/plan`.
   - Exibe alertas de planejamento (ex.: vendas tributadas concentradas no mês 1, estouro de isenção caso um ativo unitário ultrapasse o limite).
   - Tabela organizada mês a mês exibindo o valor total de vendas de ações (sempre ≤ R$ 20.000 para ações), vendas de outras classes, total a comprar de cada ativo-alvo e IR estimado a recolher.
4. **Exportação de Relatório:**
   - Botão "Baixar Relatório em PDF" (`ExportPdfButton`) envia o mesmo payload para `POST /api/v1/migration/plan.pdf`.
   - Recebe o binário `application/pdf` e dispara o download no navegador com o nome do arquivo sugerido.

---

## 6. Integração HTTP e Tratamento de Resiliência

### 6.1. Cliente Axios e Autenticação
```javascript
// src/api/client.js
import axios from 'axios';

export const api = axios.create({
  baseURL: '/api/v1',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  // Injeta token CSRF do cookie ou do endpoint /auth/csrf
  const csrfToken = document.cookie
    .split('; ')
    .find((row) => row.startsWith('csrftoken='))
    ?.split('=')[1];

  if (csrfToken && ['post', 'put', 'patch', 'delete'].includes(config.method.toLowerCase())) {
    config.headers['X-CSRFToken'] = csrfToken;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    const code = error.response?.data?.code;

    if (status === 401 && window.location.pathname !== '/login') {
      window.location.href = '/login';
    } else if (status === 409 && code === 'dlp_token_missing') {
      // Dispara evento ou atualiza estado global para abrir DlpTokenModal
      window.dispatchEvent(new CustomEvent('open-dlp-modal', { detail: { reason: 'missing' } }));
    } else if (status === 422 && code === 'dlp_token_rejected') {
      window.dispatchEvent(new CustomEvent('open-dlp-modal', { detail: { reason: 'rejected' } }));
    }
    return Promise.reject(error);
  }
);
```

### 6.2. Estratégia de Queries Concorrentes (TanStack Query)
Cada gráfico e o resumo possuem sua query isolada com `queryKey: ['portfolio', endpointName, filterState]`.

```javascript
// Exemplo em src/hooks/usePortfolioQueries.js
const { start, end, benchmarks, synthetics, ativo, classe } = usePortfolioFilters();

const payload = {
  start: start || undefined,
  end: end || undefined,
  benchmarks,
  synthetics,
  ativo: ativo || undefined,
  classe: classe || undefined,
};

// As 8 queries são declaradas via useQueries
const results = useQueries({
  queries: [
    { queryKey: ['portfolio', 'summary', payload], queryFn: ({ signal }) => fetchSummary(payload, signal) },
    { queryKey: ['portfolio', 'twr-evolution', payload], queryFn: ({ signal }) => fetchTwr(payload, signal) },
    { queryKey: ['portfolio', 'drawdown', payload], queryFn: ({ signal }) => fetchDrawdown(payload, signal) },
    { queryKey: ['portfolio', 'rolling-metrics', payload], queryFn: ({ signal }) => fetchRolling(payload, signal) },
    { queryKey: ['portfolio', 'risk-return', payload], queryFn: ({ signal }) => fetchRiskReturn(payload, signal) },
    { queryKey: ['portfolio', 'shadow', payload], queryFn: ({ signal }) => fetchShadow(payload, signal) },
    { queryKey: ['portfolio', 'yearly-summary', payload], queryFn: ({ signal }) => fetchYearly(payload, signal) },
    { queryKey: ['portfolio', 'irr-evolution', payload], queryFn: ({ signal }) => fetchIrr(payload, signal) },
  ],
});
```
- **Cancelamento Automático:** O parâmetro `signal` repassado ao Axios garante que, se o usuário alterar o período de 1A para 5A enquanto a primeira busca está em andamento, as requisições em voo do período anterior são canceladas no cliente.
- **Cache Local:** Se o usuário alternar para "1A", depois para "Tudo" e voltar para "1A", o resultado é exibido instantaneamente do cache de memória (`staleTime: 5 * 60 * 1000`).

---

## 7. O que Fica a Critério do Construtor

- Estilos específicos de animação e transições do Tailwind (ex.: opacidade de fade-in ao carregar gráficos).
- Modularização fina de componentes atômicos (ex.: divisão exata de subcomponentes dentro de `charts/` e `tables/`).
- Casos de teste unitário específicos com Vitest para validação de formatação de moeda e datas.

---

## 8. Definition of Done (Critérios de Aceite da Fase 4)

1. **Bootstrap Completo:** `frontend/` funcional com `npm install`, `npm run dev` e build de produção `npm run build` gerando bundle estático em `frontend/dist/` sem erros.
2. **Sessão e Segurança:** Login via `/login` funcionando contra o Django, CSRF injetado corretamente nas mutações e redirecionamento de 401 funcionando.
3. **Dashboard da Carteira Integrado:**
   - Todos os 8 endpoints analíticos consumidos e renderizados com Apache ECharts.
   - Skeletons exibidos durante a primeira carga assíncrona.
   - Barra de filtros funcional com atualização reativa dos gráficos.
   - Inclusão dinâmica de benchmarks do catálogo, busca de tickers (`assets/search`) e criação de sintéticos.
4. **Módulo de Migração de IR:**
   - Leitura da carteira atual via `/migration/wallet`.
   - Formulação de alocação-alvo e simulação com retorno da tabela mês a mês.
   - Download de PDF (`/migration/plan.pdf`) disparado e salvo com sucesso no navegador.
5. **Resiliência e Erros:**
   - Erros 422 de token ausente/rejeitado capturados com abertura amigável do modal de token DLP.
   - Botão de "Atualizar Carteira" invalidando o cache do backend e recarregando a interface.
6. **Testes Unitários:** Suíte de testes com Vitest cobrindo os hooks de estado e a renderização dos componentes principais.

