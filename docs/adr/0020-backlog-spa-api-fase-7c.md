# ADR 0020 — Resoluções do backlog SPA/API, robustez operacional e integridade da experiência

- **Status:** Aceita
- **Data:** 2026-10-30
- **Fase:** 7c
- **Spec:** [`../superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md`](../superpowers/specs/2026-10-28-fase-7-confiabilidade-design.md) (§5.3.1, §5.5, §5.6)
- **Plano:** [`../superpowers/plans/2026-10-29-fase-7c-spa-api.md`](../superpowers/plans/2026-10-29-fase-7c-spa-api.md)

## Contexto

A auditoria e os testes com dados reais do LAB realizados no encerramento das Fases 3/4 e 7b evidenciaram fragilidades na resiliência da API e da SPA, catalogadas no documento mestre (§2.3):
1. **Segurança e proteção contra abuso:** O endpoint de autenticação `/api/v1/auth/login` não possuía limitação de taxa (rate limiting), vulnerável a tentativas de força bruta em credenciais.
2. **Consistência de contrato e tratamento de erros:** Validações de payload e schema do Django Ninja retornavam formato de erro padrão do framework fora do envelope padronizado `ApiError` (`code`, `message`, `details`); operações concorrentes no cadastro de token DLP podiam disparar `IntegrityError`; e o endpoint `/summary` mantinha lógica de cálculo orquestrada diretamente no router.
3. **Resiliência de cache e sincronismo:** O mecanismo de `single_flight` utilizava chave simples de lock no Redis sem identificação do dono, sujeito à liberação indevida por processos expirados; a geração do cache DLP no Redis inicializava em 1, sujeita a colisão após reinício de container ou despejo de LRU; e falhas de autenticação com a API externa da DLP (`DLPAuthError`) não eram memorizadas em cache, gerando rajadas de chamadas concorrentes da SPA.
4. **Filtros e integridade dos dados da carteira:** Filtros por coluna ausente no histórico da DLP podiam ignorar o critério silenciosamente; a tabela anual continha coluna vazia quando o período abria no último dia do ano anterior; e lacunas nas bordas ou séries defasadas não eram alertadas.
5. **Usabilidade da SPA e relatórios:** A rejeição do token DLP na SPA disparava múltiplos erros em paralelo e reabria modais inoportunamente; o estado de período da análise não se refletia na URL para compartilhamento ou histórico; campos de data tinham acessibilidade e acionamento limitados; e o PDF gerado pelo planejador de IR sofria quebras de tabela e blocos de mês cortados ao meio entre páginas.

## Decisão

Implementar correções focadas em robustez e previsibilidade, preservando a retrocompatibilidade do contrato da API v1:

### 1. Limitação de Taxa no Login (`apps/accounts/rate_limit.py`)
- Adotado controle de tentativas de login baseado em Redis: máximo de **5 tentativas a cada 15 minutos** por tupla `(username, client_ip)`.
- Excesso de tentativas retorna status HTTP **`429 Too Many Requests`** com envelope de erro padrão `ApiError(code="too_many_attempts", message="Muitas tentativas de login...")`.

### 2. Envelope Padronizado em Validações de Schema (`config/api.py`, `apps/api/errors.py`)
- Erros de validação do Django Ninja (`ninja.errors.ValidationError`) são interceptados e convertidos para o envelope `ApiError` com HTTP **`422 Unprocessable Entity`** e `code: "invalid_request"`.
- Gravação do token DLP (`/api/v1/dlp/token`) trata concorrência atômica com captura de integridade (`IntegrityError`), evitando erro 500 em submissões duplicadas.
- O router `/portfolio/summary` passa a ser estritamente fino, delegando o pipeline de orquestração ao módulo `analysis.py`.

### 3. Sincronização do Período na URL e Limpeza de Sessão (`frontend/src/`)
- O período selecionado na SPA é refletido na URL via query param: `?period=1a|2a|5a|tudo|custom`.
- As transições usam `navigate(..., { replace: true })`, permitindo bookmarking e links diretos sem poluir o histórico de navegação do browser.
- No logout, o contexto de filtros é redefinido para o padrão, impedindo que preferências de um usuário vazem para sessões subsequentes no mesmo navegador.
- O login registra a rota de origem (`location.state.from`) e restaura a navegação após sucesso.

### 4. Gestão de Estado Único para Token Rejeitado ou Ausente
- A resposta `422 dlp_token_rejected` ou `409 dlp_token_missing` é tratada na SPA como um estado único de página (`TokenState`), suspendendo as demais consultas analíticas.
- Eliminado o disparo via interceptor global de rede para abertura automática de modal ao receber 422/409, garantindo que o modal de token só abra sob comando explícito do usuário.

### 5. `single_flight` com Lock Identificado por Dono (`apps/market/services/sources/dlp.py`)
- O lock distribuído para chamadas à DLP passa a gerar um identificador único de dono (`uuid.uuid4().hex`).
- A liberação do lock é executada atomicamente via script Lua que valida a posse do token antes de deletar a chave, impedindo que uma execução demorada apague o lock adquirido por outro worker.

### 6. Geração de Cache DLP Baseada em Timestamp (`apps/market/cache.py`)
- A chave de geração de cache da DLP (`dlp:hist:gen`) inicializa com `time.time_ns()` quando ausente no Redis.
- Evita reaproveitamento acidental de chaves residuais em caso de reinício do Redis ou evicção de memória.

### 7. Memorização de Rejeição de Autenticação em Cache
- Falhas `DLPAuthError` são armazenadas temporariamente em cache (`<key>:auth_err`) pelo tempo da geração ativa (com TTL curto de 30s), evitando que 8 requisições analíticas concorrentes no carregamento da tela façam 8 chamadas síncronas que a DLP rejeitaria.

### 8. Filtro Estrito sobre Colunas Ausentes
- Filtros requisitados sobre colunas não presentes no DataFrame da DLP resultam em retorno vazio acompanhado de warning em log, impedindo que linhas não filtradas sejam processadas indevidamente.

### 9. Paginação e Proteção de Layout no PDF do Planejador (`apps/migration_ir/pdf.py`)
- O gerador de PDF (`fpdf2`) aplica verificação de altura livre antes de iniciar cada mês (`keep_together`).
- Blocos mensais não são partidos entre páginas e cabeçalhos de tabela são repetidos em quebras normais, garantindo legibilidade do planejamento.

### 10. Acessibilidade e Melhorias de Interface
- O componente `Modal` incorpora `role="dialog"`, `aria-modal="true"`, foco automático e suporte à tecla `Escape`.
- O download do PDF adota o nome de arquivo informado no cabeçalho `Content-Disposition`, com revogação diferida do blob URL (`revokeObjectURL`).
- O campo de data nativo recebe `color-scheme: dark`, ícone de calendário explícito e acionamento por clique em qualquer área via `showPicker()`.
- Legendas dos gráficos ECharts utilizam rolagem horizontal (`type: 'scroll'`) evitando sobreposição com o título e a área de plotagem.

### 11. Itens Fechados sem Código e Adiamentos Justificados
- **Busca de ticker com debounce:** Dispensou alteração de código por já se encontrar implementada no componente `BenchmarkSelector` (`debouncedQuery`).
- **`catalog.defaults` não consumido na SPA:** Adiado. A remoção quebraria a compatibilidade da v1 e a adoção exige modelar persistência de preferências de benchmark por usuário.
- **Bundle JavaScript de ~1,4 MB:** Adiado para a Fase 7d (otimizações de desempenho / code-splitting dinâmico).
- **Validação de `MAX_MONTHS = 240`:** Corrigida condição de contorno que rejeitava planos válidos de exatamente 240 meses; execução da checagem após cotações externas mantida como aceita (impacto insignificante no tempo de resposta).

## Consequências

- **Compatibilidade:** O contrato da API v1 (`/api/v1/`) manteve estabilidade integral; nenhuma alteração quebradora de schema foi introduzida.
- **Segurança:** O backend passa a ser protegido contra ataques de força bruta no endpoint de login.
- **Estabilidade:** A eliminação de corridas concorrentes no lock do Redis e na gravação de tokens assegura operação contínua e sem erros 500 sob acessos paralelos.
- **Experiência do Usuário:** A navegação por períodos torna-se persistente e compartilhável via URL, a interface torna-se acessível e imune a modais intrusivos, e relatórios impressos ou exportados mantêm diagramação limpa.
