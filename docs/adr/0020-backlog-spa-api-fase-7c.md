# ADR 0020 — Resoluções do backlog SPA/API, robustez operacional e integridade da experiência

- **Status:** Aceita
- **Data:** 2026-10-30 (revisada no mesmo dia após a validação independente — ver "Revisão pós-validação")
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

### 1. Limitação de Taxa no Login (`apps/accounts/ratelimit.py`)
- Adotado controle de tentativas de login baseado em Redis: máximo de **5 tentativas a cada 15 minutos** por tupla `(username, client_ip)`.
- Excesso de tentativas retorna status HTTP **`429 Too Many Requests`** com envelope de erro padrão `ApiError(code="too_many_attempts", message="Muitas tentativas de login...")`.
- O IP vem de `REMOTE_ADDR`; `X-Real-IP` só é aceito quando `REMOTE_ADDR` é um proxy confiável (`settings.TRUSTED_PROXIES` ou, por padrão, loopback e redes privadas RFC 1918 conferidas por faixa CIDR). O nginx do `web` resolve o cliente real a partir do `X-Forwarded-For` de um proxy à frente (`real_ip`). Ver "Revisão pós-validação" para a limitação no LAB.

### 2. Envelope Padronizado em Validações de Schema (`config/api.py`, `apps/api/errors.py`)
- Erros de validação do Django Ninja (`ninja.errors.ValidationError`) são interceptados e convertidos para o envelope `ApiError` com HTTP **`422 Unprocessable Entity`** e `code: "invalid_request"`.
- Gravação do token DLP (`/api/v1/dlp/token`) trata concorrência atômica com captura de integridade (`IntegrityError`), evitando erro 500 em submissões duplicadas.
- O router `/portfolio/summary` passa a ser estritamente fino, delegando o pipeline de orquestração ao módulo `analysis.py`.

### 3. Sincronização do Período na URL e Limpeza de Sessão (`frontend/src/`)
- O período selecionado na SPA é refletido na URL via query param: `?period=1a|2a|5a|tudo|custom`.
- As transições usam `navigate(..., { replace: true })`, permitindo bookmarking e links diretos sem poluir o histórico de navegação do browser.
- No logout, o contexto de filtros é redefinido para o padrão, impedindo que preferências de um usuário vazem para sessões subsequentes no mesmo navegador.
- O login registra a rota de origem (`location.state.from`, com `pathname` **e** `search`) e restaura a navegação após sucesso, inclusive um link `/?period=...` aberto sem sessão.
- O "Sair" explícito navega para `/login` **sem** `state.from`; é isso que zera os filtros no próximo login. Sessão expirada e link sem sessão passam pelo `RequireAuth`, que guarda a origem.

### 4. Gestão de Estado Único para Token Rejeitado ou Ausente
- A resposta `422 dlp_token_rejected` ou `409 dlp_token_missing` é tratada na SPA como um estado único de página (`TokenState`), suspendendo as demais consultas analíticas.
- Eliminado o disparo via interceptor global de rede para abertura automática de modal ao receber 422/409, garantindo que o modal de token só abra sob comando explícito do usuário.

### 5. `single_flight` com Lock Identificado por Dono (`apps/market/cache.py`)
- O lock distribuído para serialização de cálculos e chamadas externas (`single_flight`) passa a gerar um identificador único de dono (`uuid.uuid4().hex`).
- A liberação do lock utiliza a abstração de cache do Django conferindo a titularidade (`cache.get(lock_key) == my_token`) antes do `cache.delete(lock_key)`, preservando a compatibilidade total com o cache de teste (`locmem`) e com o Redis, impedindo que uma execução demorada apague o lock adquirido por outro worker. O `get` + `delete` não é atômico (não há script Lua): resta uma janela de microssegundos entre os dois, aceita porque o lock é só otimização (spec: cache é otimização).

### 6. Geração de Cache DLP Baseada em Timestamp (`apps/market/services/sources/dlp.py`)
- A chave de geração de cache da DLP (`dlp:gen:<token_hash>`) inicializa com `time.time_ns()` quando ausente no Redis.
- Evita reaproveitamento acidental de chaves residuais em caso de reinício do Redis ou evicção de memória.

### 7. Memorização de Rejeição de Autenticação em Cache
- Falhas `DLPAuthError` são armazenadas temporariamente em cache (`<key>:auth`) pelo tempo da geração ativa (com TTL curto de 30s), evitando que 8 requisições analíticas concorrentes no carregamento da tela façam 8 chamadas síncronas que a DLP rejeitaria.

### 8. Filtro Estrito sobre Colunas Ausentes
- Filtros requisitados sobre colunas não presentes no DataFrame da DLP resultam em `None` (nenhuma linha casa, o mesmo retorno de um filtro sem resultado) acompanhado de warning em log, impedindo que linhas não filtradas sejam processadas indevidamente.

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

## Revisão pós-validação (2026-10-30)

Uma validação independente da Fase 7c (suítes, paridade contra as linhas de base 7b e 7c, cenários do teste do usuário rodados no LAB com a carteira real e chamadas pelo nginx) confirmou a Parte A e a Parte B. Ela achou três gaps, todos corrigidos com TDD (teste vermelho antes):

1. **Link com `?period=` perdido no login** (`LoginPage.jsx`, `Header.jsx`). Para o teste "filtros voltam ao padrão após logout" passar, o `LoginPage` descartava o `search` quando a origem era `/`. Um link `/?period=custom&start=...&end=...` aberto sem sessão caía em "Tudo". Agora o login sempre restaura `pathname + search`, e o reset de filtros vem do "Sair", que navega para `/login` sem `state.from`.
2. **Falso aviso de série defasada no IPCA** (`apps/market/gaps.py`, ver [ADR 0019](0019-integridade-dos-dados-de-mercado.md) §4). A cauda mensal usava a tolerância de 45 dias. O IPCA é datado no dia 1 do mês de referência e publicado por volta do dia 10 do mês seguinte, então o último ponto chega a ter ~70 dias normalmente. O `market_gaps` já acusava `bcb:433 defasada desde 2026-09-01 (59 dias)`, e "IPCA + 6%" ganharia aviso todo começo de mês. A cauda tem agora tolerância própria (`TAIL_TOLERANCE_DAYS`: 10 dias diária, 75 dias mensal). A tolerância entre pontos (`TOLERANCE_DAYS`, 45 dias) não muda.
3. **Proxy confiável por prefixo de string e IP do gateway no LAB** (`apps/accounts/ratelimit.py`, `frontend/nginx.conf`). O padrão aceitava qualquer `REMOTE_ADDR` começando com `172.`, incluindo faixas públicas como `172.217.0.0/16`, e até strings que não são IP. Agora a conferência usa `ipaddress` contra `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` e `::1`. No LAB, a chave gravada era `ratelimit:login:<usuário>:172.20.0.1`: a porta é publicada em `127.0.0.1:8080` e o docker-proxy entrega toda conexão com o IP do gateway da bridge. O nginx passou a usar `real_ip` (`set_real_ip_from` 127.0.0.1 e 172.16.0.0/12, `real_ip_header X-Forwarded-For`, `real_ip_recursive on`), de modo que um túnel ou reverse proxy no host que mande `X-Forwarded-For` entregue o IP real.
   - **Limitação aceita:** acessando direto por `127.0.0.1:8080` (túnel SSH ou encaminhamento de porta do editor), não existe IP de cliente além do gateway. Nesse caso o limite vale efetivamente **por usuário**: 5 senhas erradas para um username bloqueiam esse username por 15 min, de qualquer origem. É aceitável num LAB de uso pessoal. Expor o app publicamente exige um proxy que mande `X-Forwarded-For`.

Menor, corrigido junto: `DateInput` espalhava `{...rest}` depois de `onClick`/`aria-label`, e um `onClick` passado por fora anulava o `showPicker`.

Pendente, registrado no [ADR 0019](0019-integridade-dos-dados-de-mercado.md) §4: as séries IMA do BCB (`bcb:12461`, `12466`, `12467`, `12468`, `12469`) têm, depois de 05/2023, pontos que **não vieram do BCB**. São projeções por ETF gravadas pelo legado no CSV do bootstrap.

## Consequências

- **Compatibilidade:** O contrato da API v1 (`/api/v1/`) manteve estabilidade integral; nenhuma alteração quebradora de schema foi introduzida.
- **Segurança:** O backend passa a ser protegido contra ataques de força bruta no endpoint de login.
- **Estabilidade:** A eliminação de corridas concorrentes no lock do Redis e na gravação de tokens assegura operação contínua e sem erros 500 sob acessos paralelos.
- **Experiência do Usuário:** A navegação por períodos torna-se persistente e compartilhável via URL, a interface torna-se acessível e imune a modais intrusivos, e relatórios impressos ou exportados mantêm diagramação limpa.
