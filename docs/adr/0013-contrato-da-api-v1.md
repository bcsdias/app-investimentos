# ADR 0013 — Contrato da API v1 (Django Ninja)

- **Status:** Aceita
- **Data:** 2026-10-26
- **Fase:** 3 (API da Carteira)

## Contexto

A Fase 3 expõe o núcleo de cálculo (`engine/`) e os serviços de mercado (`apps.market`) como uma API REST JSON consumida pela futura SPA React (Fase 4). Era necessário travar o formato das requisições de análise, o envelope de resposta e séries para o Apache ECharts, a semântica dos códigos HTTP de erro (especialmente falhas da API DLP externa versus sessão expirada) e a proteção CSRF em rotas de sessão.

## Decisão

1. **Verbo e corpo das análises (D1):** os 8 endpoints analíticos de `/api/v1/portfolio/*` (`summary`, `twr-evolution`, `drawdown`, `rolling-metrics`, `risk-return`, `shadow`, `yearly-summary`, `irr-evolution`) usam **`POST` com corpo JSON** (`AnalysisRequest` / `RollingRequest`), evitando codificar listas de benchmarks e sintéticos estruturados `{base, operador, valor}` em query strings.
2. **Um endpoint por gráfico + `/portfolio/risk-return` (D6):** cada gráfico faz sua chamada independente (beneficiando-se do cache DLP e Redis) e o gráfico de dispersão Risco × Retorno (`compute_risk_return`) ganha rota explícita.
3. **Invalidação explícita do cache DLP (D2):** `POST /api/v1/portfolio/refresh` invalida o cache DLP do usuário autenticado antes de recarregar os gráficos, evitando um parâmetro `refresh=true` replicado em 8 chamadas simultâneas.
4. **Envelope e formato de séries (D4, D5):**
   - Toda resposta analítica usa `{"data": ..., "meta": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "missing": [str]}}`, onde `start`/`end` refletem o período efetivo do `df_combined` e `missing` lista benchmarks/sintéticos sem dados no período.
     - *Nota (2026-10-27):* no request, `start` nulo significa **"último ano"** (`hoje − 1 ano`), não "desde o início". Para o histórico completo a SPA envia `start = "1900-01-01"` (preset "Tudo") e exibe o período real a partir de `meta.start`/`meta.end`.
   - Séries temporais seguem `{"name": str, "data": [["YYYY-MM-DD", float|null], ...]}` (`NaN`/`±inf` convertidos para `null`), compatível diretamente com o eixo `time` do ECharts.
   - **A API não converte unidades:** devolve os números exatos do `engine` para preservar a verificação de paridade da Fase 6.
5. **Contrato de erros e distinção 401 vs 422 (D3):**
   - Erros de negócio seguem `{"code": str, "message": str, "details": object|null}` com mensagem em português.
   - `401` é exclusivo para ausência de sessão ou falha em `/auth/login` (`invalid_credentials`).
   - Quando a API DLP rejeita o token do usuário (HTTP 401/403 na origem), o backend captura `DLPAuthError` e responde **`422 dlp_token_rejected`** (nunca `401`), impedindo que a SPA confunda token DLP inválido com sessão Django expirada.
   - Indisponibilidade da DLP → `502 dlp_unavailable`; usuário sem token cadastrado → `409 dlp_token_missing`; carteira vazia no período/filtro → `404 portfolio_empty`; falha de CSRF → `403 csrf_failed`.

### Tabela de códigos de erro

| Situação | HTTP | `code` |
|---|---|---|
| Sem sessão autenticada | 401 | *(padrão do Ninja)* |
| Credenciais inválidas no login | 401 | `invalid_credentials` |
| Falha de validação CSRF em escrita | 403 | `csrf_failed` |
| Carteira sem dados no período/filtro | 404 | `portfolio_empty` |
| Usuário sem token DLP configurado | 409 | `dlp_token_missing` |
| Token DLP rejeitado pela API externa | 422 | `dlp_token_rejected` |
| Parâmetros inválidos de análise / busca / token | 422 | `invalid_period`, `unknown_benchmark`, `invalid_synthetic`, `too_many_benchmarks`, `invalid_window`, `invalid_token`, `invalid_query` |
| DLP indisponível (timeout, conexão, 5xx, JSON inválido) | 502 | `dlp_unavailable` |

### Verificação na API DLP real

- Em teste de integração ao vivo (`test_real_dlp_invalid_token_raises_auth_error`), um token inválido enviado a `users.dlombelloplanilhas.com/historico` retorna HTTP 401/403, confirmando o mapeamento direto para `DLPAuthError` → `422 dlp_token_rejected`.
- Para `/portfolio/filters`, a extração procura as colunas candidatas `("ativo", "ticker", "codigo")` e `("classe", "tipo")` (case-insensitive) e degrada graciosamente para `[]` com warning caso ausentes.

## Consequências

### Positivas
- A SPA tem contratos previsíveis, JSON estrito (sem `NaN`/`Infinity`) e distinção semântica clara entre sessão expirada (`401`), falta de CSRF (`403`), falta de token (`409`) e token DLP inválido (`422`).
- Números idênticos ao `engine`, mantendo intacta a comparação de paridade da Fase 6.

### Negativas / Mitigações
- Como as rotas analíticas usam `POST`, o cliente HTTP da SPA deve enviar o header `X-CSRFToken` (obtido do cookie `csrftoken` semeado em `GET /api/v1/auth/csrf`).
