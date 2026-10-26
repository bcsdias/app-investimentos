# Referência Técnica — API DLP Invest (Dlombello Planilhas)

> **Data de catalogação:** 2026-10-26  
> **Título oficial na OpenAPI:** `Dlombello Planilhas` (`v1.0.0`, OpenAPI `3.1.0`)  
> **Descrição oficial:** *"Documentação da API do sistema dlombello para integração externa."*

Este documento consolida toda a especificação pública da API da **DLP Invest**, servindo como referência permanente para o `app-investimentos` (tanto para os endpoints já consumidos em `apps/market/services/sources/dlp.py` e na Fase 3b de Migração de IR, quanto para evoluções futuras como consulta de proventos, operações, DARF e DIRPF).

---

## 1. Endereços Base e Documentação Oficial

A API responde de forma idêntica nos dois domínios oficiais (ambos servem a mesma especificação OpenAPI 3.1.0 com 23 rotas):

| Recurso | Domínio atual (`dlpinvest.com.br`) | Domínio legado (`dlombelloplanilhas.com`) |
|---|---|---|
| **Base URL** | `https://users.dlpinvest.com.br` | `https://users.dlombelloplanilhas.com` |
| **Swagger UI** | `https://users.dlpinvest.com.br/docs` | `https://users.dlombelloplanilhas.com/docs` |
| **OpenAPI JSON** | `https://users.dlpinvest.com.br/openapi.json` | `https://users.dlombelloplanilhas.com/openapi.json` |

> **Nota no `app-investimentos`:** `backend/apps/market/services/sources/dlp.py` utiliza atualmente `_BASE = "https://users.dlombelloplanilhas.com"`. Ambos apontam para o mesmo serviço backend da DLP.

---

## 2. Autenticação e Segurança

- **Cabeçalho HTTP:** `Authorization: <token>`
  - O token pessoal de acesso externo começa com o prefixo `DLP-` (ex.: `DLP-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
  - Embora o schema `securitySchemes` do OpenAPI declare `scheme: bearer`, todas as rotas aceitam diretamente o token no header `Authorization: DLP-...` (sem precisar do prefixo `Bearer `).
- **Escopo de permissões do token:**
  - **Atenção:** o token externo tem permissão **total** na conta do usuário (leitura `GET`, inclusão `POST`, alteração `PUT` e exclusão `DELETE`, inclusive rotas destrutivas como `DELETE /operacoes/excluir/todas` e `DELETE /proventos/excluir/todos`).
  - **Regra no `app-investimentos`:** nosso backend faz **exclusivamente requisições `GET`** (`/historico` e `/resumo`), armazena o token cifrado com Fernet (`UserToken`) e nunca expõe o token em respostas, logs ou mensagens de exceção.
- **Códigos HTTP de erro observados:**
  - `401` / `403`: Token inválido, ausente ou revogado (mapeado para `DLPAuthError` → `422 dlp_token_rejected` na nossa API).
  - `422`: Erro de validação de parâmetros (`HTTPValidationError` padrão FastAPI/Pydantic).
  - `5xx`: Indisponibilidade ou erro interno na DLP (`DLPUnavailableError` → `502 dlp_unavailable`).

### Exemplo mínimo em Python

```python
import requests

token = "DLP-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
headers = {
    "Content-Type": "application/json",
    "Authorization": token,
}

# Posição consolidada atual da carteira
resp = requests.get("https://users.dlpinvest.com.br/carteira", headers=headers, timeout=60)
resp.raise_for_status()
print(resp.json())
```

---

## 3. Instrução Oficial para Agentes de IA (Somente Leitura)

Guia oficial da DLP Invest para uso do token com agentes de IA (ChatGPT, Claude, Gemini, Copilot, n8n):

```text
Você tem acesso à API da DLP Invest, onde estão meus investimentos.

- Base: https://users.dlpinvest.com.br
- Especificação (OpenAPI): https://users.dlpinvest.com.br/openapi.json
- Autenticação: header "Authorization" com o meu token (começa com "DLP-").

Regras:
1. Use só requisições GET (consultas). Nunca use POST, PUT ou DELETE sem eu pedir explicitamente; antes de executar, me mostre o que vai mudar e espere a minha confirmação.
2. Nunca mostre, repita nem grave o token nas respostas.
3. Ao responder, diga de qual endpoint veio cada número.

Endpoints úteis: GET /carteira (posição atual), GET /operacoes, GET /proventos, GET /historico, GET /darf, GET /dirpf?ano=AAAA.
```

---

## 4. Mapa Completo de Endpoints (23 rotas / 8 grupos)

> **Convenção de parâmetro de carteira:** quase todas as rotas aceitam selecionar uma carteira específica. Atenção à inconsistência de nome na própria API da DLP:
> - Em `GET /carteira` e `GET /historico`, o query param chama-se **`wid`**.
> - Nas demais rotas (`/resumo`, `/operacoes`, `/proventos`, `/darf`, `/dirpf`, `/outros`, `/multicarteiras`, `/usuario`), o query param chama-se **`wallet_id`**.
> - Quando omitido, a DLP utiliza a carteira principal do usuário.

### 4.1. Histórico (`tags: ["Histórico"]`) — **Usado pelo `app-investimentos`**

| Método | Rota | Query Params | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/historico` | `wid` *(str, opc)*<br>`classe` *(str, opc)*<br>`ativo` *(str, opc)*<br>`corretora` *(str, opc)*<br>`date_ini` *(date YYYY-MM-DD, opc)*<br>`date_fim` *(date YYYY-MM-DD, opc)* | `HistoricDetailModel` ou `list[ConsolidatedHistoricDetailModel]` | Evolução diária da carteira/ativos (`historico`), mais operações de lucro e proventos do mês atual. **Consumido por `dlp.buscar_historico`.** |
| **PUT** | `/historico` | `wallet_id` *(str, opc)* | `200` | Recalcula o histórico completo da carteira (`Calculate Historic Full`). |

### 4.2. Resumo (`tags: ["Resumo"]`) — **Usado pelo Planejador de IR (`dlp.buscar_resumo_carteira`)**

| Método | Rota | Query Params | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/resumo` | `wallet_id` *(str, opc)*<br>`summary` *(bool, default `true`)*<br>`wallet` *(bool, default `true`)*<br>`multiwallet` *(bool, default `true`)*<br>`notifications` *(bool, default `true`)*<br>`start_notifications_date` *(date, opc)* | `SummaryInitModel` | Retorna num único payload o resumo de lucros/proventos (`summary`), a posição atual (`wallet`), multicarteiras (`multiwallet`), eventos (`event`) e notificações. |

### 4.3. Carteira (`tags: ["Carteira"]`)

| Método | Rota | Parâmetros / Body | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/carteira` | Query: `wid` *(str, opc)* | `list[AtivoCarteiraModel]` ou `list[ConsolidatedWalletModel]` | Posição atual detalhada por ativo em custódia (quantidade, PM, cotação atual, valor investido, proventos acumulados, XIRR do ativo). |
| **POST** | `/carteira` | Body: `WalletModel` | `list[WalletModel]` | Cria uma nova carteira. |
| **PUT** | `/carteira` | Body: `WalletModel` | `list[WalletModel]` | Renomeia uma carteira existente. |
| **DELETE** | `/carteira` | Body: `WalletModel` | `list[WalletModel]` | Exclui uma carteira. |

### 4.4. Operações (`tags: ["Operações"]`)

| Método | Rota | Parâmetros / Body | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/operacoes` | Query: `wallet_id`, `classe`, `ativo`, `date_ini` *(date)*, `date_fim` *(date)*, `corretora`, `pagina` *(int, default 1)*, `qtd_pagina` *(int)*, `order_by_ativo` *(bool, default false)*, `order_date_asc` *(bool, default false)*, `daytrade` *(bool)*, `evento` *(str)*, `observacao` *(str)*, `classe_ini` *(str)* | `PaginatedOperationModel` | Lista paginada de notas/operações (compras, vendas, desdobramentos, etc.) e lucro filtrado por tipo de ativo. |
| **POST** | `/operacoes` | Query: `wallet_id`, `calc` *(bool, default true)*, `return_option` *(bool, default false)*, `validar_duplicadas` *(bool, default false)*<br>Body: `list[OperacaoInserirModel]` | `OperationsAnswer` | Insere novas operações em lote. |
| **PUT** | `/operacoes/atualizar` | Query: `wallet_id`<br>Body: `list[OperacaoAtualizarModel]` | `OperationsAnswer` | Atualiza operações existentes por `id`. |
| **PUT** | `/operacoes/tratar` | Query: `wallet_id` | `OperationsAnswer` | Processa/recalcula tratamento de operações da carteira. |
| **DELETE** | `/operacoes/excluir` | Query: `wallet_id`<br>Body: `list[DeleteRecordModel]` (`[{"id": "..."}]`) | `OperationsAnswer` | Exclui operações específicas por `id`. |
| **DELETE** | `/operacoes/excluir/todas` | Query: `wallet_id` | `GenericDlpModel` | **Destrutivo:** apaga todas as operações da carteira. |

### 4.5. Proventos (`tags: ["Proventos"]`)

| Método | Rota | Parâmetros / Body | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/proventos` | Query: `wallet_id`, `classe`, `ativo`, `date_ini` *(date)*, `date_fim` *(date)*, `corretora`, `pagina` *(int, default 1)*, `qtd_pagina` *(int)*, `order_by_ativo` *(bool, default false)*, `order_date_asc` *(bool, default false)*, `evento` *(str)*, `observacao` *(str)*, `classe_ini` *(str)* | `PaginatedEarningsModel` | Lista paginada de proventos (dividendos, JCP, rendimentos), média TTM filtrada e totais por classe. |
| **POST** | `/proventos` | Query: `wallet_id`, `calc` *(bool, default true)*, `validar_duplicadas` *(bool, default false)*<br>Body: `list[EarningsInsertModel]` | `EarningsAnswer` | Cadastra proventos em lote. |
| **PUT** | `/proventos/atualizar` | Query: `wallet_id`<br>Body: `list[EarningsUpdateModel]` | `EarningsAnswer` | Atualiza proventos existentes por `id`. |
| **DELETE** | `/proventos/excluir` | Query: `wallet_id`<br>Body: `list[DeleteRecordModel]` | `EarningsAnswer` | Exclui uma lista de proventos por `id`. |
| **DELETE** | `/proventos/excluir/{id_provento}` | Path: `id_provento` *(str, obrig)*<br>Query: `wallet_id` | `EarningsAnswer` | Exclui um único provento pelo `id_provento`. |
| **DELETE** | `/proventos/excluir/todos` | Query: `wallet_id` | `GenericDlpModel` | **Destrutivo:** apaga todos os proventos da carteira. |

### 4.6. IRPF — DARF e Declaração Anual (`tags: ["IRPF"]`)

| Método | Rota | Parâmetros / Body | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/darf` | Query: `wallet_id` *(str, opc)* | `DarfModel` | Apuração mensal de IR / DARFs (`results`), DARFs marcados como pagos (`payments`), configurações fiscais (`user_config`) e ajustes manuais (`valores_manuais`). |
| **GET** | `/dirpf` | Query: **`ano`** *(int, obrigatório)*, `wallet_id` *(str, opc)* | `object` (JSON livre) | Dados consolidados para preenchimento da Declaração Anual de IRPF do ano-calendário informado (Bens e Direitos, Rendimentos Isentos, Tributação Exclusiva, Renda Variável). |
| **PUT** | `/darf/payment` | Query: `wallet_id`<br>Body: `list[PaidDarfModel]` | `list[PaidDarfModel]` | Marca/atualiza pagamento de DARFs (`nacional`, `internacional`, `cripto`). |
| **PUT** | `/darf/values` | Query: `wallet_id`<br>Body: `list[DarfValueModel]` | `list[DarfValueModel]` | Atualiza saldos/ajustes manuais de apuração de IR (`DarfValueEnum`). |

### 4.7. Multicarteiras / Alocação (`tags: ["Multicarteiras"]`)

| Método | Rota | Parâmetros / Body | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/multicarteiras` | Query: `wallet_id` | `list[MultiwalletsModel]` | Lista subcarteiras/metas de alocação e pesos-alvo por ativo (`MultiwalletsAssetModel`: `peso_ativo`, `preco_entrada`, `preco_teto`, etc.). |
| **POST** | `/multicarteiras` | Query: `wallet_id`<br>Body: `MultiwalletsModel` | `list[MultiwalletsModel]` | Cria uma multicarteira de alocação. |
| **PUT** | `/multicarteiras` | Query: `wallet_id`<br>Body: `MultiwalletsModel` | `list[MultiwalletsModel]` | Atualiza uma multicarteira. |
| **DELETE** | `/multicarteiras` | Query: `wallet_id`<br>Body: `MultiwalletsModel` | `list[MultiwalletsModel]` | Remove uma multicarteira. |

### 4.8. Outros Ativos / Renda Fixa Customizada (`tags: ["Outros"]`)

| Método | Rota | Parâmetros / Body | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/outros` | Query: `wallet_id` | `list[UserOthersConfigModel]` | Lista configurações de ativos customizados/renda fixa (ex.: indexador `% CDI` + taxa adicional a.a., setor/subsetor, atualização automática). |
| **PUT** | `/outros` | Query: `wallet_id`<br>Body: `list[UserOthersConfigModel]` | `list[UserOthersConfigModel]` | Cria ou atualiza configurações de ativos customizados. |
| **DELETE** | `/outros` | Query: `wallet_id`<br>Body: `list[UserOthersConfigDeleteModel]` | `list[UserOthersConfigModel]` | Remove configurações de ativos customizados. |

### 4.9. Usuário (`tags: ["Usuário"]`)

| Método | Rota | Parâmetros / Body | Resposta `200` | Descrição |
|---|---|---|---|---|
| **GET** | `/usuario` | Query: `wallet_id` | `UsuarioModel` | Dados da conta, lista de carteiras (`WalletModel`), moeda base, preferências de cálculo e status do plano (`AlphaUserModel`). |
| **PUT** | `/usuario/config` | Query: `wallet_id`<br>Body: `UsuarioConfigAtualizarModel` | `UsuarioModel` | Atualiza preferências fiscais e de cálculo (moeda base, cálculo por corretora, regras de isenção 20k, compensação de prejuízos). |
| **GET** | `/usuario/alpha/{email_pagamento}` | Path: `email_pagamento` *(str)* | `AlphaUserModel` | Consulta assinatura/plano Alpha pelo e-mail de pagamento. |
| **PUT** / **DELETE** | `/usuario/alpha/planilha/id` | Body: `AlphaPlanilhaUpdate` | `bool` | Vincula/desvincula ID de planilha Google Sheets à conta Alpha. |

---

## 5. Principais Schemas de Dados (Modelos OpenAPI)

### 5.1. `HistoricDetailModel` e `HistoricModel` (`GET /historico`)
Retornado em `GET /historico`:
```jsonc
{
  "historico": [
    {
      "ativo": "PETR4",
      "date": "2025-01-02",
      "qtd": 100.0,
      "pm": 32.5,
      "price": 36.8,
      "corretora": "CLEAR",
      "market_cod": "BVMF:PETR4",
      "classe": "ACAO",
      "vlr_investido": 3250.0,
      "vlr_mercado": 3680.0,
      "lucro": 0.0,
      "daytrade": 0.0,
      "proventos": 0.0,
      "volume": 0.0,
      "pm_usd": null,
      "pm_fx": 32.5
    }
  ],
  "lucro_mes_atual": [ /* OperacaoModel[] */ ],
  "proventos_mes_atual": [ /* EarningsModel[] */ ]
}
```
> **Campos obrigatórios de `HistoricModel`:** `ativo`, `date`, `qtd`, `pm`, `price`, `corretora`, `market_cod`, `classe`, `vlr_investido`, `vlr_mercado`, `lucro`, `daytrade`, `proventos`, `volume`.  
> Isso garante que `ativo` e `classe` estão sempre presentes em cada linha de `/historico`, alimentando diretamente o endpoint `GET /api/v1/portfolio/filters` e os filtros de `fetch_user_portfolio`.

### 5.2. `AtivoCarteiraModel` (`GET /carteira` e `GET /resumo -> wallet`)
Representa cada posição aberta na carteira:
- **Obrigatórios:**
  - `ativo` (`str`) — código/sigla do ativo
  - `qtd` (`number`) — quantidade em custódia
  - `pm` (`number`) — preço médio na moeda original do ativo
  - `pm_fx` (`number`) — preço médio convertido para a moeda base da carteira (BRL)
  - `corretora` (`str`)
  - `classe` (`str`) — ex.: `ACAO`, `FII`, `ETF`, `STOCK`, `ETF_US`, `ETF_GB`, `REIT`, `CRIPTO`, `RF`, etc.
  - `market_cod` (`str`) — ex.: `BVMF:PETR4`, `LON:VWRA`, `NASDAQ:AAPL`
  - `moeda` (`str`) — ex.: `BRL`, `USD`, `GBP`
  - `setor` (`str`), `subsetor` (`str`)
  - `vlr_investido` (`number`) — custo total investido na posição
- **Opcionais / Calculados:**
  - `price` (`number | null`) — cotação atual na moeda do ativo
  - `price_fx` (`number | null`) — cotação atual convertida para a moeda base (BRL)
  - `change`, `changepct`, `change_fx`, `changepct_fx` (`number | null`) — variação do dia
  - `pm_usd` (`number | null`), `fator_pts` (`number | null`)
  - `earnings` (`number | null`) — proventos acumulados recebidos no ativo
  - `trades` (`number | null`) — resultado realizado em trades no ativo
  - `xirr` (`number | null`) — TIR individual do ativo
  - `date_ini` (`date | null`) — data da primeira compra
  - `nome_pregao` (`str | null`)

### 5.3. `SummaryInitModel` e `SummaryModel` (`GET /resumo`)
- `price_update_time` (`str`), `update_user_frontend` (`bool`)
- `wallet`: `list[AtivoCarteiraModel] | null`
- `multiwallet`: `list[MultiwalletsModel] | null`
- `notifications`: `list[NotificationsModel] | null`
- `event`: `list[EventNewsModel] | null`
- `summary` (`SummaryModel | null`):
  - `profits`: `list[ProfitSummary]` (`{asset_type: str, profit: number, daytrade: bool}`)
  - `earnings`: `list[EarningsSummary]` (`{asset_type: str, total: number}`)
  - `earnings_average_ttm` (`number`) — média mensal de proventos nos últimos 12 meses
  - `operations_values` (`SummaryOperationsEarnings | null`): `{assets: [str], classes: [str], events: [str], brokers: [str]}`
  - `earnings_values` (`SummaryOperationsEarnings | null`): `{assets: [str], classes: [str], events: [str], brokers: [str]}`

### 5.4. `OperacaoModel` (`GET /operacoes -> result[]`)
- **Obrigatórios:** `id`, `ativo`, `date`, `evento`, `qtd`, `preco`, `taxas`, `corretora`, `moeda`, `date_ini`.
- **Opcionais:** `irrf`, `observacao`, `daytrade`, `market_cod`, `classe`, `fluxo_caixa`, `volume`, `lucro`, `qtd_atual`, `pm_atual`, `qtd_ant`, `pm_ant`, `ptax`, `pm_fx`, `pm_usd`, `ptax_usd`, `nota_ptax` (`{cotacao_compra, cotacao_venda, data_hora_cotacao, tipo_boletim}`), `nome_pregao`, `outros_price`.

### 5.5. `EarningsModel` (`GET /proventos -> result[]`)
- **Obrigatórios:** `id`, `ativo`, `date`, `evento`, `valor`, `corretora`.
- **Opcionais:** `taxas`, `irrf`, `moeda`, `date_com`, `observacao`, `market_cod`, `classe`, `ptax`, `vlr_bruto`, `irrf_total`, `vlr_liquido`, `nota_ptax`.

### 5.6. `DarfModel` e Enums Fiscais (`GET /darf`)
- **`DarfEnum`:** `["nacional", "internacional", "cripto"]`
- **`DarfValueEnum`** (tipos de saldos/ajustes fiscais mensais):
  - Swing Trade (`st_*`): `st_isentosL13043`, `st_isentos20k`, `st_acoes`, `st_outros`, `st_ouro`, `st_opcoes`, `st_dolar`, `st_indice`, `st_futuro`, `st_termo`, `st_irrf`
  - Day Trade (`dt_*`): `dt_acoes`, `dt_outros`, `dt_ouro`, `dt_opcoes`, `dt_dolar`, `dt_indice`, `dt_futuro`, `dt_termo`, `dt_irrf`
  - FIIs (`fii_*`): `fii_resultado`, `fii_irrf`
- **`UsuarioConfigModel`** (`user_config`):
  - `moeda_base` (`str`), `calc_por_corretora` (`bool`), `dia_liq_ptax` (`int`)
  - Flags de regra de isenção de R$ 20k e subscrição: `dt_vd20k`, `outros_vd20k`, `opcoes_vd20k`, `vsub_FII`, `vsub_isento`, `bdr_like_acao`, `usar_saldo_vd20k`
  - Saldos iniciais/prejuízos a compensar: `darf_inicio` (`2000..2026`), `tributavel_normal`, `tributavel_daytrade`, `tributavel_fii`, `darf_total`.

---

## 6. Como o `app-investimentos` Utiliza a API DLP

| Módulo no `app-investimentos` | Endpoint DLP consumido | Como usa |
|---|---|---|
| [`dlp.buscar_historico`](../backend/apps/market/services/sources/dlp.py) | `GET /historico` | Passa `ativo`, `classe`, `corretora` opcionais; lê `payload["historico"]` para DataFrame, normaliza `data` → `date` e cacheia no Redis por 10 min (`TTL_DLP = 600s`) com single-flight e invalidação por geração. |
| [`dataset.fetch_user_portfolio`](../backend/engine/dataset.py) | `GET /historico` *(via `dlp.buscar_historico`)* | Agrupa `vlr_mercado`, `vlr_investido` e `proventos` por `date`, calcula `fluxo = diff(vlr_investido) - proventos`, ajusta o gap no `start_date` e calcula a série TWR base 100 da carteira. |
| [`portfolio.filters`](../backend/apps/api/routers/portfolio.py) | `GET /historico` *(via `dlp.buscar_historico`)* | Extrai valores únicos e ordenados das colunas `ativo` e `classe` da carteira inteira para popular os seletores de filtro da UI. |
| [`dlp.buscar_resumo_carteira`](../backend/apps/market/services/sources/dlp.py) | `GET /resumo?summary=true&wallet=true&multiwallet=false&notifications=false` | Retorna `summary` e `wallet` (`list[AtivoCarteiraModel]`) para o Planejador de Migração de Carteira com Isenção de IR (Fase 3b). |
