# Architecture Decision Records

Registro imutável das decisões de arquitetura da migração. Um arquivo por decisão,
`NNNN-titulo-curto.md`, **append-only**: uma decisão errada não se apaga — cria-se
uma nova ADR que a supera (`Supersedes: NNNN`) e marca-se a antiga como `Substituída`.

Formato de cada ADR:

- **Status** — Aceita | Substituída por NNNN | Descontinuada
- **Contexto** — o problema, as forças em jogo
- **Decisão** — o que foi decidido, no presente ("Usar…")
- **Consequências** — o que isso implica, bom e ruim

Complementam (não substituem) as tabelas "decisões travadas" dos specs em
`../superpowers/specs/`. Aqui fica o *porquê* consultável a longo prazo.

## Índice

| # | Decisão | Fase |
|---|---|---|
| [0001](0001-infra-lab-compartilhada.md) | Infra LAB compartilhada: um `lab-postgres`/`lab-redis` para o LAB, database `appinvest` dedicado, rede `lab-net` | 0 |
| [0002](0002-modelo-user-customizado-dia-1.md) | `accounts.User(AbstractUser)` customizado desde o primeiro `migrate` | 1 |
| [0003](0003-reuso-da-chave-fernet-do-legado.md) | Reusar a chave Fernet do `secrets.toml` do app legado | 1 |
| [0004](0004-psycopg-binary-3.md) | Driver Postgres: `psycopg[binary]` 3.x | 1 |
| [0005](0005-sem-django-cors-headers.md) | Sem `django-cors-headers` (SPA + API na mesma origem) | 1 |
| [0006](0006-prefixo-lab-containers-appinvest-codigo.md) | Containers mantêm prefixo `lab-*`; database e código usam `appinvest` | 0 |
| [0007](0007-modelagem-usertoken-marketseries.md) | `UserToken` OneToOne cifrado; `MarketSeries` com `UniqueConstraint(series_key, reference_date)` | 1 |
| [0008](0008-logging-stdout-apenas.md) | Logging só `StreamHandler` (stdout), sem arquivo | 1 |
| [0009](0009-django-ninja-stub-desde-fase-1.md) | `django-ninja` já pinado na Fase 1, como stub | 1 |
| [0010](0010-python-dotenv-puro.md) | Gerência de env com `python-dotenv` puro | 1 |
| [0011](0011-marketseries-valor-cru-kind.md) | `MarketSeries` armazena valor cru por fonte + campo `kind` (`rate`/`level`) | 2 |
| [0012](0012-engine-copia-literal-fase-2-5.md) | Cópia literal do engine de cálculo puro na Fase 2.5 e revisão pós-Fase 6 | 2.5 |
| [0013](0013-contrato-da-api-v1.md) | Contrato da API v1: endpoints analíticos `POST`, envelope `data/meta`, séries ECharts e `422` para token DLP rejeitado | 3 |
| [0014](0014-tickers-sob-demanda.md) | Tickers sob demanda (`ativos.csv` + fetch-on-read), invalidação DLP por geração e single-flight via `cache.add` | 3 |
| [0015](0015-planejador-de-ir.md) | Planejador de IR: alvos livres, associação manual, regra de isenção por classe, IR estimado, nunca inventar preço | 3b |
| [0016](0016-empacotamento-no-lab.md) | Empacotamento no LAB: dev no host × containers `lab-appinvest-*` em `:8080`, Chromium ARM64, `supercronic`, timeouts 180 s, Supabase descartado | 5 |
| [0017](0017-paridade-legado.md) | Paridade com o legado em duas camadas (mercado congelado 1e-6 × ponta a ponta 1e-4) e linha de base da Fase 7 sem `src` | 6 |
| [0018](0018-metodologia-de-retorno-dlp.md) | Metodologia de cálculo de retorno alinhada à planilha DLP (fluxo com lucro, HPR aporte-início/resgate-fim, XIRR com fallback, rf = CDI) | 7a |
| [0019](0019-integridade-dos-dados-de-mercado.md) | Integridade dos dados de mercado, invalidação de cache por geração global e tratamento de lacunas com ffill e meta.gaps | 7b |


