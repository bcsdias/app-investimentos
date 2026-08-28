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
