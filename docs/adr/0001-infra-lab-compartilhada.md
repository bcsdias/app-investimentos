# 0001 — Infra LAB compartilhada

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 0

## Contexto

O "Docker LAB" hospeda mais de um projeto. `app-investimentos` precisa de PostgreSQL e
Redis. Duas abordagens: (1) um `lab-postgres` e um `lab-redis` únicos servindo o LAB
inteiro, com um database dedicado por projeto; (2) um par de containers por projeto.

A opção 2 multiplica RAM, portas e manutenção. A 1 concentra recursos mas exige isolar
os dados de cada projeto dentro da instância compartilhada.

## Decisão

Usar **um `lab-postgres` e um `lab-redis` compartilhados**, ligados por uma rede Docker
externa **`lab-net`**. O `app-investimentos` recebe:

- um **database dedicado `appinvest`** (role `appinvest` com `CREATEDB`) no `lab-postgres`;
- um **logical DB do Redis** (`redis://localhost:6379/1`) no `lab-redis`.

Os compose files (`/data/projetos/lab-postgres/`, `/data/projetos/lab-redis/`) ficam
**fora do repo do app** — esses diretórios não são repositórios git. Só o
`backend/scripts/bootstrap_db.sh` (cria role+database, idempotente) é versionado.

Nesta fase o Django roda **no host**; conteinerizar o backend é a Fase 5. O `settings.py`
já é escrito para os dois modos, trocando só variáveis de ambiente.

## Consequências

- **+** Uma instância de cada serviço para todo o LAB; menos RAM/portas.
- **+** `bootstrap_db.sh` idempotente = setup reproduzível sem clonar compose de infra.
- **−** Isolamento lógico, não físico: disciplina obrigatória de nunca deixar tabela do
  app cair no `banco_lab` (ver [0006](0006-prefixo-lab-containers-appinvest-codigo.md)).
- **−** Infra versionada só parcialmente (o script). O estado real dos containers vive fora do git.
- Healthchecks adicionados a ambos os containers; ambos entram na `lab-net`.
