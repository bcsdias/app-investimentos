# Documentação — `app-investimentos`

Documentação da reformulação arquitetural (migração Streamlit → Django + React no Docker LAB).

## Índice

| Documento | Conteúdo |
|---|---|
| [`consolidacao_arquitetura_e_migracao.md`](consolidacao_arquitetura_e_migracao.md) | **Documento mestre.** Estado atual do repo (§2), decisões de arquitetura, roteiro de fases (0 → 6). |
| [`adr/`](adr/README.md) | **Architecture Decision Records** — uma decisão por arquivo, imutável. O *porquê* consultável a longo prazo. |
| [`../backend/README.md`](../backend/README.md) | Setup, rodar e testar o backend Django. |
| [`../backend/CLAUDE.md`](../backend/CLAUDE.md) | Convenções para agentes trabalhando em `backend/`. |
| [`superpowers/specs/`](superpowers/specs/) | Specs de design, um por incremento (`AAAA-MM-DD-<tema>-design.md`). |
| [`superpowers/plans/`](superpowers/plans/) | Planos de implementação passo a passo, um por incremento. |
| [`ferramentas-de-desenvolvimento.md`](ferramentas-de-desenvolvimento.md) | Quais plugins/skills do Claude Code usar em cada fase (geração de código, revisão, segurança). |
| [`seguranca-dados-pessoais.md`](seguranca-dados-pessoais.md) | **Repo público:** o que nunca versionar (carteira, segredos, logs), checklist antes de commitar, como limpar o histórico se algo vazar, registro da limpeza de 2026-10-20. |
| [`historico/`](historico/README.md) | Drafts e revisões anteriores, substituídos pelo documento mestre. |

## Documentação viva

Alguns documentos refletem o **estado atual** do repositório e são atualizados **ao fechar
cada fase**, como último passo explícito da Definition of Done do plano. Ver o
[checklist de pós-fase](#checklist-de-atualização-ao-fechar-uma-fase) abaixo — todo plano
deve incluí-lo na última task para não deixar doc para trás (aconteceu nas Fases 2/2.5:
`docs/README.md` e o §4 do documento mestre ficaram desatualizados e foram corrigidos depois).

### Checklist de atualização ao fechar uma fase

Rodar **todos** os itens abaixo antes de considerar a fase concluída. Marcar o que não se
aplica com "n/a" no commit de docs.

| # | Arquivo / seção | O que atualizar |
|---|---|---|
| 1 | `consolidacao_arquitetura_e_migracao.md` §2.0 ("Já implementado") | Linha nova (ou revisão) da camada entregue: local, situação, contagem de testes **total atual**, ADRs novos. Atualizar o título da seção com as fases cobertas. |
| 2 | `consolidacao_arquitetura_e_migracao.md` §2.2 ("O que **não** existe ainda") | Remover o que passou a existir. |
| 3 | `consolidacao_arquitetura_e_migracao.md` §2.3 ("Dívida técnica") | Marcar itens resolvidos (mantendo o registro histórico); **acrescentar** dívida nova assumida de propósito (pendências de revisão adiadas, decisões "por enquanto"). |
| 4 | `consolidacao_arquitetura_e_migracao.md` §4 ("Estrutura de Diretórios Alvo") | Reconciliar o desenho com o que **de fato** foi construído quando o layout divergir do plano original. |
| 5 | `consolidacao_arquitetura_e_migracao.md` §9 (checklist de ordem) | Marcar `[x]` a(s) fase(s) fechada(s). |
| 6 | `docs/README.md` (este arquivo) — "Fluxo de trabalho" | Atualizar a linha de fases concluídas e o "próximo incremento". |
| 7 | `docs/adr/` + `docs/adr/README.md` | ADR novo por decisão de arquitetura travada na fase; linha correspondente no índice. |
| 8 | `backend/README.md` | Bloco **"Estado"** (topo), seção **"Estrutura"** (novos módulos/pacotes) e a contagem de testes no bloco **"Rodar"** (`# N testes`). |
| 9 | `backend/CLAUDE.md` | Só se surgiram convenções novas de código/estrutura. |
| 10 | `ferramentas-de-desenvolvimento.md` | Só se a fase mudou quais skills/plugins usar dali em diante. |
| 11 | Specs/planos consumidos | Se a implementação divergiu do plano de forma relevante, anotar no commit e (se afeta fases seguintes) no spec correspondente. |

## Fluxo de trabalho

Cada fase do roteiro do documento mestre passa por: **design** (`superpowers/specs/`) →
**plano** (`superpowers/plans/`) → **implementação** (TDD, commits por task na branch `migracao-django`).

Fases **0 + 1** (infra LAB + esqueleto do backend Django) concluídas em 2026-08-28.
Fases **2 + 2.5** (serviços de mercado `apps/market/` + engine puro `backend/engine/`)
concluídas em 2026-08-31, com correções de revisão aplicadas na sequência.
Próximo incremento: **Fase 3** — endpoints da API (Django Ninja).
