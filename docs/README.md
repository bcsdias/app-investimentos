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
| [`historico/`](historico/README.md) | Drafts e revisões anteriores, substituídos pelo documento mestre. |

## Documentação viva

`consolidacao_arquitetura_e_migracao.md` §2 (estado atual) e `adr/` são atualizados **ao
fechar cada fase**, como último passo da Definition of Done do plano — junto com
`backend/CLAUDE.md` se surgirem convenções novas.

## Fluxo de trabalho

Cada fase do roteiro do documento mestre passa por: **design** (`superpowers/specs/`) →
**plano** (`superpowers/plans/`) → **implementação** (TDD, commits por task na branch `migracao-django`).

Fases 0 + 1 (infra LAB + esqueleto do backend Django) **concluídas em 2026-08-28**.
Próximo incremento: **Fase 2** — serviços de mercado (`apps/market/`, ingestão agendada).
