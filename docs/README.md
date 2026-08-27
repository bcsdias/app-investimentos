# Documentação — `app-investimentos`

Documentação da reformulação arquitetural (migração Streamlit → Django + React no Docker LAB).

## Índice

| Documento | Conteúdo |
|---|---|
| [`consolidacao_arquitetura_e_migracao.md`](consolidacao_arquitetura_e_migracao.md) | **Documento mestre.** Estado atual do repo, decisões de arquitetura, roteiro de fases (0 → 6). |
| [`superpowers/specs/`](superpowers/specs/) | Specs de design, um por incremento (`AAAA-MM-DD-<tema>-design.md`). |
| [`superpowers/plans/`](superpowers/plans/) | Planos de implementação passo a passo, um por incremento. |
| [`historico/`](historico/README.md) | Drafts e revisões anteriores, substituídos pelo documento mestre. |

## Fluxo de trabalho

Cada fase do roteiro do documento mestre passa por: **design** (`superpowers/specs/`) →
**plano** (`superpowers/plans/`) → **implementação** (TDD, commits por task na branch `lab-dev`).

Incremento atual: **Fases 0 + 1** — infraestrutura LAB + esqueleto do backend Django.
