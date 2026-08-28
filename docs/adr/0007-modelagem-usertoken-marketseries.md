# 0007 — Modelagem de `UserToken` e `MarketSeries`

**Status:** Aceita · **Data:** 2026-08-28 · **Fase:** 1

## Contexto

Dois modelos entram no `apps.core` na Fase 1:

- **`UserToken`** — substitui a tabela `user_tokens` do Supabase (PK = e-mail, um token DLP
  por usuário, cifrado com Fernet).
- **`MarketSeries`** — armazena os pontos diários das séries de mercado (benchmarks,
  índices, taxas) que hoje só existem em cache volátil. Base da ingestão agendada (Fase 2).

## Decisão

**`UserToken`**

- `user = OneToOneField(AUTH_USER_MODEL, on_delete=CASCADE, related_name="dlp_token")` —
  um token por usuário, garantido pelo banco; acesso ergonômico via `user.dlp_token`.
- `encrypted_token = TextField()` — sempre o ciphertext Fernet. Métodos `set_token(raw)` /
  `get_token()` encapsulam `apps.core.security`. O valor plano nunca toca o campo.
- `created_at` / `updated_at` automáticos.

**`MarketSeries`**

- `series_key: CharField(120)` (ex.: `"bcb:12"`, `"yf:^BVSP"`), `source: CharField(16)`
  (`BCB` | `YF` | `TD` | `B3` | `PTAX`), `reference_date: DateField`,
  `value: DecimalField(20, 8)`, `updated_at`.
- `UniqueConstraint(["series_key", "reference_date"], name="uniq_series_point")` — no máximo
  um ponto por série por dia; a ingestão faz upsert contra essa constraint.
- Index em `(series_key, reference_date)` e `ordering` por essas colunas.
- `DecimalField` (não `FloatField`): é dado financeiro, precisão exata importa.

## Consequências

- **+** Integridade no banco, não na aplicação: impossível ter 2 tokens por usuário ou 2
  pontos para o mesmo `(série, data)`.
- **+** `set_token`/`get_token` = ponto único onde a cripto acontece.
- **−** O formato exato de `series_key` / `source` ainda não está fechado — é refinamento
  da Fase 2. Os `CharField` largos absorvem a indefinição por ora.
- **−** `Admin` de `UserToken` precisa de cuidado especial (sem "add", `encrypted_token`
  read-only) para nunca gravar valor plano — já implementado.
