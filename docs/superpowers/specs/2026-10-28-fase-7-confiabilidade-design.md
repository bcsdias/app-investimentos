# Design — Fase 7 (Confiabilidade antes de desempenho)

> **Data:** 2026-10-28
> **Branch:** `migracao-django`
> **Documento mestre:** [`../../consolidacao_arquitetura_e_migracao.md`](../../consolidacao_arquitetura_e_migracao.md) (§2.3, §5 Fase 7, §9 item 12)
> **Fases anteriores:** 0 a 6 — specs em `docs/superpowers/specs/`
> **ADRs relevantes:** [0012](../../adr/0012-engine-copia-literal-fase-2-5.md) (engine cópia literal, revisão pós-Fase 6), [0014](../../adr/0014-tickers-sob-demanda.md) (`.L` via PTAX USD), [0016](../../adr/0016-empacotamento-no-lab.md) (empacotamento no LAB), [0017](../../adr/0017-paridade-legado.md) (paridade e linha de base da Fase 7)

---

## 1. Intenção e porquê

O app já está no ar no LAB, mas ninguém o usou a fundo. A Fase 6 provou que ele calcula
**igual ao legado** — não que calcula **certo**: o engine foi copiado literalmente (ADR 0012),
com bugs conhecidos, e o usuário não tem uma referência externa (planilha, extrato) para
conferir os números.

A Fase 7 deixa o app **confiável para uso diário**: números corretos, sem dados de mercado
inválidos e sem bugs que atrapalhem o uso. O usuário vai começar a usar o app de verdade em
paralelo e trazer o que parecer estranho.

**Prioridade do usuário (decisão dele):** correção primeiro; desempenho por último, só
depois que o resto estiver verde. Lentidão hoje não é problema.

**Sucesso:**
- um gabarito de carteiras sintéticas com respostas calculadas à mão, todo verde;
- os bugs de recorte de período corrigidos e a revisão do engine (ADR 0012) feita contra esse gabarito;
- dados de mercado sem cache defasado e sem buracos silenciosos;
- o backlog de bugs da SPA/API (§2.3 do doc mestre) zerado ou explicitamente adiado;
- o container do LAB reconstruído com tudo isso, para o usuário testar.

## 2. Decisões tomadas com o usuário

1. **Ordem:** correção → integridade de dados → bugs de uso → **otimização por último**
   (inverte a ordem do §5 Fase 7 do doc mestre, que listava a otimização como item 1).
2. **Oráculo:** sem referência externa, a correção é provada por um **gabarito montado à mão**
   (carteiras sintéticas pequenas com resultado conhecido). O legado/linha de base prova só
   "não mudou o que não devia".
3. **Uso real em paralelo:** o usuário testa o app no LAB; cada suspeita vira **primeiro um
   caso no gabarito** (ou teste de regressão), só depois correção.
4. **Escopo inclui o backlog de bugs da SPA/API** (§2.3 do doc mestre — revisão de 2026-10-27
   e "menores da revisão final"), não só o engine.
5. **Convenções financeiras seguem o padrão de mercado**, decididas pelo builder e
   documentadas em ADR (o usuário delegou: "veja como é o padrão de mercado e documente").
   Ver §5.3.1 para as escolhas já feitas.
6. **A DLP não divulga como calcula** (`proventos` = dividendos/rendimentos recebidos de ações
   e FIIs, nas palavras do usuário). A semântica dos campos foi inferida dos dados (§3.1) e
   será confirmada contra a planilha oficial da DLP (§5.2.1).

## 3. Fatos levantados no código

### 3.1. Semântica do `/historico` da DLP (inferida do histórico congelado, 2026-10-28)

Análise de `reports/parity/20261027-182533/baseline/hist/w1-completo.pkl` (92 ativos, 122
datas, 2.950 linhas), sem expor valores:

| Campo | O que é (evidência) |
|---|---|
| frequência | **mensal** (intervalos de 28–31 dias), apesar da doc chamar de "evolução diária" |
| `vlr_investido` | `qtd × pm` em 100 % das linhas → **custo médio da posição**; cai na venda pelo custo (78 ativos com quedas) |
| `vlr_mercado` | `qtd × price` (99,96 %) |
| `proventos` | **do período, não acumulado**: só 2 de 50 ativos com provento são monotônicos; aparece também com `qtd = 0` |
| `lucro` | **lucro realizado na venda do período**: ≠ 0 em 89 % das linhas com venda, ~0 % nas outras; ≈ `(preço − pm) × qtd vendida` |
| `daytrade` | sempre 0 nesta carteira |

A planilha oficial da DLP (`dlp_sheet/`, exportada pelo usuário) tem na aba `historico` uma
coluna **`fluxo_caixa`** que a API não entrega, e na aba `dashboard` **TWR mensal, TWR
acumulado e TIR calculados pela própria DLP**. A cópia exportada está sem dados (nunca foi
atualizada); os `.gs` só chamam `executarScript(...)` remoto — a fórmula não está neles.

### 3.2. Bug central: o fluxo externo ignora o lucro realizado

`fluxo = Δvlr_investido − proventos` (em `dataset.py` e `twr.py`). Numa venda, o dinheiro que
sai da carteira é `qtd_vendida × preço = Δcusto + lucro`, mas o engine só vê `Δcusto`: o
**lucro realizado some** do TWR (vira "perda" na venda com lucro e "ganho" na venda com
prejuízo). Se o dinheiro é reinvestido, a parcela do lucro aparece como **aporte externo**. O
mesmo `fluxo` alimenta o XIRR (`compute_irr_evolution`), o `/shadow` e os KPIs.

Medido na carteira real (janela completa): TWR acumulado **40,9 %** com a fórmula atual ×
**169,7 %** com `fluxo = Δvlr_investido − lucro − proventos` (44 de 122 meses têm venda).

**Confirmado contra a planilha oficial da DLP (2026-10-28)**, atualizada pelo usuário e
exportada em `dlp_sheet/google_sheets_DLP_html/` (aba `historico` com 2.952 linhas, aba
`dashboard` com TWR mensal de out/2024 a set/2026):

| Fórmula do fluxo (fluxo no início do mês, como o engine) | erro médio no TWR mensal × DLP | erro máximo |
|---|---|---|
| atual: `Δvlr_investido − proventos` | 0,53 pp | 4,11 pp |
| **corrigida: `Δvlr_investido − lucro − proventos`** | **0,004 pp** | **0,03 pp** |

**Metodologia completa da DLP reproduzida** pela aba oculta `dashboard.` (export
`dashboard..html`: série mensal de jul/2016 a set/2026 com `sum vlrInvest`, `sum vlrMercado`,
`sum proventos`, `sum lucro_trades`, `fluxo_caixa`, `TWR % a.m`, `TWR_acc`):

- No nível da carteira, `fluxo_caixa = Δvlr_investido − lucro − proventos` em **123/123** meses.
- TWR mensal: **aporte líquido no início do mês, resgate líquido no fim**:
  `r = (M_t − min(F,0)) / (M_{t−1} + max(F,0)) − 1`.
  Bate em **122/122** meses (erro máx. 0,005 pp, arredondamento) e o acumulado
  (**195,95 %**) é idêntico ao `TWR_acc` da DLP. "Fluxo sempre no início" (estrutura atual do
  engine) erra nos 7 meses de resgate grande (ex.: mar/2020, −11,9 % × −3,8 %), e soma 171 %.
- A coluna `fluxo_caixa` **por ativo** da aba `historico` usa outra convenção de sinal e não é
  necessária.

Pendência do oráculo: a aba `dashboard` mostra a TIR total da carteira (a.m. e a.a.) —
comparar com o XIRR corrigido.

### 3.3. Demais fatos

- `engine/dataset.build_dataset` concatena a carteira (mensal) com benchmarks (diários) e faz
  `ffill`: a curva da carteira é uma escada diária. Vol/Sharpe móveis com `sqrt(252)` sobre
  essa escada dão aproximadamente a anualização mensal (`sqrt(12)·σ_mensal`) por coincidência
  estatística — rever explicitamente na §5.3.

- `engine/dataset.fetch_user_portfolio` (`backend/engine/dataset.py:29`):
  - filtra `grp` só por `start_date`; `end_date` só é aplicado ao `twr` dentro de
    `calculate_twr`. `portfolio_df` vai até hoje mesmo com `end_date` no passado → `/summary`
    (`pat_fim`, `aportes`, `ganho`, `xirr`), `/shadow` e `/irr-evolution` ignoram `end_date`.
  - após `filt["vlr_investido"] += gap`, `filt["fluxo"]` continua com o valor anterior ao
    ajuste; `compute_irr_evolution` (`engine/compute.py:63`) usa `-fluxo` como fluxos.
- `engine/twr.calculate_twr` refaz `to_datetime`, `groupby("date").sum()` e `fluxo` (alvo da
  otimização no fim). `np.where(denominador != 0, …)` sem epsilon; primeira linha sempre
  tratada como aporte.
- `engine/irr.calculate_xirr`: Newton-Raphson, chute 0.1, base 365, clamp `-0.99`, não ordena
  datas, aceita convergência por passo pequeno sem checar `npv`, e no fim devolve `r` se
  `|npv| < 0.1` (tolerância em R$, independente da escala da carteira).
- `engine/metrics.py`: constantes `252` e `rf_constant=0.10` embutidas.
- `apps/market/series.get_series` cacheia em `market:series:<nome>:<start>:<end>` com TTL 24 h;
  `ingest_market_data` grava no Postgres e aquece o cache, mas **não invalida** as chaves já
  existentes.
- `ingest_market_data` incremental começa em `last_date + 1` — nunca vê buracos no meio.
- Buracos de fonte conhecidos: `bcb:12469` (IMA-S, 2024, 275 dias), `yf:LFTS11.SA` (2025, 151 dias).
- Linha de base: `reports/parity/20261027-182533/baseline/` (local), recarregável por
  `scripts/parity/` (`load_baseline` + `run_engine` + `compare_frames`, ADR 0017).

## 4. Não-objetivos e anti-objetivos

**Não-objetivos:**
- Otimização de desempenho antes de todas as outras frentes estarem verdes.
- Features novas (SSO, carteiras compostas, Tesouro por PU, novos benchmarks).
- Code-splitting do bundle de 1,4 MB (desempenho → fica com a otimização, ou adiado).
- Câmbio GBP/GBp para `.L` (ADR 0014) — fica no backlog, salvo se o usuário trouxer caso real.

**Anti-objetivos (falharia mesmo "funcionando"):**
- Mudar um número sem um teste do gabarito que explique **por que** o novo é o certo.
- Mudança de número aparecer contra a linha de base em métrica/janela que a correção **não**
  promete afetar.
- "Consertar" buraco de dados de mercado **inventando** valores (interpolação silenciosa)
  — buraco que não dá para preencher fica visível, não escondido.
- Quebrar o contrato da API v1 (ADR 0013) sem necessidade; se algo mudar, é aditivo.

## 5. O que construir (na ordem)

### 5.1. Rebuild do container (pré-requisito)
`docker compose up -d --build` no LAB para levar `fad1dc8`. Repetir ao fim de cada frente que
o usuário deva testar (§5.6).

### 5.2. Gabarito do engine
Suíte de testes em `backend/engine/tests/` (ex.: `test_gabarito.py`) com carteiras sintéticas
pequenas cuja resposta é calculada à mão e escrita no próprio teste, com o raciocínio em
comentário. Cobertura mínima:
- TWR: aporte único com valorização; aportes intermediários; resgate parcial; proventos;
  zeragem e reentrada.
- XIRR: casos com resposta fechada (ex.: −1000 hoje, +1100 em 1 ano → 10 %); datas fora de
  ordem; carteira grande e pequena (tolerância não pode depender da escala); taxa negativa.
- Drawdown, vol/Sharpe móveis, risco × retorno, tabela anual: ao menos um caso cada.
- Recorte de período: `start_date` no meio do histórico (gap), `end_date` no passado,
  ambos — conferindo `portfolio_df`, KPIs do `/summary`, `/shadow`, `/irr-evolution`.

Os testes dos bugs conhecidos (lucro realizado, `end_date`, `gap`) nascem **vermelhos**. Os
casos de venda incluem venda com lucro, com prejuízo, com e sem reinvestimento.

#### 5.2.1. Oráculo externo: planilha oficial da DLP
A confirmação do fluxo já foi feita (§3.2). Na fase, transformar isso em teste reproduzível:
1. Um teste `integration` (fora do CI, como os de token real) lê o export local
   `dlp_sheet/google_sheets_DLP_html/dashboard..html` e compara, mês a mês, o TWR do engine
   com `TWR % a.m` (tolerância 0,01 pp) e o acumulado com `TWR_acc`. Pulado se o export não
   existir.
2. Fechar a pendência da §3.2 (TIR total).
3. Resultado no ADR **sem valores em R$** (só razões, % e contagens).

A planilha é do usuário (`Cópia de v7.0 DLP Gestão de Investimentos`, Drive id
`1FmO8ix2vTc4zjobf0Mjtb9MFgvjH9RuCw8M1TANqTKc`); o código dela (`.gs` baixando JS ofuscado
do servidor da DLP por `eval`) **não** é desofuscado nem baixado — só os resultados das abas
são usados.

### 5.3. Correções no engine
1. **Fluxo externo com lucro realizado:** `fluxo = Δvlr_investido − lucro − proventos`
   (confirmado pela DLP, §3.2), em um único lugar usado por TWR, XIRR, shadow e KPIs.
   No TWR, aporte líquido no início e resgate líquido no fim do mês (§3.2).
   `ganho` do `/summary` passa a ser retorno total (inclui proventos e lucro realizado).
2. `end_date` cortando `grp`/`portfolio_df`.
3. `fluxo` recalculado após o ajuste de `gap`.
4. Revisão ADR 0012 (`irr.py`, `twr.py`, `metrics.py`): cada ponto vira teste no gabarito;
   corrige-se o que o teste provar errado. Ex.: XIRR com ordenação de datas, critério de
   convergência relativo, fallback robusto (`scipy.optimize.brentq` se já disponível, ou
   bisseção própria) — **escolha do builder**, desde que o gabarito passe.

Após cada correção: rodar a regressão contra a linha de base e registrar quais métricas/janelas
mudaram. Só podem mudar as que a correção promete (ex.: `end_date` afeta só casos com
`end_date` < última data; o `gap` afeta `w3-meio` em `/irr-evolution`). Diferença fora disso
para e investiga.

A linha de base é congelada: após as correções, salvar **uma nova** linha de base (Fase 7) para
a otimização da §5.7 comparar com tolerância zero. Ela sai dos **mesmos** `hist`/`market`
congelados (sem nova coleta na DLP), para não misturar mudança de dado com mudança de código.
A matriz de casos ganha uma janela com `end_date` antes da última data (nenhuma das atuais
exercita o corte); `w2-12m` e `w3-meio` exercitam o `gap`. A correção do fluxo (item 1) e a
do XIRR mudam **todas** as janelas — esperado.

#### 5.3.1. Convenções de mercado (decididas aqui, registradas em ADR)

| Tema | Convenção | Referência |
|---|---|---|
| TWR | retorno ponderado pelo tempo com fluxos externos apenas (aporte/resgate); provento é distribuição (saída) e entra no retorno do período; lucro realizado não é fluxo externo. Com dado mensal, **aporte líquido no início e resgate líquido no fim do mês** — a convenção exata da DLP (§3.2), escolhida para os números baterem com o que o usuário vê na DLP | GIPS + DLP |
| XIRR | Actual/365 a partir do 1º fluxo; datas ordenadas; solver robusto com fallback por intervalo (`brentq` se `scipy` entrar, ou bisseção própria); convergência relativa à escala dos fluxos | Excel `XIRR` / `numpy-financial` |
| TIR que não converge | `null` na API, "não calculável" na SPA (sem número inventado) | — |
| Anualização | 252 dias úteis para dados diários; para a carteira (mensal), retornos mensais × 12 / `sqrt(12)` — ou tornar explícito que a escada diária equivale | ANBIMA / mercado BR |
| Sharpe | taxa livre de risco = **CDI do período** (série real; hoje cai para 10 % fixo se a série faltar — o fixo some, falta de série vira aviso) | padrão BR |
| Ganho no período | retorno total: `pat_fim − pat_ini − aportes_líquidos + proventos + lucro realizado` (coerente com o fluxo) | — |
| Recorte em `start_date` | patrimônio de mercado no 1º dia vira o aporte inicial fictício; aporte real no mesmo dia soma a ele | GIPS (valor de abertura) |
| Buraco de dado de mercado | `ffill` (último valor conhecido, padrão de mercado para preço sem negociação) **mais** aviso aditivo em `meta` com série e intervalo; nunca interpolação | — |

### 5.4. Integridade dos dados de mercado (`apps/market`)
1. **Invalidação do cache:** quando a ingestão grava pontos novos numa série, as chaves
   `market:series:*` afetadas deixam de ser servidas (por versão/geração por série ou delete
   por padrão — escolha do builder; considerar séries derivadas/sintéticas que dependem da
   série base, ex.: `IPCA + 6%` ← IPCA, `S&P 500 BRL` ← S&P × PTAX).
2. **Buracos no meio:** a ingestão incremental detecta lacunas além do esperado para a
   cadência da série (diária útil vs. mensal) e tenta re-buscar o intervalo. Um comando de
   diagnóstico lista os buracos de todas as séries.
3. **Buracos de fonte** (`bcb:12469`, `yf:LFTS11.SA`): tentar re-buscar; se a fonte não tem o
   dado, registrar e **não** interpolar. Tratamento conforme §5.3.1: `ffill` + aviso aditivo
   em `meta` (série e intervalo), exibido pela SPA junto do `PeriodNotice`.

### 5.5. Backlog de bugs da SPA/API
Todos os itens abertos do §2.3 do doc mestre ("Backlog da revisão de 2026-10-27" e "Menores
da revisão final da branch de correções"), exceto os de desempenho (bundle, debounce da busca
— este pode entrar se for trivial). Destaques:
- token DLP rejeitado/ilegível como **estado único de página** (fim das 8 telas de erro iguais
  e do modal reabrindo);
- `PeriodNotice` só com token configurado; "Personalizado" após "Tudo" sem `01/01/1900`;
- `MAX_MONTHS` aceitando exatamente 240 meses;
- corrida no `save_token`, 422 de schema do Ninja fora do envelope `ApiError`, cálculo dentro
  do router `summary`, `start` default quando só `end` vem, filtro DLP ignorado se a coluna
  faltar, lock do `single_flight` sem dono, geração do cache DLP despejável pelo LRU;
- SPA: `location.state.from`, filtros sobrevivendo ao logout, token não salvo persistindo no
  modal, acessibilidade do `Modal`, mensagem de "primeira carga DLP", `Content-Disposition`/
  `revokeObjectURL`, `period` na URL, código morto, layout do PDF, rate limit de login.

Políticas que o backlog não definia (decididas aqui):
- **Rate limit de login:** 5 falhas em 15 min por (usuário, IP), contador no Redis já
  existente, sem dependência nova; resposta `429` no envelope `ApiError`.
- **`period` na URL:** query string `?period=1a|3a|5a|tudo|custom&start=…&end=…` (ISO).
- **PDF:** nenhuma linha de tabela nem gráfico cortado entre páginas.

O builder pode adiar um item com justificativa no doc mestre; adiamentos aparecem na lista
de revisão do usuário ao fim da fase.

### 5.6. Uso real pelo usuário
Pontos de espera no plano: depois da §5.3 e depois da §5.5, o builder faz o rebuild do
container (`docker compose up -d --build`) e **para** até o usuário testar. Achados viram
teste antes de correção (decisão 3) e entram como tasks novas. A fase não fecha com achado
aberto sem decisão.

### 5.7. Otimização (por último)
Pré-parse de `date` antes de `mc.set_df` em `dlp.buscar_historico`; `calculate_twr` não refazer
`to_datetime`/`groupby`/`fluxo` quando recebe o `grp` pronto. Critério: **zero** diferença
contra a linha de base nova (§5.3) e gabarito verde.

### 5.8. Documentação
ADR novo para as decisões que mudam números (recortes, XIRR) e para a invalidação de cache;
checklist de `docs/README.md` ao fechar a fase.

## 6. Restrições
- TDD conforme `backend/CLAUDE.md` (ver falhar antes de implementar); um commit por task,
  Conventional Commits + Gitmoji em português, sem `Co-Authored-By`.
- Contrato da API v1 só com mudanças aditivas.
- Token/credenciais só por env; relatórios versionados sem valores em R$.
- Linha de base e carteira real nunca versionadas (`reports/` fica fora do git).
- `dlp_sheet/` (export da planilha com dados da carteira) **nunca** versionado — entra no
  `.gitignore` na primeira task.
- Acesso à planilha DLP só leitura; o token DLP não é usado para escrever nada (só `GET`).

## 7. Deixado para o builder
- Estrutura exata do gabarito e fórmulas de cada caso.
- Mecanismo de invalidação de cache e de detecção de buracos.
- Algoritmo do XIRR, desde que robusto e coberto pelo gabarito.
- Dividir o plano em subplanos por frente (engine / mercado / SPA-API / otimização), se ajudar.

## 8. Decisões minhas (para o usuário revisar)
- Gabarito como testes pytest em `backend/engine/tests/`, não planilha.
- Salvar uma linha de base nova após as correções, para a otimização comparar com tolerância zero.
- Bundle/code-splitting fora (é desempenho).
- Buraco de fonte que não dá para preencher: `ffill` + aviso visível, nunca interpolado.
- Todas as convenções da §5.3.1 e as políticas da §5.5 (rate limit, `period` na URL, PDF).
- Teste de oráculo contra o export local da planilha DLP (integration, fora do CI).
- TWR com a convenção exata da DLP (aporte no início, resgate no fim do mês), para os números
  do app baterem com os da DLP, em vez de Modified Dietz ou "tudo no início".
- Ordem: o oráculo DLP e o fluxo vêm antes das demais correções do engine, porque mudam
  todos os números.
