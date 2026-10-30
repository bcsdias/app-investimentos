# Segurança — dados pessoais e segredos

> O repositório é **público de propósito** (portfólio). Tudo que entra num commit — inclusive
> em commits antigos, mesmo que o arquivo seja apagado depois — fica visível para qualquer um.

## O que nunca pode ser versionado

| Categoria | Exemplos | Onde fica de verdade |
|---|---|---|
| **Dados da carteira** | posições, quantidades, PM, corretora, proventos, movimentações, histórico da API DLP, relatórios/gráficos gerados a partir dela | `privado/` (ignorado) ou fora do repo |
| **Segredos** | token DLP, `FERNET_KEY`, `DJANGO_SECRET_KEY`, senha do Postgres, chaves Supabase/Upstash, OAuth Google | `backend/.env`, `.streamlit/secrets.toml` (ignorados) |
| **Logs e dumps de erro** | `log/`, `*.log`, `erros.txt`, `log.txt` | só local |
| **Rastros da máquina** | caminhos `/home/<usuário>/`, hostnames, IPs internos | — |

Dados de **mercado público** (`data/raw/`, `data/static/`, `ativos.csv`) podem ser versionados.

## Convenções

- **Dado pessoal local vai em `privado/`** na raiz — a pasta inteira é ignorada pelo git.
- **Fixtures de teste são sempre sintéticas** (`r@example.com`, valores inventados). Nunca
  copiar uma resposta real da API DLP para um teste.
- **Só `*.example` dos arquivos de segredo** é versionado, e só com placeholders
  (`SUA_CHAVE_FERNET_AQUI`).
- O bloco *"Dados pessoais — NUNCA versionar"* do [`.gitignore`](../.gitignore) cobre os
  nomes já usados no passado (`*Bruno*`, `*movimentac*`, `carteira*.json`, `reports/`,
  `*.zip`, `*.xlsx`…). Arquivo novo com dado pessoal e nome diferente → acrescentar a regra.

## Checklist antes de commitar

```bash
git status                       # algum arquivo inesperado?
git diff --cached --stat         # o que REALMENTE vai no commit
git check-ignore -v <arquivo>    # confirma que um arquivo sensível está ignorado
```

Evitar `git add -A` / `git add .` com arquivos soltos na raiz; preferir adicionar por caminho.
Cuidado com `git add -f`: ele ignora o `.gitignore`.

## Se algo vazar

Apagar o arquivo num commit novo **não resolve** — ele continua no histórico.

1. **Segredo vazou?** Rotacionar **primeiro** (gerar chave/token novo e invalidar o antigo).
   Para a senha do Postgres há `backend/scripts/rotate_db_password.sh`. Reescrever o
   histórico não desfaz o vazamento de um segredo; rotacionar desfaz.
2. **Backup** do estado atual: `git clone --mirror <url> bkp.git` +
   `git -C bkp.git bundle create bkp.bundle --all`. Guardar fora do repo — ele contém o vazamento.
3. **Reescrever** num clone `--mirror` fresco com
   [`git-filter-repo`](https://github.com/newren/git-filter-repo):
   `git filter-repo --invert-paths --paths-from-file caminhos.txt` (um caminho por linha;
   `glob:*padrão*` para curingas). Tags são reescritas junto.
4. **Verificar** antes de publicar: `git log --all --name-only | grep <padrão>` vazio, mesmas
   branches/tags, `git diff --stat` entre o HEAD antigo e o novo só com as remoções esperadas.
5. **Force push** de tudo: `git push --force <url> 'refs/heads/*:refs/heads/*' 'refs/tags/*:refs/tags/*'`.
6. **Ressincronizar** os clones de trabalho (`git fetch --force --prune --tags` +
   `git reset --hard origin/<branch>`) e **não** dar push de clones antigos — o histórico
   sujo voltaria. Forks e caches do GitHub mantêm os dados antigos: conferir *Insights → Forks*
   e, se necessário, pedir purga ao GitHub Support.

## Registro — limpeza de 2026-10-20

Auditoria dos 161 commits encontrou **dados financeiros pessoais** (nenhuma credencial).
Removidos de todo o histórico com `git-filter-repo`:

- `reports/`, `reports.zip`
- `dlombello/planilha_dlombello/`, `dlombello_old/planilha_dlombello/`,
  `apoio/dlombello_old/planilha_dlombello/`, `dlombello/graficos/`
- `comparativos/movimentacoes.xlsx`, `apoio/comparativos/movimentacoes.xlsx`
- `historico_completo.txt`, `hist.txt` (e as cópias em `apoio/`), `apoio/carteira.json`
- `log/`, `erros.txt`, `log.txt` (este expunha o usuário da máquina)

O código exploratório antigo (`apoio/**/*.py`, `dlombello/*.py`, `app/`) foi preservado.

Na sequência, uma segunda reescrita (`git filter-repo --mailmap`) trocou o autor/committer
de **todos** os commits — 120 estavam com o e-mail corporativo, o resto com o e-mail
pessoal — por `Bruno Dias <1091333+bcsdias@users.noreply.github.com>`. O conteúdo das
branches ficou idêntico; só os metadados mudaram.

**Todos os SHAs mudaram** — referências antigas a commits não valem mais:

| Ref | Original | Após remover arquivos | Após mailmap (atual) |
|---|---|---|---|
| `main` / tag `legado-pre-src` | `4d048a9` | `520771e` | `315893e` |
| `legado` / tag `legado-streamlit-v3.5` | `d09e2a3` | `171b9f4` | `18f1015` |
| `migracao-django` | `13c8ccd` | `d7ca615` | `e2a99f7` (+ commits seguintes) |

## Identidade nos commits

A conta do GitHub tem **"Keep my email addresses private"** e **"Block command line pushes
that expose my email"** ativados: um push com commit assinado por e-mail real é **recusado**.
Por isso, o git deve estar configurado com o e-mail noreply em toda máquina usada:

```bash
git config --global user.name  "Bruno Dias"
git config --global user.email "1091333+bcsdias@users.noreply.github.com"
git config user.email            # conferir dentro do repo (não deve haver override local)
```

Nunca commitar com a identidade da máquina do trabalho (e-mail corporativo).
