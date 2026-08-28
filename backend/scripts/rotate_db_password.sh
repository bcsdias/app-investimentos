#!/usr/bin/env bash
# Rotaciona a senha do role/database 'appinvest'.
#
# O que faz:
#   1. gera uma senha forte (chars A-Za-z0-9-_, seguros para SQL e dotenv);
#   2. aplica no Postgres chamando bootstrap_db.sh (idempotente: faz ALTER ROLE);
#   3. espelha a MESMA senha em backend/.env (APPINVEST_DB_PASSWORD=).
#
# A senha nao e impressa em momento nenhum. Ela fica so no role do Postgres
# e no backend/.env (que e git-ignored).
#
# Uso (de qualquer diretorio):
#   bash /data/projetos/app-investimentos/backend/scripts/rotate_db_password.sh
#
# Depois, verifique:
#   cd /data/projetos/app-investimentos/backend
#   .venv/bin/python manage.py migrate --check
#   .venv/bin/pytest -q
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PY="$BACKEND_DIR/.venv/bin/python"
ENV_FILE="$BACKEND_DIR/.env"

[ -x "$PY" ]       || { echo "ERRO: $PY nao encontrado. Crie o venv primeiro."; exit 1; }
[ -f "$ENV_FILE" ] || { echo "ERRO: $ENV_FILE nao existe."; exit 1; }

echo ">> Gerando nova senha..."
NEW_DB_PASS="$("$PY" -c 'import secrets; print(secrets.token_urlsafe(24))')"

echo ">> Aplicando no Postgres (ALTER ROLE via bootstrap_db.sh)..."
APPINVEST_DB_PASSWORD="$NEW_DB_PASS" bash "$SCRIPT_DIR/bootstrap_db.sh"

echo ">> Espelhando a senha em $ENV_FILE ..."
NEW_DB_PASS="$NEW_DB_PASS" "$PY" - "$ENV_FILE" <<'PY'
import os, sys, pathlib
new = os.environ["NEW_DB_PASS"]
p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines()
found = False
out = []
for ln in lines:
    if ln.startswith("APPINVEST_DB_PASSWORD="):
        out.append(f"APPINVEST_DB_PASSWORD={new}")
        found = True
    else:
        out.append(ln)
if not found:
    raise SystemExit("ERRO: linha 'APPINVEST_DB_PASSWORD=' nao encontrada no .env")
p.write_text("\n".join(out) + "\n")
print("   APPINVEST_DB_PASSWORD atualizado.")
PY

unset NEW_DB_PASS
echo ">> OK. Senha rotacionada. Agora verifique:"
echo "     cd $BACKEND_DIR"
echo "     .venv/bin/python manage.py migrate --check"
echo "     .venv/bin/pytest -q"
