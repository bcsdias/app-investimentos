#!/usr/bin/env bash
# Cria o role e o database 'appinvest' no container lab-postgres (idempotente).
# Requer acesso ao daemon Docker do host.
#
# Uso:
#   APPINVEST_DB_PASSWORD='...' bash backend/scripts/bootstrap_db.sh
set -euo pipefail

CONTAINER="${LAB_POSTGRES_CONTAINER:-lab-postgres}"
SUPERUSER="${POSTGRES_SUPERUSER:-postgres}"
DB_NAME="appinvest"
DB_ROLE="appinvest"
DB_PASS="${APPINVEST_DB_PASSWORD:?defina APPINVEST_DB_PASSWORD}"

run_psql() { docker exec -i "$CONTAINER" psql -v ON_ERROR_STOP=1 -U "$SUPERUSER" "$@"; }

echo ">> Garantindo role '$DB_ROLE'..."
if [ "$(run_psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_ROLE'")" = "1" ]; then
  run_psql -c "ALTER ROLE $DB_ROLE LOGIN CREATEDB PASSWORD '$DB_PASS';"
  echo "   role já existia; atributos/senha atualizados."
else
  run_psql -c "CREATE ROLE $DB_ROLE LOGIN CREATEDB PASSWORD '$DB_PASS';"
  echo "   role criado."
fi

echo ">> Garantindo database '$DB_NAME'..."
if [ "$(run_psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'")" = "1" ]; then
  echo "   database já existia."
else
  run_psql -c "CREATE DATABASE $DB_NAME OWNER $DB_ROLE;"
  echo "   database criado."
fi

run_psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_ROLE;"
echo ">> OK: $DB_ROLE@$DB_NAME pronto."
