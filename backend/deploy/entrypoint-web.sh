#!/bin/sh
set -e

python manage.py migrate --noinput
python manage.py collectstatic --noinput
exec gunicorn config.wsgi \
  --bind 0.0.0.0:8000 \
  --workers 2 \
  --worker-class gthread \
  --threads 4 \
  --timeout 180 \
  --access-logfile - \
  --error-logfile -

