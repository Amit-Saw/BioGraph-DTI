#!/usr/bin/env bash
# Production server (gunicorn, one worker, 8 threads).  Reads .env if present.
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
if [ -f .env ]; then set -a; . ./.env; set +a; fi
exec gunicorn -c gunicorn.conf.py wsgi:app
