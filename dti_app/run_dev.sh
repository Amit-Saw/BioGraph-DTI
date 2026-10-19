#!/usr/bin/env bash
# Development server (single process, http://127.0.0.1:5000)
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
python app.py
