"""Production server settings:  gunicorn -c gunicorn.conf.py wsgi:app

ONE worker on purpose: the models (and the in-memory job list) live in the worker process.
Concurrency comes from threads."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("DTI_THREADS", "8"))
timeout = 600            # first request may need to download / run ESM-2
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
