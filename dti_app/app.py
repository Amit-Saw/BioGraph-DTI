"""Flask application factory.

Run (development):  python app.py
Run (production):   gunicorn -c gunicorn.conf.py wsgi:app      (keep ONE worker: models are loaded once)
"""
import logging
import os
import threading

from flask import Flask, jsonify, request

import config
from services.errors import UserError
from services.jobs import JobManager

log = logging.getLogger("dti")


def create_app(load_models=None, registry=None, reference=None):
    """load_models=False skips loading checkpoints (tests); `registry`/`reference` allow injecting fakes."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    app.json.sort_keys = False
    for d in (config.CACHE_DIR, config.UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if load_models is None:
        load_models = os.environ.get("DTI_LOAD_MODELS", "1") == "1"

    app.registry = registry
    if app.registry is None and load_models:
        from dti.protein import ESMEmbedder, EmbeddingCache
        from dti.registry import ModelRegistry
        app.registry = ModelRegistry(EmbeddingCache(ESMEmbedder()))
        n = app.registry.load_all()
        print(f"[dti] loaded {n} model(s) from {config.CHECKPOINT_DIR}")
        if n == 0:
            print("[dti] WARNING: no checkpoints found - predictions are disabled. See QUICKSTART.md")

    from services.chem import ReferenceIndex
    app.reference = reference if reference is not None else ReferenceIndex()
    app.jobs = JobManager(workers=config.JOB_WORKERS)

    # build the similarity index in the background so the first request is not slow
    if reference is None and load_models and os.environ.get("DTI_PRELOAD_REFERENCE", "1") == "1":
        def warm():
            for ep in config.ENDPOINTS:
                try:
                    if app.reference.available(ep):
                        app.reference._load(ep)
                        print(f"[dti] reference index ready for {ep}")
                except Exception as e:                          # noqa: BLE001
                    print(f"[dti] could not build reference index for {ep}: {e}")
        threading.Thread(target=warm, daemon=True).start()

    from routes.api import bp as api_bp
    from routes.pages import bp as pages_bp
    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    @app.get("/health")
    def health():
        n = 0 if app.registry is None else len(app.registry.models)
        return jsonify(status="ok", models_loaded=n)

    @app.errorhandler(UserError)
    def user_error(e):
        return jsonify(error=str(e)), e.status

    @app.errorhandler(413)
    def too_big(_):
        return jsonify(error=f"Upload larger than {config.MAX_UPLOAD_MB} MB."), 413

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return jsonify(error="Not found."), 404
        return "Page not found.", 404

    @app.errorhandler(Exception)
    def unexpected(e):
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return e
        log.exception("unhandled error")
        if request.path.startswith("/api/"):
            return jsonify(error="Internal server error. Check the server log."), 500
        return "Internal server error.", 500

    return app


if __name__ == "__main__":
    create_app().run(debug=False, host=config.HOST, port=config.PORT, threaded=True)
