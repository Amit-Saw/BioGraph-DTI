"""HTML pages."""
import json

import pandas as pd
from flask import Blueprint, current_app, render_template

import config

bp = Blueprint("pages", __name__)


@bp.get("/")
def home():
    return render_template("index.html", active="home")


@bp.get("/predict")
def predict():
    return render_template("predict.html", active="predict")


@bp.get("/batch")
def batch():
    return render_template("batch.html", active="batch")


@bp.get("/screen")
def screen():
    return render_template("screen.html", active="screen")


@bp.get("/compare")
def compare():
    return render_template("compare.html", active="compare")


@bp.get("/dock")
def dock():
    return render_template("dock.html", active="dock")


@bp.get("/about")
def about():
    table, cols = [], []
    f = config.RESULTS_DIR / "comparison_table.csv"
    if f.exists():
        df = pd.read_csv(f)
        cols = [c for c in ("endpoint", "split", "model", "epochs_run", "n_test", "RMSE", "MAE", "R2",
                            "Pearson", "Spearman") if c in df.columns]
        table = df[cols].fillna("").to_dict("records")
    calib = None
    cf = config.RESULTS_DIR / "applicability.json"
    if cf.exists():
        try:
            calib = json.loads(cf.read_text())
        except Exception:
            calib = None
    plots_dir = config.BASE_DIR / "static" / "img" / "plots"
    plots = sorted(p.name for p in plots_dir.glob("*.png")) if plots_dir.exists() else []
    reg = current_app.registry
    return render_template("about.html", active="about", table=table, cols=cols, calib=calib, plots=plots,
                           models=reg.available() if reg else [],
                           thresholds={"high": config.SIM_HIGH, "medium": config.SIM_MID})
