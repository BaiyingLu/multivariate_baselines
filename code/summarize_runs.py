"""
Collect every finished run under runs_dir into two tables:

    summary.csv            one row per run (config + test metrics)
    summary_by_config.csv  mean ± std across seeds for each
                           (features, norm, d_model, history, horizon)

Usage:
    python summarize_runs.py --runs_dir ../runs
"""

import argparse
import glob
import json
import os

import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_KEYS = ["features", "norm", "d_model", "history", "horizon"]


def load_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)

    row = {
        "run_name": cfg["run_name"], "features": " ".join(cfg["features"]), "norm": cfg["norm"],
        "d_model": cfg["d_model"], "history": cfg["history"], "horizon": cfg["horizon"],
        "seed": cfg["seed"], "n_params": cfg["n_params"], "status": m["status"],
        "best_epoch": m["best_epoch"], "train_min": round(cfg.get("train_seconds", float("nan")) / 60, 1),
        "rmse": m["overall"]["rmse"], "mae": m["overall"]["mae"],
        "subject_mean_rmse": m["overall"]["subject_mean_rmse"],
        "persistence_rmse": m["overall"]["persistence_rmse"],
    }
    for ds, v in m["by_dataset"].items():
        row[f"rmse_{ds}"] = v["rmse"]
    for w, v in m["by_window_agg"].items():
        row[f"rmse_{w}"] = v["rmse"]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=os.path.join(CODE_DIR, "..", "runs"))
    args = p.parse_args()

    run_dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(args.runs_dir, "*", "metrics.json")))
    if not run_dirs:
        print(f"No finished runs in {args.runs_dir}")
        return

    runs = pd.DataFrame([load_run(d) for d in run_dirs]).sort_values(CONFIG_KEYS + ["seed"])
    runs.to_csv(os.path.join(args.runs_dir, "summary.csv"), index=False)

    metric_cols = [c for c in runs.columns if c.startswith("rmse") or c in ("mae", "subject_mean_rmse")]
    grouped = runs.groupby(CONFIG_KEYS, sort=False)
    by_config = grouped[metric_cols].agg(["mean", "std"])
    by_config.columns = [f"{m}_{s}" for m, s in by_config.columns]
    by_config.insert(0, "n_seeds", grouped.size())
    by_config.insert(1, "n_params", grouped["n_params"].first())
    by_config.insert(2, "persistence_rmse", grouped["persistence_rmse"].first())
    by_config = by_config.reset_index()
    by_config.to_csv(os.path.join(args.runs_dir, "summary_by_config.csv"), index=False)

    view = by_config[CONFIG_KEYS + ["n_seeds", "n_params"]].copy()
    for m in ("rmse", "mae", "subject_mean_rmse"):
        view[m] = by_config[f"{m}_mean"].map("{:.2f}".format) + " ± " + by_config[f"{m}_std"].fillna(0).map("{:.2f}".format)
    view["persistence_rmse"] = by_config["persistence_rmse"].map("{:.2f}".format)
    pd.set_option("display.width", 200)
    print(view.to_string(index=False))
    print(f"\nSaved {len(runs)} runs → {args.runs_dir}/summary.csv, summary_by_config.csv")


if __name__ == "__main__":
    main()
