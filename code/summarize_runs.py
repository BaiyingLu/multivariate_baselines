"""
Collect every finished run under one or more runs directories into two tables:

    {prefix}summary.csv            one row per run (config + test metrics)
    {prefix}summary_by_config.csv  mean ± std across seeds for each
                                   (features, norm, d_model, history, horizon)

Tables are written to --out_dir (default: the first --runs_dir).

Usage:
    python summarize_runs.py --runs_dir ../runs
    # combined view written next to the new runs, leaving runs/summary*.csv untouched
    python summarize_runs.py --runs_dir ../runs ../runs_c_bsteps --out_dir ../runs_c_bsteps --prefix combined_
"""

import argparse
import glob
import json
import os

import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# "samples" (the sample-set folder name) keeps aligned and non-aligned sample sets apart
CONFIG_KEYS = ["features", "norm", "d_model", "history", "horizon", "samples"]
PIVOT_METRICS = ["rmse", "rmse_30min", "rmse_MEAL", "rmse_Mix", "rmse_BASELINE+NOCTURNAL"]


def load_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)

    row = {
        "run_name": cfg["run_name"], "features": " ".join(cfg["features"]), "norm": cfg["norm"],
        "d_model": cfg["d_model"], "history": cfg["history"], "horizon": cfg["horizon"],
        "samples": os.path.basename(os.path.normpath(cfg["data_dir"])),
        "seed": cfg["seed"], "n_params": cfg["n_params"], "status": m["status"],
        "best_epoch": m["best_epoch"], "train_min": round(cfg.get("train_seconds", float("nan")) / 60, 1),
        "rmse": m["overall"]["rmse"], "mae": m["overall"]["mae"],
        "subject_mean_rmse": m["overall"]["subject_mean_rmse"],
        "persistence_rmse": m["overall"]["persistence_rmse"],
        # step-wise RMSE, so a 60-min model's 30-min accuracy can be compared with a 30-min model
        "rmse_30min": m["per_horizon_rmse"].get("30", float("nan")),
        "rmse_60min": m["per_horizon_rmse"].get("60", float("nan")),
    }
    for ds, v in m["by_dataset"].items():
        row[f"rmse_{ds}"] = v["rmse"]
    for w, v in m["by_window_agg"].items():
        row[f"rmse_{w}"] = v["rmse"]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", nargs="+", default=[os.path.join(CODE_DIR, "..", "runs")],
                   help="one or more folders containing run subfolders")
    p.add_argument("--out_dir", default=None, help="where to write the tables (default: first --runs_dir)")
    p.add_argument("--prefix", default="", help="filename prefix for the output tables")
    p.add_argument("--pivot", action="store_true",
                   help="also print metric tables with rows (horizon, features) and columns history")
    args = p.parse_args()
    out_dir = args.out_dir or args.runs_dir[0]
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"{args.prefix}summary.csv")
    by_config_path = os.path.join(out_dir, f"{args.prefix}summary_by_config.csv")

    run_dirs = sorted(os.path.dirname(f) for rd in args.runs_dir
                      for f in glob.glob(os.path.join(rd, "*", "metrics.json")))
    if not run_dirs:
        print(f"No finished runs in {args.runs_dir}")
        return

    runs = pd.DataFrame([load_run(d) for d in run_dirs]).sort_values(CONFIG_KEYS + ["seed"])
    runs.to_csv(summary_path, index=False)

    metric_cols = [c for c in runs.columns if c.startswith("rmse") or c in ("mae", "subject_mean_rmse")]
    grouped = runs.groupby(CONFIG_KEYS, sort=False)
    by_config = grouped[metric_cols].agg(["mean", "std"])
    by_config.columns = [f"{m}_{s}" for m, s in by_config.columns]
    by_config.insert(0, "n_seeds", grouped.size())
    by_config.insert(1, "n_params", grouped["n_params"].first())
    by_config.insert(2, "persistence_rmse", grouped["persistence_rmse"].first())
    by_config = by_config.reset_index()
    by_config.to_csv(by_config_path, index=False)

    shown_keys = [k for k in CONFIG_KEYS if k != "samples" or by_config["samples"].nunique() > 1]
    view = by_config[shown_keys + ["n_seeds", "n_params"]].copy()
    for m in ("rmse", "mae", "subject_mean_rmse"):
        view[m] = by_config[f"{m}_mean"].map("{:.2f}".format) + " ± " + by_config[f"{m}_std"].fillna(0).map("{:.2f}".format)
    view["persistence_rmse"] = by_config["persistence_rmse"].map("{:.2f}".format)
    pd.set_option("display.width", 200)
    print(view.to_string(index=False))

    if args.pivot:
        for m in PIVOT_METRICS:
            if f"{m}_mean" not in by_config:
                continue
            table = by_config.pivot_table(index=["norm", "d_model", "horizon", "features"],
                                          columns="history", values=f"{m}_mean")
            print(f"\n{m} (mean over seeds) — columns: history steps")
            print(table.round(2).to_string())

    print(f"\nSaved {len(runs)} runs → {summary_path}, {by_config_path}")


if __name__ == "__main__":
    main()
