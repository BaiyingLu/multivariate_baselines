"""
Test-set metrics in mg/dL. The primary target is the endpoint y[:, -1]
(horizon × 5 min ahead); per-horizon RMSE covers every step.

Pooled metrics use all forecasts together; subject_mean_* average the
per-subject values so large subjects (e.g. HUPA0027P) do not dominate.
"""

import numpy as np
import pandas as pd

STEP_MINUTES = 5

# 8 event windows → 5 aggregated buckets (same as the LSTM slide figures)
WINDOW_AGG = {
    "STABLE_BASELINE": "BASELINE+NOCTURNAL",
    "NOCTURNAL": "BASELINE+NOCTURNAL",
    "MEAL": "MEAL",
    "BOLUS": "BOLUS",
    "EXERCISE": "EXERCISE",
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _group_metrics(df: pd.DataFrame, key: str) -> dict:
    out = {}
    for name, g in df.groupby(key, sort=True):
        out[str(name)] = {
            "n": int(len(g)),
            "rmse": rmse(g.y_true, g.y_pred),
            "mae": mae(g.y_true, g.y_pred),
            "persistence_rmse": rmse(g.y_true, g.last_bg),
        }
    return out


def per_subject_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, dataset), g in df.groupby(["subject", "dataset"], sort=True):
        rows.append({
            "subject": subject, "dataset": dataset, "n": len(g),
            "rmse": rmse(g.y_true, g.y_pred), "mae": mae(g.y_true, g.y_pred),
            "persistence_rmse": rmse(g.y_true, g.last_bg),
        })
    return pd.DataFrame(rows)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, last_bg: np.ndarray,
             subject: np.ndarray, dataset: np.ndarray, window: np.ndarray):
    """y_true / y_pred: (N, horizon) in mg/dL. Returns (metrics dict, per-subject DataFrame)."""
    horizon = y_true.shape[1]
    df = pd.DataFrame({
        "y_true": y_true[:, -1], "y_pred": y_pred[:, -1], "last_bg": last_bg,
        "subject": subject, "dataset": dataset, "window": window,
    })
    df["window_agg"] = df["window"].map(WINDOW_AGG).fillna("Mix")
    subjects = per_subject_table(df)

    by_dataset = _group_metrics(df, "dataset")
    for ds, m in by_dataset.items():
        s = subjects[subjects.dataset == ds]
        m["subject_mean_rmse"] = float(s.rmse.mean())
        m["n_subjects"] = int(len(s))

    metrics = {
        "endpoint_minutes": horizon * STEP_MINUTES,
        "overall": {
            "n": int(len(df)),
            "n_subjects": int(len(subjects)),
            "rmse": rmse(df.y_true, df.y_pred),
            "mae": mae(df.y_true, df.y_pred),
            "subject_mean_rmse": float(subjects.rmse.mean()),
            "subject_mean_mae": float(subjects.mae.mean()),
            "persistence_rmse": rmse(df.y_true, df.last_bg),
            "persistence_mae": mae(df.y_true, df.last_bg),
        },
        "per_horizon_rmse": {
            str((h + 1) * STEP_MINUTES): rmse(y_true[:, h], y_pred[:, h]) for h in range(horizon)
        },
        "per_horizon_mae": {
            str((h + 1) * STEP_MINUTES): mae(y_true[:, h], y_pred[:, h]) for h in range(horizon)
        },
        "by_dataset": by_dataset,
        "by_window": _group_metrics(df, "window"),
        "by_window_agg": _group_metrics(df, "window_agg"),
    }
    return metrics, subjects
