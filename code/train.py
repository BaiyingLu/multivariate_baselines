"""
Train and evaluate a Transformer encoder + MLP forecaster on the fixed samples
from prepare_samples.py.

    Baseline A (CGM only):             --features bg
    Baseline B (naive multivariate):   --features bg carbs bolus

Examples:
    python train.py --data_dir <samples/h12_f6_trs1_tes1> --features bg --norm scaled --seed 0
    python train.py --data_dir <...> --features bg carbs bolus --norm none --d_model 128

Outputs (runs_dir/run_name/):
    config.json           arguments, sample counts, parameter count, best epoch, status
    norm_stats.json       normaliser (bg mean / std, covariate references)
    train_log.csv         one row per epoch
    model.pt              best checkpoint (lowest validation loss)
    metrics.json          test metrics in mg/dL (endpoint, per horizon, per dataset, per window)
    per_subject.csv       test RMSE / MAE per subject
    test_predictions.npz  all horizons: y_true, y_pred, subject, dataset, target_ts, target_window
    predictions/<subject>/<subject>_predictions.csv
                          endpoint only, same format as the LSTM baseline
                          (target_timestamp, y_true, y_pred, window)
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import NORM_MODES, Normalizer, load_split
from metrics import evaluate
from model import TransformerForecaster, count_parameters

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_TAGS = {("bg",): "A", ("bg", "carbs", "bolus"): "B"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--data_dir", required=True, help="folder with train/valid/test.npz")
    p.add_argument("--features", nargs="+", default=["bg"],
                   help="input columns from the npz; must include bg (default: bg)")
    p.add_argument("--norm", choices=NORM_MODES, default="scaled")
    # model
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--post_ln", action="store_true", help="use post-LN instead of the default pre-LN")
    # training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    p.add_argument("--lr_patience", type=int, default=3, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=0.0, help="max grad norm; 0 disables")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    # output
    p.add_argument("--runs_dir", default=os.path.join(CODE_DIR, "..", "runs"))
    p.add_argument("--run_name", default=None,
                   help="default: {A|B|features}_{norm}_d{d_model}_h{history}_f{horizon}_seed{seed}")
    p.add_argument("--skip_existing", action="store_true", help="exit if this run already has metrics.json")
    args = p.parse_args()

    if "bg" not in args.features:
        p.error("--features must include bg")
    return args


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA / MPS


def default_run_name(args, history: int, horizon: int) -> str:
    tag = BASELINE_TAGS.get(tuple(args.features), "-".join(args.features))
    return f"{tag}_{args.norm}_d{args.d_model}_h{history}_f{horizon}_seed{args.seed}"


def history_horizon(data_dir: str):
    with np.load(os.path.join(data_dir, "valid.npz")) as d:
        return int(d["X"].shape[1]), int(d["y"].shape[1])


@torch.no_grad()
def predict(model, X: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    history, horizon = history_horizon(args.data_dir)
    run_name = args.run_name or default_run_name(args, history, horizon)
    out_dir = os.path.abspath(os.path.join(args.runs_dir, run_name))

    if args.skip_existing and os.path.exists(os.path.join(out_dir, "metrics.json")):
        print(f"[skip] {run_name} already finished")
        return
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"Run      : {run_name}")
    print(f"Device   : {device}")
    print(f"Features : {args.features}   norm: {args.norm}")

    # ── data ────────────────────────────────────────────────────────────────
    splits = {s: load_split(args.data_dir, s, args.features) for s in ("train", "valid", "test")}
    norm = Normalizer.fit(args.norm, args.features, splits["train"])

    def to_device(split):
        X = torch.from_numpy(norm.transform_x(splits[split]["X"])).to(device)
        y = torch.from_numpy(norm.transform_y(splits[split]["y"])).to(device)
        return X, y

    X_train, y_train = to_device("train")
    X_valid, y_valid = to_device("valid")
    X_test, _ = to_device("test")
    n_samples = {s: int(len(splits[s]["y"])) for s in splits}
    print(f"Samples  : {n_samples}   history {history}, horizon {horizon}")

    # ── model ───────────────────────────────────────────────────────────────
    model_kwargs = dict(
        c_in=len(args.features), t_in=history, t_out=horizon, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, ffn_mult=args.ffn_mult,
        dropout=args.dropout, norm_first=not args.post_ln,
    )
    model = TransformerForecaster(**model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"Params   : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr,
    )

    config = {
        "run_name": run_name, **vars(args), "data_dir": os.path.abspath(args.data_dir),
        "history": history, "horizon": horizon, "n_samples": n_samples, "n_params": n_params,
        "model_kwargs": model_kwargs, "device": str(device), "torch_version": torch.__version__,
    }
    write_json(os.path.join(out_dir, "config.json"), config)
    write_json(os.path.join(out_dir, "norm_stats.json"), norm.to_dict())

    # ── training ────────────────────────────────────────────────────────────
    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,valid_loss,valid_rmse_endpoint_mgdl,lr,seconds\n")

    y_valid_mgdl = splits["valid"]["y"][:, -1]
    best_loss, best_epoch, best_state = float("inf"), 0, None
    bad_epochs, status = 0, "max_epochs"
    n_train = len(X_train)
    t_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = torch.zeros((), device=device)
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = F.mse_loss(model(X_train[idx]), y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += loss.detach() * len(idx)
        train_loss = (loss_sum / n_train).item()

        pred_valid = predict(model, X_valid, args.eval_batch_size)
        valid_loss = F.mse_loss(pred_valid, y_valid).item()
        pred_valid_mgdl = norm.inverse_y(pred_valid[:, -1].cpu().numpy())
        valid_rmse = float(np.sqrt(np.mean((pred_valid_mgdl - y_valid_mgdl) ** 2)))
        lr = optimizer.param_groups[0]["lr"]
        seconds = time.time() - t0

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6g},{valid_loss:.6g},{valid_rmse:.4f},{lr:.3g},{seconds:.1f}\n")
        print(f"epoch {epoch:3d} | train {train_loss:.5g} | valid {valid_loss:.5g} | "
              f"valid RMSE@{horizon * 5}min {valid_rmse:6.2f} mg/dL | lr {lr:.1e} | {seconds:.1f}s")

        if not (np.isfinite(train_loss) and np.isfinite(valid_loss)):
            status = "diverged"
            print("Loss is not finite — stopping.")
            break

        scheduler.step(valid_loss)
        if valid_loss < best_loss:
            best_loss, best_epoch, bad_epochs = valid_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"state_dict": best_state, "model_kwargs": model_kwargs,
                        "normalizer": norm.to_dict(), "epoch": epoch},
                       os.path.join(out_dir, "model.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                status = "early_stopped"
                break

    train_seconds = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── test evaluation (mg/dL) ─────────────────────────────────────────────
    test = splits["test"]
    y_pred = norm.inverse_y(predict(model, X_test, args.eval_batch_size).cpu().numpy())
    metrics, per_subject = evaluate(test["y"], y_pred, test["last_bg"],
                                    test["subject"], test["dataset"], test["target_window"])
    metrics.update(run_name=run_name, best_epoch=best_epoch, status=status)

    write_json(os.path.join(out_dir, "metrics.json"), metrics)
    per_subject.to_csv(os.path.join(out_dir, "per_subject.csv"), index=False)
    np.savez(os.path.join(out_dir, "test_predictions.npz"),
             y_true=test["y"], y_pred=y_pred.astype(np.float32), subject=test["subject"],
             dataset=test["dataset"], target_ts=test["target_ts"], target_window=test["target_window"])

    pred_df = pd.DataFrame({
        "target_timestamp": pd.to_datetime(test["target_ts"]).strftime("%Y-%m-%d %H:%M:%S"),
        "y_true": test["y"][:, -1], "y_pred": y_pred[:, -1], "window": test["target_window"],
        "subject": test["subject"],
    })
    for subject, g in pred_df.groupby("subject", sort=True):
        subj_dir = os.path.join(out_dir, "predictions", subject)
        os.makedirs(subj_dir, exist_ok=True)
        g.drop(columns="subject").to_csv(
            os.path.join(subj_dir, f"{subject}_predictions.csv"), index=False, float_format="%.4f")

    config.update(best_epoch=best_epoch, best_valid_loss=best_loss, status=status,
                  train_seconds=round(train_seconds, 1))
    write_json(os.path.join(out_dir, "config.json"), config)

    o = metrics["overall"]
    print(f"\n[{run_name}] status={status} best_epoch={best_epoch} "
          f"test RMSE@{metrics['endpoint_minutes']}min={o['rmse']:.2f}  MAE={o['mae']:.2f}  "
          f"(persistence RMSE={o['persistence_rmse']:.2f})  → {out_dir}")


if __name__ == "__main__":
    main()
