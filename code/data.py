"""
Load forecasting samples produced by
multivariate_data_preprocessing/sample_prepare/prepare_samples.py and apply
one of two normalisation schemes.

    none    raw values for inputs and targets (bg / y in mg/dL)
    scaled  bg and y      : global z-score, mean / std from the train split only
            carbs         : log1p(x) / log1p(80)
            bolus         : log1p(x) / log1p(15)
            total_insulin : log1p(x) / log1p(15)
            steps         : log1p(x) / log1p(250)
            smoothed_step : log1p(x) / log1p(150)
            iob           : clip(x, 0, 10) / 10
            cob           : clip(x, 0, 60) / 60
            basal         : clip(x, 0, 0.5) / 0.5

The covariate references match the LSTM baseline loaders, so zeros (no event)
stay exactly zero. Because bg / y use one global affine map, MSE in the scaled
space is proportional to MSE in mg/dL.
"""

import os
from dataclasses import asdict, dataclass

import numpy as np

# (transform, reference value) for every covariate in the npz files
COVARIATE_SCALING = {
    "carbs":         ("log1p", 80.0),
    "bolus":         ("log1p", 15.0),
    "total_insulin": ("log1p", 15.0),
    "steps":         ("log1p", 250.0),
    "smoothed_step": ("log1p", 150.0),
    "iob":           ("clip", 10.0),
    "cob":           ("clip", 60.0),
    "basal":         ("clip", 0.5),
}

NORM_MODES = ("none", "scaled")


def load_split(data_dir: str, split: str, features: list) -> dict:
    """Load one split and keep only the requested feature columns (in the given order)."""
    path = os.path.join(data_dir, f"{split}.npz")
    with np.load(path) as d:
        names = [str(n) for n in d["feature_names"]]
        missing = [f for f in features if f not in names]
        if missing:
            raise ValueError(f"Features {missing} not in {path}; available: {names}")

        X_all = d["X"]
        return {
            "X": np.ascontiguousarray(X_all[..., [names.index(f) for f in features]], dtype=np.float32),
            "y": d["y"].astype(np.float32),
            "last_bg": X_all[:, -1, names.index("bg")].astype(np.float32),  # for the persistence reference
            "subject": d["subject"],
            "dataset": d["dataset"],
            "target_ts": d["target_ts"],
            "target_window": d["target_window"],
        }


@dataclass
class Normalizer:
    mode: str
    features: list
    bg_mean: float = 0.0
    bg_std: float = 1.0

    @classmethod
    def fit(cls, mode: str, features: list, train: dict) -> "Normalizer":
        if mode not in NORM_MODES:
            raise ValueError(f"norm must be one of {NORM_MODES}, got '{mode}'")
        unknown = [f for f in features if f != "bg" and f not in COVARIATE_SCALING]
        if unknown:
            raise ValueError(f"No scaling rule defined for {unknown}")
        if mode == "none":
            return cls(mode, list(features))

        bg = train["X"][..., features.index("bg")].astype(np.float64)
        return cls(mode, list(features), bg_mean=float(bg.mean()), bg_std=float(bg.std()))

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return X
        out = np.empty_like(X)
        for c, name in enumerate(self.features):
            x = X[..., c]
            if name == "bg":
                out[..., c] = (x - self.bg_mean) / self.bg_std
                continue
            kind, ref = COVARIATE_SCALING[name]
            x = np.clip(x, 0.0, None)
            if kind == "log1p":
                out[..., c] = np.log1p(x) / np.log1p(ref)
            else:
                out[..., c] = np.clip(x, 0.0, ref) / ref
        return out

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return y
        return ((y - self.bg_mean) / self.bg_std).astype(np.float32)

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return y
        return y * self.bg_std + self.bg_mean

    def to_dict(self) -> dict:
        out = asdict(self)
        if self.mode == "scaled":
            out["covariate_scaling"] = {f: COVARIATE_SCALING[f] for f in self.features if f != "bg"}
        return out
