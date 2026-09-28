# Transformer Baselines A / B — Result Summary

Last updated: 2026-09-28 · 18 runs (3 seeds × 6 configurations), all finished with early stopping.
Pending: Baseline C (`bg iob cob`) and B+steps (`bg carbs bolus steps`) — see [Next steps](#next-steps).

## Setup

| Item | Setting |
|---|---|
| Samples | `sample_prepare/samples/h12_f6_trs1_tes1` — 57 T1D subjects from 5 datasets (OhioT1DM 6, Bris 15, HUPA 21, T1DM-UOM 7, UCHTT1DM 8) |
| Split | Within-subject chronological split (no held-out subjects). Train 724,717 / valid 181,148 (last 20% of each subject's train samples) / test 224,687 |
| Window | History 12 steps (60 min) → horizon 6 steps (30 min), stride 1, no sample crosses a CGM gap > 6 min |
| Model | Point-wise-token Transformer encoder (pre-LN, 2 layers, 4 heads, FFN 4·d, dropout 0.1) + MLP head on the last token, direct 6-step output |
| Parameters | d=64: 103,270 (A) / 103,398 (B) · d=128: 407,238 (A) / 407,494 (B) |
| Baseline A | `bg` |
| Baseline B | `bg carbs bolus` — grid-aligned, zero when no event, fused by a shared `Linear(C, d)` (early fusion) |
| `none` normalisation | Raw inputs and targets (mg/dL, g, U) |
| `scaled` normalisation | bg and y: global z-score from train (mean 151.55, std 57.16) · carbs: `log1p(x)/log1p(80)` · bolus: `log1p(x)/log1p(15)` |
| Training | AdamW (lr 1e-3, wd 1e-5), batch 512, MSE over 6 steps, ReduceLROnPlateau (patience 3, ×0.5), early stopping on valid loss (patience 10), max 100 epochs, seeds 0/1/2; Colab CUDA ≈ 9 s/epoch |
| Metric | RMSE (mg/dL) of the 30-min endpoint, pooled over all test forecasts, unless noted. *Subject-mean* = mean of per-subject RMSE. Windows use the event label at the target timestamp |

**LSTM reference.** Martinsson et al. LSTM from `multivariate_lstm_baseline`, recomputed on the same 57 subjects and the same 224,687 test samples. Single seed (60). Its *raw* variant uses `bg carbs bolus steps`; its *IOB/COB* variant uses `bg iob cob smoothed_step`.

## Results

### Overall (mean ± std over 3 seeds)

| Model | Norm | d | RMSE | MAE | Subject-mean RMSE |
|---|---|---|---|---|---|
| A | none | 64 | 24.08 ± 0.12 | 16.94 ± 0.02 | 22.35 ± 0.09 |
| B | none | 64 | 24.11 ± 0.55 | 17.09 ± 0.30 | 22.36 ± 0.42 |
| A | scaled | 64 | 21.68 ± 0.04 | 15.11 ± 0.01 | 20.10 ± 0.06 |
| A | scaled | 128 | 21.59 ± 0.01 | 15.04 ± 0.03 | 19.98 ± 0.04 |
| **B** | **scaled** | **64** | **21.37 ± 0.00** | **14.91 ± 0.04** | **19.89 ± 0.02** |
| B | scaled | 128 | 21.34 ± 0.02 | 14.87 ± 0.02 | 19.87 ± 0.05 |
| LSTM univariate | ×0.01 | – | 21.79 | 15.32 | 20.27 |
| LSTM raw | log1p | – | 21.39 | 14.95 | 20.01 |
| LSTM IOB/COB | clip | – | 21.42 | 15.02 | 20.09 |
| Persistence | – | – | 26.34 | – | – |

### By dataset (RMSE, mean over seeds)

| Model | Bris | HUPA | OhioT1DM | T1DM-UOM | UCHTT1DM |
|---|---|---|---|---|---|
| A none d64 | 26.72 | 17.74 | 20.91 | 24.61 | 24.21 |
| B none d64 | 26.71 | 17.85 | 20.84 | 24.77 | 24.24 |
| A scaled d64 | 24.19 | 15.40 | 19.27 | 21.97 | 23.00 |
| A scaled d128 | 24.11 | 15.29 | 19.13 | 21.87 | 23.18 |
| **B scaled d64** | **23.81** | **15.25** | **18.93** | **21.76** | **22.80** |
| B scaled d128 | 23.79 | 15.20 | 18.88 | 21.80 | 22.98 |
| LSTM univariate | 24.23 | 15.62 | 19.59 | 22.22 | 23.33 |
| LSTM raw | 23.76 | 15.48 | 19.21 | 21.70 | 22.90 |
| LSTM IOB/COB | 23.75 | 15.56 | 19.28 | 21.97 | 22.63 |

### By event window (RMSE, mean over seeds)

Share of test samples: BASELINE+NOCTURNAL 61.9% (139,015) · Mix 19.3% (43,305) · BOLUS 12.2% (27,447) · MEAL 4.3% (9,682) · EXERCISE 2.3% (5,238).

| Model | BASELINE+NOCTURNAL | BOLUS | EXERCISE | MEAL | Mix |
|---|---|---|---|---|---|
| A none d64 | 19.77 | 27.45 | 24.72 | 33.38 | 30.87 |
| B none d64 | 19.87 | 27.81 | 25.03 | 32.96 ± 0.74 | 30.64 |
| A scaled d64 | 17.96 | 24.34 | 21.92 | 29.10 | 27.92 |
| A scaled d128 | 17.90 | 24.24 | 21.87 | 28.91 | 27.80 |
| **B scaled d64** | **17.87** | **24.02** | **22.09** | **27.93** | **27.29** |
| B scaled d128 | 17.86 | 24.06 | 22.08 | 27.79 | 27.23 |
| LSTM univariate | 18.18 | 24.39 | 21.83 | 29.18 | 27.89 |
| LSTM raw | 17.85 | 23.97 | 21.75 | 28.35 | 27.39 |
| LSTM IOB/COB | 17.89 | 24.03 | 22.04 | 28.37 | 27.36 |

### Effect of adding covariates (B − A, negative = B better)

| Window | Transformer d64 | Transformer d128 | LSTM (raw − univariate) |
|---|---|---|---|
| BASELINE+NOCTURNAL | −0.09 | −0.04 | −0.33 |
| BOLUS | −0.32 | −0.18 | −0.42 |
| EXERCISE | **+0.17** | **+0.21** | −0.08 |
| MEAL | **−1.17** | **−1.12** | −0.83 |
| Mix | −0.63 | −0.57 | −0.50 |
| Overall | −0.31 | −0.25 | −0.40 |

## Findings

1. **Normalisation is a prerequisite.** Without it, both models are ≈ 2.4–2.7 mg/dL worse (A 24.08 vs 21.68; B 24.11 vs 21.37). The `none` runs reach their best validation loss at epoch 1–3, then the validation loss oscillates (≈ 290–500 mg/dL²) and training stops early. They also fit the training set worse: training MSE ≈ 415–430 mg/dL², against ≈ 220–230 mg/dL² for `scaled`. So the problem is optimisation, not only overfitting. This matches the LayerNorm analysis: with raw bg ≈ 150, `Linear(1, d)` + LayerNorm maps 100 and 300 mg/dL to nearly identical tokens.
2. **Without normalisation the covariates are not used.** B `none` ≈ A `none` (24.11 vs 24.08), and B `none` has a 5× larger seed spread (± 0.55). Later grids can drop the `none` setting.
3. **With normalisation, covariates give a small but stable overall gain:** −0.31 mg/dL (d64), with seed std ≤ 0.04. B beats A on all five datasets (Bris −0.38, OhioT1DM −0.34, T1DM-UOM −0.21, UCHTT1DM −0.20, HUPA −0.15).
4. **The gain is concentrated in event windows.** MEAL −1.17 and Mix −0.63 against −0.09 in BASELINE+NOCTURNAL. Overall RMSE hides this, because 62% of test samples are event-free. **Event-window metrics should be the primary evaluation.**
5. **Large headroom remains.** For A, MEAL windows are 11.1 mg/dL worse than BASELINE+NOCTURNAL (29.10 vs 17.96). Grid-aligned covariates close only ≈ 10% of that gap (−1.17). This is the target for the event-token model.
6. **EXERCISE windows get slightly worse with covariates** (+0.17 at d64, +0.21 at d128). A plausible cause is that B has no activity input: during exercise, carbs = bolus = 0, which may read as "stable". EXERCISE is only 2.3% of test samples, so significance is untested. B+steps addresses this.
7. **Model capacity is not the bottleneck.** d=128 (≈ 4× parameters) changes RMSE by only −0.09 (A) and −0.03 (B). d=128 reaches a lower training loss without better validation loss. d=64 is kept as the default.
8. **The backbone is not a confounder.** On identical test samples, Transformer A beats the LSTM univariate by 0.11 (21.68 vs 21.79), and Transformer B matches the LSTM raw (21.37 vs 21.39). The LSTM uses 5× more parameters (≈ 528k) and, for raw, also steps. Per window, Transformer B is 0.42 better than LSTM raw in MEAL and 0.34 worse in EXERCISE.

## Caveats

- Seed std only measures training variance. No per-subject significance test has been run yet (paired Wilcoxon / bootstrap), so small differences (≈ 0.1–0.3 overall, EXERCISE +0.17) are not yet established.
- The LSTM reference is a single seed with a different feature set (raw includes steps), so Transformer–LSTM differences below ≈ 0.1 are within noise.
- There are no held-out subjects. Results describe future forecasting for subjects seen in training.
- Window labels are taken at the target timestamp. A sample can contain an event in its history and still be labelled BASELINE+NOCTURNAL at the target.

## Next steps

1. **Baselines C and B+steps** (scaled, d=64, seeds 0–2), run with `command/run_c_bsteps.sh`. Results go to `runs_c_bsteps/`, separate from `runs/`.
   - C `bg iob cob`: physiological alignment, the same modalities as B with a different encoding. Note that `iob` is computed from total insulin (basal + bolus), so C also carries basal information that B does not.
   - B+steps `bg carbs bolus steps`: tests whether activity input fixes the EXERCISE regression. It uses the same features as the LSTM raw variant.
2. **Per-subject paired tests**: A vs B, B vs C, and Transformer vs LSTM, overall and per window.
3. **Reference for the event-token model (GLEAM):** B scaled d64 — overall 21.37, MEAL 27.93, Mix 27.29, BOLUS 24.02, EXERCISE 22.09.
