# Transformer Baselines — Result Summary

Last updated: 2026-09-28 · 24 runs, all finished with early stopping:
- A / B: 3 seeds × 6 configurations, results in `runs/`
- C / B+steps: 3 seeds × 2 configurations, results in `runs_c_bsteps/`

## Setup

| Item | Setting |
|---|---|
| Samples | `sample_prepare/samples/h12_f6_trs1_tes1` — 57 T1D subjects from 5 datasets (OhioT1DM 6, Bris 15, HUPA 21, T1DM-UOM 7, UCHTT1DM 8) |
| Split | Within-subject chronological split (no held-out subjects). Train 724,717 / valid 181,148 (last 20% of each subject's train samples) / test 224,687 |
| Window | History 12 steps (60 min) → horizon 6 steps (30 min), stride 1, no sample crosses a CGM gap > 6 min |
| Model | Point-wise-token Transformer encoder (pre-LN, 2 layers, 4 heads, FFN 4·d, dropout 0.1) + MLP head on the last token, direct 6-step output. Inputs are fused by a shared `Linear(C, d)` (early fusion) |
| Parameters | d=64: 103,270 (A) · 103,398 (B, C) · 103,462 (B+steps). d=128: 407,238 (A) / 407,494 (B) |
| `none` normalisation | Raw inputs and targets (mg/dL, g, U) |
| `scaled` normalisation | bg and y: global z-score from train (mean 151.55, std 57.16) · carbs `log1p(x)/log1p(80)` · bolus `log1p(x)/log1p(15)` · steps `log1p(x)/log1p(250)` · iob `clip(x,0,10)/10` · cob `clip(x,0,60)/60` |
| Training | AdamW (lr 1e-3, wd 1e-5), batch 512, MSE over 6 steps, ReduceLROnPlateau (patience 3, ×0.5), early stopping on valid loss (patience 10), max 100 epochs, seeds 0/1/2; Colab CUDA ≈ 9 s/epoch |
| Metric | RMSE (mg/dL) of the 30-min endpoint, pooled over all test forecasts, unless noted. *Subject-mean* = mean of per-subject RMSE. Windows use the event label at the target timestamp |

**Input feature sets**

| Name | Features | Encoding of events |
|---|---|---|
| A | `bg` | – |
| B | `bg carbs bolus` | Raw alignment: event value at its 5-min slot, zero elsewhere |
| C | `bg iob cob` | Physiological alignment: IOB (triangular kernel, from basal + bolus) and COB (Hovorka two-compartment) |
| B+steps | `bg carbs bolus steps` | B plus raw 5-min step counts (same features as the LSTM raw variant) |

**LSTM reference.** Martinsson et al. LSTM from `multivariate_lstm_baseline`, recomputed on the same 57 subjects and the same 224,687 test samples. It is **not** trained under the same protocol:
- single seed (60);
- train stride 12, i.e. 60,655 fit samples vs 724,717 here;
- Gaussian NLL loss;
- bg scaled by ×0.01;
- ≈ 528k parameters.

Its *raw* variant uses `bg carbs bolus steps` and its *IOB/COB* variant uses `bg iob cob smoothed_step`.

## Results

### Overall (mean ± std over 3 seeds)

| Model | Norm | d | RMSE | MAE | Subject-mean RMSE |
|---|---|---|---|---|---|
| A | none | 64 | 24.08 ± 0.12 | 16.94 ± 0.02 | 22.35 ± 0.09 |
| B | none | 64 | 24.11 ± 0.55 | 17.09 ± 0.30 | 22.36 ± 0.42 |
| A | scaled | 64 | 21.68 ± 0.04 | 15.11 ± 0.01 | 20.10 ± 0.06 |
| A | scaled | 128 | 21.59 ± 0.01 | 15.04 ± 0.03 | 19.98 ± 0.04 |
| B | scaled | 64 | 21.37 ± 0.00 | 14.91 ± 0.04 | 19.89 ± 0.02 |
| B | scaled | 128 | 21.34 ± 0.02 | 14.87 ± 0.02 | 19.87 ± 0.05 |
| C | scaled | 64 | 21.31 ± 0.05 | 14.85 ± 0.01 | 19.90 ± 0.01 |
| **B+steps** | **scaled** | **64** | **21.17 ± 0.04** | **14.77 ± 0.05** | **19.80 ± 0.03** |
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
| B scaled d64 | 23.81 | 15.25 | 18.93 | 21.76 | 22.80 |
| B scaled d128 | 23.79 | 15.20 | 18.88 | 21.80 | 22.98 |
| C scaled d64 | 23.69 | 15.28 | 18.98 | 21.93 | 22.74 |
| **B+steps scaled d64** | **23.58** | **15.18** | **18.83** | **21.47** | 22.80 |
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
| B scaled d64 | 17.87 | 24.02 | 22.09 | 27.93 | 27.29 |
| B scaled d128 | 17.86 | 24.06 | 22.08 | 27.79 | 27.23 |
| C scaled d64 | 17.83 | 24.13 | 21.90 | **27.68** | 27.16 |
| **B+steps scaled d64** | **17.70** | **23.84** | **21.52** | 27.77 | **27.06** |
| LSTM univariate | 18.18 | 24.39 | 21.83 | 29.18 | 27.89 |
| LSTM raw | 17.85 | 23.97 | 21.75 | 28.35 | 27.39 |
| LSTM IOB/COB | 17.89 | 24.03 | 22.04 | 28.37 | 27.36 |

### Covariate effect relative to A (scaled, d=64; negative = better)

| Window | B − A | C − A | B+steps − A | B+steps − B | C − B | LSTM raw − LSTM uni |
|---|---|---|---|---|---|---|
| BASELINE+NOCTURNAL | −0.09 | −0.13 | −0.26 | −0.17 | −0.04 | −0.33 |
| BOLUS | −0.32 | −0.21 | −0.50 | −0.18 | +0.11 | −0.42 |
| EXERCISE | +0.17 | −0.02 | −0.40 | −0.57 | −0.19 | −0.08 |
| MEAL | −1.17 | −1.42 | −1.33 | −0.16 | −0.25 | −0.83 |
| Mix | −0.63 | −0.76 | −0.86 | −0.23 | −0.13 | −0.50 |
| Overall | −0.31 | −0.37 | −0.51 | −0.20 | −0.06 | −0.40 |

### Where are the events in event-labelled test samples?

A window label says an event happened in the 120 min before the **target** timestamp (bolus: 10–120 min). The table shows where that event lies relative to the 60-min input window, which ends 30 min before the target.

| Event location | MEAL-labelled samples (51,277) | BOLUS-labelled samples (70,043) |
|---|---|---|
| Inside the input window (visible to B / B+steps) | 52.2% | 60.5% |
| Only before the input window (visible only through IOB/COB in C) | 26.6% | 26.4% |
| Only after the forecast origin (visible to no model) | 21.2% | 13.0% |

"MEAL-labelled" means any label containing MEAL (MEAL, MEAL+BOLUS, MEAL+EXERCISE, MEAL+BOLUS+EXERCISE); BOLUS is defined the same way.

## Findings

1. **Normalisation is a prerequisite.** Without it, both models are ≈ 2.4–2.7 mg/dL worse (A 24.08 vs 21.68; B 24.11 vs 21.37). The `none` runs reach their best validation loss at epoch 1–3, then the validation loss oscillates (≈ 290–500 mg/dL²). They also fit the training set worse: training MSE ≈ 415–430 mg/dL² vs ≈ 220–230 mg/dL² for `scaled`. So this is an optimisation problem, not only overfitting. It matches the LayerNorm analysis: with raw bg ≈ 150, `Linear(1, d)` + LayerNorm maps 100 and 300 mg/dL to nearly identical tokens.
2. **Without normalisation the covariates are not used.** B `none` ≈ A `none` (24.11 vs 24.08), with a 5× larger seed spread (± 0.55). Later grids drop the `none` setting.
3. **Grid-aligned covariates help mainly in event windows.** For B: MEAL −1.17 and Mix −0.63, against −0.09 in BASELINE+NOCTURNAL. Overall RMSE hides this because 62% of test samples are event-free, so **event-window metrics should be the primary evaluation.** B beats A on all five datasets.
4. **Activity (steps) is the most useful single addition.** B+steps is the best configuration: 21.17 overall, −0.51 vs A and −0.20 vs B.
   - It removes B's EXERCISE regression: 22.09 → 21.52, now 0.40 better than A.
   - It also helps BASELINE+NOCTURNAL (−0.17) and BOLUS (−0.18). The EXERCISE label only marks sustained bouts, so lighter everyday activity carries information in all windows.
   - It improves every dataset except UCHTT1DM (8 subjects, 3–8 days each).
5. **Physiological encoding ≈ raw encoding.** C vs B is −0.06 overall, about one seed std.
   - By window it is mixed: better in MEAL (−0.25), Mix and EXERCISE; worse in BOLUS (+0.11).
   - By dataset it is also mixed: better in Bris and UCHTT1DM; worse in T1DM-UOM (+0.17), OhioT1DM and HUPA.
   - The LSTM shows the same pattern (raw 21.39 vs IOB/COB 21.42).
   - How carbs and insulin are encoded on the 5-min grid is not the main lever.
6. **Event visibility limits every grid-aligned model.** In 47.8% of MEAL-labelled test samples the meal is not in the 60-min input window:
   - 26.6% lie before the window. Only C sees these, through the COB decay, which is the likely reason C is best in MEAL (27.68).
   - 21.2% occur after the forecast origin, so no current model can see them.
   - This points to two design levers for the event-token model:
     1. an event lookback longer than the CGM window (cheap with sparse tokens);
     2. an explicit decision on whether events announced at or after the forecast origin may be used (e.g. meal announcements in AID).
7. **Large headroom remains in meal windows.** For A, MEAL is 11.1 mg/dL worse than BASELINE+NOCTURNAL (29.10 vs 17.96). The best baseline closes ≈ 13% of that gap (C: −1.42).
8. **Model capacity is not the bottleneck.** d=128 (≈ 4× parameters) changes RMSE by only −0.09 (A) and −0.03 (B), reaching a lower training loss without better validation loss. d=64 is kept.
9. **The Transformer vs LSTM comparison is not yet controlled.**
   - With identical features, Transformer B+steps beats LSTM raw by 0.22 (21.17 vs 21.39) and is better in every window. Transformer A beats LSTM univariate by 0.11.
   - But the LSTM was trained on 12× fewer samples (stride 12), with a different loss and bg scaling, and a single seed.
   - These gaps therefore cannot be attributed to the backbone. An LSTM retrained in this pipeline is needed before claiming the backbone is not a confounder.

## Caveats

- Seed std only measures training variance. No per-subject significance test has been run yet (paired Wilcoxon / bootstrap). Differences ≲ 0.1–0.2 overall — C vs B in particular — are not established.
- There are no held-out subjects. Results describe future forecasting for subjects seen in training.
- Window labels are taken at the target timestamp, so part of every event window is not observable from the input (see the event-location table).
- `iob` is computed from total insulin (basal + bolus), so C carries basal information that B does not.

## Next steps

1. **Complete the 2×2 grid-aligned design** (encoding × activity): add `bg iob cob smoothed_step` (scaled, d=64, 3 seeds).
2. **LSTM in the same pipeline**: same samples (stride 1), MSE loss, same normalisation, 3 seeds. This makes the backbone comparison fair.
3. **Per-subject paired tests**: A vs B, B vs C, B vs B+steps, Transformer vs LSTM; overall and per window.
4. **Visibility-split evaluation**: report MEAL / BOLUS windows separately for "event in input window", "event before window" and "event after origin".
5. **Reference for the event-token model (GLEAM)**: B+steps scaled d64 when GLEAM uses meal, bolus and activity tokens — overall 21.17, MEAL 27.77, Mix 27.06, BOLUS 23.84, EXERCISE 21.52. Report C alongside it as the physiological-alignment reference.
