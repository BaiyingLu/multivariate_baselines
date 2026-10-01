# Transformer Baselines — Result Summary

Last updated: 2026-10-01 · 78 runs, all finished with early stopping:
- **Part 1 — 60-min history baselines** (24 runs): A / B in `runs/`, C / B+steps in `runs_c_bsteps/`
- **Part 2 — History length × prediction horizon** (54 runs): A / B+steps / C+steps in `runs_history/`

Contents: Part 1 (setup, results, findings) · Part 2 (setup, overall results, per-window results, findings) · Caveats · Next steps

---

## Part 1 — 60-min history baselines

### Setup

| Item | Setting |
|---|---|
| Samples | `sample_prepare/samples/h12_f6_trs1_tes1` — 57 T1D subjects from 5 datasets (OhioT1DM 6, Bris 15, HUPA 21, T1DM-UOM 7, UCHTT1DM 8) |
| Split | Within-subject chronological split (no held-out subjects). Train 724,717 / valid 181,148 (last 20% of each subject's train samples) / test 224,687 |
| Window | History 12 steps (60 min) → horizon 6 steps (30 min), stride 1, no sample crosses a CGM gap > 6 min |
| Model | Point-wise-token Transformer encoder (pre-LN, 2 layers, 4 heads, FFN 4·d, dropout 0.1) + MLP head on the last token, direct multi-step output. Inputs are fused by a shared `Linear(C, d)` (early fusion) |
| Parameters | d=64: 103,270 (A) · 103,398 (B, C) · 103,462 (B+steps). d=128: 407,238 (A) / 407,494 (B) |
| `none` normalisation | Raw inputs and targets (mg/dL, g, U) |
| `scaled` normalisation | bg and y: global z-score from train (mean 151.55, std 57.16) · carbs `log1p(x)/log1p(80)` · bolus `log1p(x)/log1p(15)` · steps `log1p(x)/log1p(250)` · smoothed_step `log1p(x)/log1p(150)` · iob `clip(x,0,10)/10` · cob `clip(x,0,60)/60` |
| Training | AdamW (lr 1e-3, wd 1e-5), batch 512, MSE over all horizon steps, ReduceLROnPlateau (patience 3, ×0.5), early stopping on valid loss (patience 10), max 100 epochs, seeds 0/1/2; Colab CUDA ≈ 9 s/epoch |
| Metric | RMSE (mg/dL) of the endpoint (last horizon step), pooled over all test forecasts, unless noted. *Subject-mean* = mean of per-subject RMSE. Windows use the event label at the target timestamp |

**Input feature sets**

| Name | Features | Encoding of events |
|---|---|---|
| A | `bg` | – |
| B | `bg carbs bolus` | Raw alignment: event value at its 5-min slot, zero elsewhere |
| C | `bg iob cob` | Physiological alignment: IOB (triangular kernel, from basal + bolus) and COB (Hovorka two-compartment) |
| B+steps | `bg carbs bolus steps` | B plus raw 5-min step counts (same features as the LSTM raw variant) |
| C+steps (Part 2) | `bg iob cob smoothed_step` | C plus 30-min rolling mean of steps (same features as the LSTM IOB/COB variant) |

**LSTM reference.** Martinsson et al. LSTM from `multivariate_lstm_baseline`, recomputed on the same 57 subjects and the same 224,687 test samples. It is **not** trained under the same protocol:
- single seed (60);
- train stride 12, i.e. 60,655 fit samples vs 724,717 here;
- Gaussian NLL loss;
- bg scaled by ×0.01;
- ≈ 528k parameters.

Its *raw* variant uses `bg carbs bolus steps` and its *IOB/COB* variant uses `bg iob cob smoothed_step`.

### Results

#### Overall (mean ± std over 3 seeds)

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

#### By dataset (RMSE, mean over seeds)

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

#### By event window (RMSE, mean over seeds)

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

#### Covariate effect relative to A (scaled, d=64; negative = better)

| Window | B − A | C − A | B+steps − A | B+steps − B | C − B | LSTM raw − LSTM uni |
|---|---|---|---|---|---|---|
| BASELINE+NOCTURNAL | −0.09 | −0.13 | −0.26 | −0.17 | −0.04 | −0.33 |
| BOLUS | −0.32 | −0.21 | −0.50 | −0.18 | +0.11 | −0.42 |
| EXERCISE | +0.17 | −0.02 | −0.40 | −0.57 | −0.19 | −0.08 |
| MEAL | −1.17 | −1.42 | −1.33 | −0.16 | −0.25 | −0.83 |
| Mix | −0.63 | −0.76 | −0.86 | −0.23 | −0.13 | −0.50 |
| Overall | −0.31 | −0.37 | −0.51 | −0.20 | −0.06 | −0.40 |

#### Where are the events in event-labelled test samples?

A window label says an event happened in the 120 min before the **target** timestamp (bolus: 10–120 min). The table shows where that event lies relative to the 60-min input window, which ends 30 min before the target.

| Event location | MEAL-labelled samples (51,277) | BOLUS-labelled samples (70,043) |
|---|---|---|
| Inside the input window (visible to B / B+steps) | 52.2% | 60.5% |
| Only before the input window (visible only through IOB/COB in C) | 26.6% | 26.4% |
| Only after the forecast origin (visible to no model) | 21.2% | 13.0% |

"MEAL-labelled" means any label containing MEAL (MEAL, MEAL+BOLUS, MEAL+EXERCISE, MEAL+BOLUS+EXERCISE); BOLUS is defined the same way.

### Findings (Part 1)

1. **Normalisation is a prerequisite.** Without it, both models are ≈ 2.4–2.7 mg/dL worse (A 24.08 vs 21.68; B 24.11 vs 21.37). The `none` runs reach their best validation loss at epoch 1–3, then the validation loss oscillates (≈ 290–500 mg/dL²). They also fit the training set worse: training MSE ≈ 415–430 mg/dL² vs ≈ 220–230 mg/dL² for `scaled`. So this is an optimisation problem, not only overfitting. It matches the LayerNorm analysis: with raw bg ≈ 150, `Linear(1, d)` + LayerNorm maps 100 and 300 mg/dL to nearly identical tokens.
2. **Without normalisation the covariates are not used.** B `none` ≈ A `none` (24.11 vs 24.08), with a 5× larger seed spread (± 0.55). Later grids drop the `none` setting.
3. **Grid-aligned covariates help mainly in event windows.** For B: MEAL −1.17 and Mix −0.63, against −0.09 in BASELINE+NOCTURNAL. Overall RMSE hides this because 62% of test samples are event-free, so **event-window metrics should be the primary evaluation.** B beats A on all five datasets.
4. **Activity (steps) is the most useful single addition.** B+steps is the best configuration: 21.17 overall, −0.51 vs A and −0.20 vs B.
   - It removes B's EXERCISE regression: 22.09 → 21.52, now 0.40 better than A.
   - It also helps BASELINE+NOCTURNAL (−0.17) and BOLUS (−0.18). The EXERCISE label only marks sustained bouts, so lighter everyday activity carries information in all windows.
   - It improves every dataset except UCHTT1DM (8 subjects, 3–8 days each).
5. **Physiological encoding ≈ raw encoding (at 60-min history).** C vs B is −0.06 overall, about one seed std.
   - By window it is mixed: better in MEAL (−0.25), Mix and EXERCISE; worse in BOLUS (+0.11).
   - By dataset it is also mixed: better in Bris and UCHTT1DM; worse in T1DM-UOM (+0.17), OhioT1DM and HUPA.
   - The LSTM shows the same pattern (raw 21.39 vs IOB/COB 21.42).
   - How carbs and insulin are encoded on the 5-min grid is not the main lever. Part 2 shows that this changes with a longer history.
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

---

## Part 2 — History length × prediction horizon

### Setup

| Item | Setting |
|---|---|
| Question | Does a longer input history help, and does it change the ranking of raw vs physiological event encoding? |
| Sample sets | `sample_prepare/new_sample/h{H}_f{F}_trs1_tes1_mh48_mf12` for history H ∈ {12, 24, 48} steps (1 / 2 / 4 h) and horizon F ∈ {6, 12} steps (30 / 60 min) |
| Alignment | Built with `--min_history 48 --min_horizon 12`. A forecast origin is kept only if 4 h of history and 60 min of future exist, so **all six sets share the same forecast origins** (train 703,980 / valid 175,968 / test 217,135). Within a horizon the targets are identical too, so only the input length differs |
| Features | A `bg` · B+steps `bg carbs bolus steps` · C+steps `bg iob cob smoothed_step` |
| Model / training | As in Part 1: `scaled`, d=64, 3 seeds, MSE over all F steps (one model per horizon). Parameters 103,270–105,964 (the positional embedding grows with H) |
| Persistence | 26.26 (30 min) · 42.17 (60 min) |

The aligned test set is 3.4% smaller than Part 1's (217,135 vs 224,687), so absolute numbers are not directly comparable with Part 1. Relative differences agree: B+steps − A at 1-h history / 30 min is −0.51 in both parts.

### Overall results (mean ± std over 3 seeds)

**30-min horizon**

| Features | History | RMSE | MAE | Subject-mean RMSE |
|---|---|---|---|---|
| A | 1 h | 21.60 ± 0.05 | 15.07 ± 0.11 | 19.97 ± 0.10 |
| A | 2 h | 21.54 ± 0.02 | 15.01 ± 0.06 | 19.89 ± 0.04 |
| A | 4 h | 21.51 ± 0.03 | 15.02 ± 0.04 | 19.88 ± 0.01 |
| B+steps | 1 h | 21.09 ± 0.05 | 14.72 ± 0.04 | 19.67 ± 0.06 |
| B+steps | 2 h | 20.98 ± 0.02 | 14.65 ± 0.03 | 19.58 ± 0.05 |
| **B+steps** | **4 h** | **20.95 ± 0.02** | **14.60 ± 0.01** | **19.55 ± 0.03** |
| C+steps | 1 h | 21.05 ± 0.01 | 14.67 ± 0.02 | 19.62 ± 0.05 |
| C+steps | 2 h | 21.01 ± 0.02 | 14.65 ± 0.02 | 19.64 ± 0.05 |
| C+steps | 4 h | 20.99 ± 0.04 | 14.65 ± 0.06 | 19.71 ± 0.12 |

**60-min horizon**

| Features | History | RMSE | MAE | Subject-mean RMSE |
|---|---|---|---|---|
| A | 1 h | 35.85 ± 0.03 | 25.94 ± 0.05 | 33.51 ± 0.01 |
| A | 2 h | 35.71 ± 0.04 | 25.84 ± 0.08 | 33.39 ± 0.16 |
| A | 4 h | 35.65 ± 0.05 | 25.82 ± 0.08 | 33.41 ± 0.11 |
| B+steps | 1 h | 35.20 ± 0.02 | 25.42 ± 0.08 | 33.22 ± 0.07 |
| B+steps | 2 h | 35.02 ± 0.02 | 25.27 ± 0.05 | 33.17 ± 0.08 |
| **B+steps** | **4 h** | **34.93 ± 0.08** | 25.21 ± 0.12 | **33.16 ± 0.07** |
| C+steps | 1 h | 35.05 ± 0.09 | 25.23 ± 0.05 | 33.18 ± 0.10 |
| C+steps | 2 h | 34.97 ± 0.03 | 25.22 ± 0.08 | 33.20 ± 0.11 |
| C+steps | 4 h | 35.03 ± 0.07 | **25.20 ± 0.03** | 33.32 ± 0.19 |

**Effect of a longer history (4 h − 1 h; negative = better)**

| Features | 30 min | 60 min | 60 min, MEAL window |
|---|---|---|---|
| A | −0.09 | −0.20 | −0.47 |
| B+steps | −0.14 | −0.27 | −1.34 |
| C+steps | −0.06 | −0.02 | −0.65 |

**Raw vs physiological encoding (B+steps − C+steps; negative = raw better)**

| History | 30 min | 60 min |
|---|---|---|
| 1 h | +0.04 | +0.15 |
| 2 h | −0.03 | +0.05 |
| 4 h | −0.04 | −0.10 |

**Covariate effect (RMSE − A at the same history)**

| History | B+steps, 30 min | C+steps, 30 min | B+steps, 60 min | C+steps, 60 min |
|---|---|---|---|---|
| 1 h | −0.51 | −0.55 | −0.65 | −0.80 |
| 2 h | −0.56 | −0.53 | −0.69 | −0.74 |
| 4 h | −0.56 | −0.52 | −0.72 | −0.62 |

**30-min accuracy of the 60-min models** (`rmse_30min` of the F=12 model minus RMSE of the F=6 model at the same history; positive = the dedicated 30-min model is better)

| Features | 1 h | 2 h | 4 h |
|---|---|---|---|
| A | +0.04 | +0.02 | +0.07 |
| B+steps | +0.05 | +0.12 | +0.20 |
| C+steps | +0.05 | +0.16 | +0.29 |

### Per-window results

**Test samples per window** (identical across the three histories — confirms the alignment)

| Horizon | STABLE_BASELINE | NOCTURNAL | MEAL | BOLUS | EXERCISE | MEAL+BOLUS | MEAL+EXERCISE | BOLUS+EXERCISE | MEAL+BOLUS+EXERCISE |
|---|---|---|---|---|---|---|---|---|---|
| 30 min | 88,897 (40.9%) | 45,834 (21.1%) | 9,205 (4.2%) | 26,387 (12.2%) | 5,131 (2.4%) | 37,278 (17.2%) | 668 (0.3%) | 1,669 (0.8%) | 2,066 (1.0%) |
| 60 min | 89,020 | 45,619 | 9,184 | 26,423 | 5,144 | 37,342 | 666 | 1,669 | 2,068 |

Merged buckets: BASELINE+NOCTURNAL = STABLE_BASELINE + NOCTURNAL; Mix = all combinations of two or more events.

**RMSE by merged window — 30 min**

| Features | History | BASELINE+NOCTURNAL | BOLUS | EXERCISE | MEAL | Mix |
|---|---|---|---|---|---|---|
| A | 1 h | 17.90 | 24.21 | 21.93 | 29.09 | 27.90 |
| A | 2 h | 17.83 | 24.17 | 21.81 | 28.96 | 27.85 |
| A | 4 h | 17.81 | 24.21 | 21.88 | 28.76 | 27.78 |
| B+steps | 1 h | 17.62 | 23.73 | 21.62 | 27.73 | 27.02 |
| B+steps | 2 h | 17.60 | 23.53 | 21.67 | **27.24** | **26.84** |
| B+steps | 4 h | **17.52** | **23.53** | **21.50** | 27.37 | 26.86 |
| C+steps | 1 h | 17.59 | 23.72 | 21.75 | 27.49 | 26.94 |
| C+steps | 2 h | 17.56 | 23.70 | 21.70 | 27.44 | 26.90 |
| C+steps | 4 h | 17.54 | 23.62 | 22.02 | 27.42 | 26.86 |

**RMSE by merged window — 60 min**

| Features | History | BASELINE+NOCTURNAL | BOLUS | EXERCISE | MEAL | Mix |
|---|---|---|---|---|---|---|
| A | 1 h | 29.45 | 41.92 | 35.50 | 50.47 | 45.38 |
| A | 2 h | 29.36 | 41.78 | 35.60 | 50.24 | 45.14 |
| A | 4 h | 29.37 | 41.63 | **35.49** | 50.00 | 45.01 |
| B+steps | 1 h | 29.15 | 40.81 | 36.47 | 48.59 | 44.36 |
| B+steps | 2 h | 29.01 | 40.55 | 36.34 | 47.84 | 44.26 |
| B+steps | 4 h | 29.08 | **40.33** | 36.50 | **47.25** | **43.98** |
| C+steps | 1 h | 28.93 | 40.81 | 36.33 | 48.31 | 44.25 |
| C+steps | 2 h | **28.91** | 40.58 | 37.35 | 47.87 | 44.14 |
| C+steps | 4 h | 28.98 | 40.94 | 36.69 | 47.66 | 44.12 |

**RMSE by fine window — 30 min** (MEAL, BOLUS and EXERCISE are in the merged table)

| Features | History | STABLE_BASELINE | NOCTURNAL | MEAL+BOLUS | MEAL+EXERCISE | BOLUS+EXERCISE | MEAL+BOLUS+EXERCISE |
|---|---|---|---|---|---|---|---|
| A | 1 h | 19.29 | 14.83 | 27.72 | 26.16 | 30.29 | 29.55 |
| A | 2 h | 19.22 | 14.77 | 27.67 | 26.25 | 30.24 | 29.53 |
| A | 4 h | 19.16 | 14.85 | 27.59 | 26.22 | 30.19 | 29.52 |
| B+steps | 1 h | 19.02 | 14.51 | 26.88 | 25.00 | 29.23 | 28.33 |
| B+steps | 2 h | 18.99 | 14.53 | 26.70 | 24.80 | 29.01 | 28.04 |
| B+steps | 4 h | 18.89 | 14.51 | 26.74 | 25.11 | 28.69 | 27.99 |
| C+steps | 1 h | 18.99 | 14.49 | 26.74 | 25.46 | 29.90 | 28.40 |
| C+steps | 2 h | 18.95 | 14.48 | 26.69 | 25.54 | 30.15 | 28.46 |
| C+steps | 4 h | 18.91 | 14.52 | 26.64 | 25.84 | 30.08 | 28.36 |

**RMSE by fine window — 60 min**

| Features | History | STABLE_BASELINE | NOCTURNAL | MEAL+BOLUS | MEAL+EXERCISE | BOLUS+EXERCISE | MEAL+BOLUS+EXERCISE |
|---|---|---|---|---|---|---|---|
| A | 1 h | 31.65 | 24.58 | 45.30 | 43.97 | 48.63 | 44.64 |
| A | 2 h | 31.54 | 24.56 | 45.08 | 44.24 | 47.73 | 44.52 |
| A | 4 h | 31.47 | 24.76 | 44.94 | 43.82 | 47.15 | 44.89 |
| B+steps | 1 h | 31.37 | 24.23 | 44.33 | 42.70 | 46.95 | 43.19 |
| B+steps | 2 h | 31.19 | 24.18 | 44.26 | 43.14 | 45.92 | 43.07 |
| B+steps | 4 h | 31.18 | 24.48 | 43.94 | 42.68 | 46.07 | 43.40 |
| C+steps | 1 h | 31.10 | 24.17 | 44.12 | 43.74 | 47.86 | 43.84 |
| C+steps | 2 h | 31.03 | 24.24 | 43.90 | 43.69 | 49.12 | 44.41 |
| C+steps | 4 h | 31.10 | 24.29 | 43.94 | 43.77 | 47.83 | 44.38 |

**Covariate effect per window** (RMSE − A at the same history; range over the three histories; negative = better)

| Window | B+steps, 30 min | C+steps, 30 min | B+steps, 60 min | C+steps, 60 min |
|---|---|---|---|---|
| STABLE_BASELINE | −0.23 to −0.27 | −0.25 to −0.30 | −0.28 to −0.35 | −0.37 to −0.55 |
| NOCTURNAL | −0.24 to −0.34 | −0.29 to −0.34 | −0.28 to −0.38 | −0.32 to −0.47 |
| MEAL | **−1.36 to −1.72** | −1.34 to −1.60 | **−1.88 to −2.75** | −2.16 to −2.37 |
| BOLUS | −0.48 to −0.68 | −0.47 to −0.59 | −1.11 to −1.30 | −0.69 to −1.20 |
| EXERCISE | −0.14 to −0.38 | −0.18 to +0.14 | **+0.74 to +1.01** | **+0.83 to +1.75** |
| MEAL+BOLUS | −0.84 to −0.97 | −0.95 to −0.98 | −0.82 to −1.00 | −1.00 to −1.18 |
| MEAL+EXERCISE | −1.11 to −1.45 | −0.38 to −0.71 | −1.10 to −1.27 | −0.05 to −0.55 |
| BOLUS+EXERCISE | −1.06 to −1.50 | −0.09 to −0.39 | −1.08 to −1.81 | −0.77 to +1.39 |
| MEAL+BOLUS+EXERCISE | −1.22 to −1.53 | −1.07 to −1.16 | −1.45 to −1.49 | −0.11 to −0.80 |

**Share of the overall MSE reduction by window** (B+steps vs A at 4-h history; a window's share = n · (RMSE_A² − RMSE_B²) / total)

| Window | 30 min | 60 min |
|---|---|---|
| MEAL+BOLUS | 34% | 30% |
| STABLE_BASELINE + NOCTURNAL | 27% | 21% |
| BOLUS | 17% | 26% |
| MEAL | 14% | 22% |
| Other combinations with exercise | 8% | 5% |
| EXERCISE | 2% | −3% |

**Effect of a longer history per window** (4 h − 1 h; negative = better)

| Horizon | Features | STABLE_BASELINE | NOCTURNAL | MEAL | BOLUS | EXERCISE | MEAL+BOLUS |
|---|---|---|---|---|---|---|---|
| 30 min | A | −0.13 | +0.02 | −0.33 | 0.00 | −0.05 | −0.13 |
| 30 min | B+steps | −0.13 | 0.00 | −0.36 (2 h: −0.49) | −0.20 | −0.12 | −0.14 |
| 30 min | C+steps | −0.08 | +0.03 | −0.07 | −0.10 | +0.27 | −0.10 |
| 60 min | A | −0.18 | +0.18 | −0.47 | −0.29 | −0.01 | −0.36 |
| 60 min | B+steps | −0.19 | +0.25 | **−1.34** | **−0.48** | +0.03 | −0.39 |
| 60 min | C+steps | 0.00 | +0.12 | −0.65 | +0.13 | +0.36 | −0.18 |

**Raw vs physiological encoding per window** (B+steps − C+steps; negative = raw better)

| Horizon | History | STABLE_BASELINE | NOCTURNAL | MEAL | BOLUS | EXERCISE | MEAL+BOLUS | BOLUS+EXERCISE |
|---|---|---|---|---|---|---|---|---|
| 30 min | 1 h | +0.03 | +0.02 | +0.24 | +0.01 | −0.13 | +0.14 | −0.67 |
| 30 min | 4 h | −0.02 | −0.01 | −0.05 | −0.09 | −0.52 | +0.10 | −1.39 |
| 60 min | 1 h | +0.27 | +0.06 | +0.28 | 0.00 | +0.14 | +0.21 | −0.91 |
| 60 min | 4 h | +0.08 | +0.19 | **−0.41** | **−0.61** | −0.19 | 0.00 | −1.76 |

### Findings (Part 2)

**Overall**

1. **A longer history helps, but only a little, and the gain saturates quickly.**
   - At 30 min the best improvement is −0.14 (B+steps), mostly from 1 h → 2 h.
   - At 60 min the gains are larger (A −0.20, B+steps −0.27): older information matters more for longer horizons.
   - Even CGM-only A improves slightly, so older CGM history has some value on its own.
2. **With a longer history, raw event encoding overtakes physiological encoding.**
   - At 1-h history C+steps is better (60 min: by 0.15). At 4-h history B+steps is better (60 min: by 0.10).
   - C+steps gains almost nothing from a longer history (60 min: −0.02), because IOB/COB already fold in 4–5 h of past events.
   - **The value of IOB/COB is therefore mainly that they carry older events, not their physiological prior.** Given enough raw event history, the model learns the absorption dynamics itself, and slightly better than the fixed kernels.
   - This supports the event-token design: keep raw events, look back further for events than for CGM, and let the model learn the temporal response.
3. **The covariate gain is stable but small in relative terms:** ≈ −0.5 at 30 min and ≈ −0.7 at 60 min, about 2% of RMSE in both cases.
4. **Multi-step training trades near-term for far-term accuracy.**
   - At the 30-min step, a 60-min model is worse than a dedicated 30-min model, and the gap grows with history (B+steps +0.05 → +0.20, C+steps +0.05 → +0.29).
   - The MSE over 12 steps is dominated by the larger far-horizon errors.
   - Train one model per horizon, or weight the loss per step.
5. **Subject-mean RMSE shows smaller history gains.** For C+steps, 4-h history is *worse* than 1 h on subject-mean RMSE (30 min: 19.71 vs 19.62; 60 min: 33.32 vs 33.18), while the pooled RMSE barely moves. The pooled gains may be concentrated in a few large subjects (e.g. HUPA0027P), and C+steps may slightly overfit with long inputs. Differences of 0.1–0.3 need per-subject paired tests.

**Per window**

6. **Per sample, the gain is largest in MEAL windows; in total, MEAL+BOLUS contributes most.**
   - MEAL (meal without bolus) gains ≈ 1.4–1.7 per sample at 30 min and up to 2.75 at 60 min, but it is only 4% of samples.
   - MEAL+BOLUS gains ≈ 1 per sample, but at 17% of samples it contributes 30–34% of the overall improvement.
   - Event-free windows gain only ≈ 0.3 per sample, yet as 62% of samples they still contribute 21–27%.
7. **A longer history helps mainly in MEAL and BOLUS windows, and mainly for B+steps.**
   - At 60 min, 1 h → 4 h improves B+steps by −1.34 in MEAL and −0.48 in BOLUS, the largest history effects in the study. Meal and insulin effects last hours, so raw events must stay inside the input window to be useful.
   - At 30 min, B+steps is best in MEAL at 2-h history (27.24, −0.49 vs 1 h) with no further gain at 4 h. A 2-h input already covers the whole MEAL label window (120 min before the target), which makes the 26.6% of meals that fell before the 1-h window visible (Part 1, finding 6).
   - NOCTURNAL gets slightly *worse* with a longer history at 60 min (+0.12 to +0.25 for all feature sets). Evening events may act as noise for overnight forecasts.
8. **The raw-over-physiological crossover is driven by MEAL and BOLUS windows.**
   - At 60 min with 4-h history, B+steps beats C+steps by 0.41 (MEAL) and 0.61 (BOLUS).
   - C+steps keeps a small edge in event-free windows (60 min, 1 h: STABLE_BASELINE +0.27 in its favour; 4 h: NOCTURNAL +0.19).
   - Likely cause: IOB is computed from total insulin (basal + bolus), so C carries basal information that B+steps lacks, and basal drives fasting and overnight glucose. This can be tested with `bg carbs bolus basal steps`.
9. **EXERCISE windows are a clear negative result at 60 min.**
   - All six covariate configurations (both encodings × three histories) are 0.74–1.75 mg/dL worse than CGM-only A. This is the only window where covariates systematically hurt; at 30 min they give a small gain.
   - A plausible cause: the model learns "more steps → glucose falls" from mostly non-exercise data, but glucose 1 h after exercise varies by person and exercise type (fall or rebound). Grid-aligned step counts mislead longer-horizon forecasts.
   - Caveats: EXERCISE has ≈ 5,100 test samples, possibly from few subjects, and C+steps at 2 h (37.35) is out of line with 1 h and 4 h, which suggests a seed outlier. The seed std (`rmse_EXERCISE_std` in `summary_by_config.csv`) and a per-subject breakdown still need to be checked.
10. **In windows that combine an event with exercise, B+steps is consistently better than C+steps** (BOLUS+EXERCISE: −0.67 to −1.76). The two differ in raw 5-min steps vs a 30-min rolling mean, and the smoothing may blur exercise onset and offset. These windows have only 670–2,100 samples, so this is a hypothesis; `bg iob cob steps` would test it.
11. **Event windows remain far harder than event-free windows.**
    - Best 30-min model: MEAL 27.24 and MEAL+BOLUS 26.64 vs STABLE_BASELINE 18.89 and NOCTURNAL 14.48, i.e. 1.4–1.9×.
    - The best configurations close only ≈ 14% of the MEAL vs BASELINE+NOCTURNAL gap: 30 min 11.2 → 9.6; 60 min 21.0 → 18.2.
    - The 21% of meals that start after the forecast origin cannot be helped by any input length.

**Reference configurations for the event-token model**

| Horizon | Best grid-aligned baseline | Overall | MEAL | BOLUS | MEAL+BOLUS | EXERCISE |
|---|---|---|---|---|---|---|
| 30 min | B+steps, 2-h history | 20.98 | 27.24 | 23.53 | 26.70 | 21.67 |
| 30 min | B+steps, 4-h history | 20.95 | 27.37 | 23.53 | 26.74 | 21.50 |
| 60 min | B+steps, 4-h history | 34.93 | 47.25 | 40.33 | 43.94 | 36.50 (A: 35.49) |

---

## Caveats

- Seed std only measures training variance. No per-subject significance test has been run yet (paired Wilcoxon / bootstrap). Differences ≲ 0.1–0.3 — C vs B, history effects, small windows — are not established.
- Per-window seed std is not shown in this file. It is stored as `rmse_<window>_std` in each `summary_by_config.csv`.
- Windows that combine an event with exercise (MEAL+EXERCISE, BOLUS+EXERCISE, MEAL+BOLUS+EXERCISE) have only 670–2,100 test samples each and may come from few subjects.
- There are no held-out subjects. Results describe future forecasting for subjects seen in training.
- Window labels are taken at the target timestamp, so part of every event window is not observable from the input (see the event-location table in Part 1).
- `iob` is computed from total insulin (basal + bolus), so C / C+steps carry basal information that B / B+steps do not.
- Part 1 and Part 2 use different test sets (224,687 vs 217,135 samples). Compare absolute numbers only within a part.

## Next steps

1. **Verify the EXERCISE result:** check the per-window seed std and run a per-subject breakdown of the 60-min EXERCISE degradation (many subjects or one or two?).
2. **Per-subject paired tests:** A vs B+steps, B+steps vs C+steps, 1 h vs 4 h history; overall and per window.
3. **Two feature ablations** (only `--features` changes, aligned sets, 3 seeds):
   - `bg carbs bolus basal steps` — does basal explain C's edge in event-free windows?
   - `bg iob cob steps` — does raw vs smoothed steps explain B's edge in exercise windows?
4. **LSTM in the same pipeline:** same samples (stride 1), MSE loss, same normalisation, 3 seeds, so the backbone comparison is fair.
5. **Visibility-split evaluation:** report MEAL / BOLUS windows separately for "event in input window", "event before window" and "event after origin".
6. **Design implications for the event-token model (GLEAM):**
   - CGM window of 1–2 h; event lookback ≥ 4 h, including basal insulin;
   - a richer activity representation than step counts (intensity, duration, time since the bout ended);
   - one model per horizon;
   - report EXERCISE separately — every grid-aligned baseline fails there at 60 min;
   - compare against the reference configurations above.
