
## Tables

**Table 1.** Cohorts.

| Set | Recordings | Patients | Binary-task n | AF n (%) | Duration, s, median (IQR) | Age, y, median (IQR) | Female % |
|---|---|---|---|---|---|---|---|
| CinC 2017 development | 6,822 | not available | 6,595 | 590 (8.9) | 30.0 (30.0–30.0) | not available | not available |
| CinC 2017 internal test | 1,706 | not available | 1,649 | 148 (9.0) | 30.0 (30.0–30.0) | not available | not available |
| PTB-XL lead I (external) | 21,799 | 18,869 | 21,799 | 1,514 (6.9) | 10.0 (10.0–10.0) | 62.0 (50.0–72.0) | 47.9 |

CinC 2017 class counts (development / test): N 4040, O 1965, A 590, ~ 227 ; N 1010, O 491, A 148, ~ 57. The noisy class (~) is excluded from the binary task.

**Table 2.** Performance on clean recordings, standard training. Mean over five seeds [95% bootstrap CI; patient-level for PTB-XL].

| Test set | Model | AUROC | AUPRC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Internal (n = 1,649) | Rule (RMSSD/RR) | 0.848 [0.821–0.871] | 0.259 [0.216–0.309] | 0.400 [0.344–0.451] | 0.730 [0.650–0.794] | 0.811 [0.790–0.831] |
| Internal (n = 1,649) | Logistic regression | 0.972 [0.955–0.986] | 0.845 [0.787–0.901] | 0.785 [0.736–0.831] | 0.818 [0.760–0.873] | 0.974 [0.966–0.981] |
| Internal (n = 1,649) | Gradient boosting | 0.981 [0.973–0.989] | 0.874 [0.824–0.920] | 0.814 [0.767–0.859] | 0.853 [0.797–0.906] | 0.976 [0.968–0.983] |
| Internal (n = 1,649) | 1-D CNN | 0.984 [0.977–0.990] | 0.842 [0.777–0.901] | 0.808 [0.761–0.852] | 0.849 [0.791–0.901] | 0.975 [0.968–0.982] |
| PTB-XL (n = 21,799) | Rule (RMSSD/RR) | 0.926 [0.922–0.930] | 0.323 [0.305–0.344] | 0.467 [0.447–0.488] | 0.666 [0.641–0.691] | 0.912 [0.907–0.916] |
| PTB-XL (n = 21,799) | Logistic regression | 0.987 [0.984–0.990] | 0.884 [0.865–0.901] | 0.827 [0.812–0.842] | 0.807 [0.786–0.827] | 0.989 [0.988–0.991] |
| PTB-XL (n = 21,799) | Gradient boosting | 0.990 [0.988–0.992] | 0.894 [0.876–0.911] | 0.853 [0.839–0.867] | 0.842 [0.823–0.861] | 0.990 [0.989–0.992] |
| PTB-XL (n = 21,799) | 1-D CNN | 0.987 [0.985–0.990] | 0.877 [0.858–0.895] | 0.831 [0.817–0.845] | 0.863 [0.845–0.882] | 0.984 [0.982–0.986] |

**Table 3.** AUROC under real noise (internal test set, standard training): value [95% CI]; Δ = paired change versus clean [95% CI].

| Noise, SNR | Rule | Logistic regression | Gradient boosting | 1-D CNN |
|---|---|---|---|---|
| Clean | 0.848 [0.821–0.871] | 0.972 [0.955–0.986] | 0.981 [0.973–0.989] | 0.984 [0.977–0.990] |
| Muscle artifact, +6 dB | 0.787 [0.755–0.814]; Δ −0.061 [−0.074 to −0.048] | 0.972 [0.954–0.986]; Δ −0.000 [−0.006 to +0.006] | 0.977 [0.963–0.987]; Δ −0.005 [−0.012 to +0.000] | 0.980 [0.972–0.987]; Δ −0.004 [−0.006 to −0.001] |
| Muscle artifact, 0 dB | 0.691 [0.653–0.725]; Δ −0.157 [−0.179 to −0.135] | 0.967 [0.952–0.980]; Δ −0.005 [−0.014 to +0.004] | 0.967 [0.952–0.979]; Δ −0.015 [−0.022 to −0.008] | 0.971 [0.961–0.980]; Δ −0.012 [−0.019 to −0.007] |
| Muscle artifact, −6 dB | 0.614 [0.570–0.655]; Δ −0.234 [−0.274 to −0.199] | 0.935 [0.908–0.960]; Δ −0.037 [−0.062 to −0.014] | 0.940 [0.922–0.956]; Δ −0.042 [−0.055 to −0.030] | 0.938 [0.919–0.956]; Δ −0.046 [−0.063 to −0.030] |
| Electrode motion, +6 dB | 0.790 [0.759–0.818]; Δ −0.058 [−0.071 to −0.045] | 0.974 [0.965–0.983]; Δ +0.002 [−0.009 to +0.017] | 0.975 [0.965–0.984]; Δ −0.006 [−0.012 to −0.000] | 0.976 [0.966–0.984]; Δ −0.008 [−0.013 to −0.003] |
| Electrode motion, 0 dB | 0.681 [0.645–0.716]; Δ −0.167 [−0.191 to −0.142] | 0.952 [0.939–0.965]; Δ −0.020 [−0.034 to −0.003] | 0.954 [0.938–0.970]; Δ −0.027 [−0.039 to −0.015] | 0.958 [0.942–0.972]; Δ −0.026 [−0.038 to −0.014] |
| Electrode motion, −6 dB | 0.553 [0.510–0.598]; Δ −0.295 [−0.333 to −0.255] | 0.855 [0.824–0.884]; Δ −0.117 [−0.147 to −0.087] | 0.884 [0.857–0.911]; Δ −0.097 [−0.121 to −0.075] | 0.897 [0.870–0.922]; Δ −0.087 [−0.111 to −0.063] |

Baseline wander changed AUROC by ≤ 0.010 at every SNR (full grid in `noise_degradation.csv`).

**Table 4.** Direction of errors: sensitivity and specificity at the training-set threshold [95% CI].

| Training | Model | Clean sens. | Clean spec. | EM −6 dB sens. | EM −6 dB spec. | MA −6 dB sens. | MA −6 dB spec. |
|---|---|---|---|---|---|---|---|
| standard | Rule (RMSSD/RR) | 0.730 [0.650–0.794] | 0.811 [0.790–0.831] | 0.905 [0.858–0.951] | 0.252 [0.232–0.274] | 0.919 [0.874–0.958] | 0.249 [0.227–0.270] |
| standard | Logistic regression | 0.818 [0.760–0.873] | 0.974 [0.966–0.981] | 0.715 [0.655–0.780] | 0.814 [0.796–0.831] | 0.673 [0.608–0.746] | 0.970 [0.962–0.977] |
| standard | Gradient boosting | 0.853 [0.797–0.906] | 0.976 [0.968–0.983] | 0.759 [0.697–0.823] | 0.818 [0.800–0.835] | 0.697 [0.630–0.764] | 0.953 [0.944–0.962] |
| standard | 1-D CNN | 0.849 [0.791–0.901] | 0.975 [0.968–0.982] | 0.424 [0.359–0.499] | 0.984 [0.978–0.989] | 0.399 [0.333–0.468] | 0.990 [0.985–0.994] |
| augmented | Rule (RMSSD/RR) | 0.730 [0.650–0.794] | 0.811 [0.790–0.831] | 0.905 [0.858–0.951] | 0.252 [0.232–0.274] | 0.919 [0.874–0.958] | 0.249 [0.227–0.270] |
| augmented | Logistic regression | 0.800 [0.738–0.860] | 0.977 [0.970–0.984] | 0.689 [0.625–0.760] | 0.866 [0.850–0.881] | 0.622 [0.544–0.698] | 0.983 [0.977–0.989] |
| augmented | Gradient boosting | 0.851 [0.797–0.906] | 0.977 [0.970–0.984] | 0.659 [0.589–0.731] | 0.930 [0.917–0.941] | 0.662 [0.589–0.735] | 0.974 [0.967–0.981] |
| augmented | 1-D CNN | 0.855 [0.800–0.905] | 0.974 [0.967–0.981] | 0.526 [0.461–0.597] | 0.981 [0.974–0.987] | 0.674 [0.614–0.738] | 0.977 [0.970–0.984] |

EM, electrode motion; MA, muscle artifact.

**Table 5.** Quality-gated abstention with a fixed threshold at the 5th percentile of training quality (NeuroKit2 averageQRS ≥ 0.614).

| Condition | Withheld, all | Withheld, AF | GB AUROC (retained) | GB F1 (retained) | CNN AUROC (retained) | CNN F1 (retained) |
|---|---|---|---|---|---|---|
| Internal, clean | 5.2% | 7.4% | 0.981 | 0.818 | 0.983 | 0.802 |
| Internal, EM +6 dB | 7.6% | 14.2% | 0.976 | 0.737 | 0.976 | 0.775 |
| Internal, EM 0 dB | 13.0% | 20.9% | 0.957 | 0.627 | 0.962 | 0.711 |
| Internal, EM −6 dB | 24.4% | 32.4% | 0.897 | 0.422 | 0.908 | 0.552 |
| PTB-XL, clean | 51.5% | 46.8% | 0.989 | 0.861 | 0.987 | 0.835 |

GB, gradient boosting. Without abstention (internal, EM −6 dB): GB AUROC 0.884, F1 0.424; CNN 0.897, 0.534.

## Figure legends

**Figure 1.** AUROC versus SNR for each noise type and model (standard training); shaded bands are 95% bootstrap CIs. (`fig1_noise_auroc.png`)

**Figure 2.** (a) NeuroKit2 R-peak detection F1 versus SNR, against clean-signal detections (±150 ms). (b) Classifier AUROC versus beat-detection F1 across all corrupted conditions. (`fig2_beat_detection.png`)

**Figure 3.** AUROC under noise with standard (solid) and noise-augmented (dashed) training. (`fig3_augmentation_auroc.png`)

**Figure 4.** F1 on retained recordings as the lowest-quality recordings are withheld, on clean data and electrode motion at 6 dB. (`fig4_abstention.png`)

**Figure 5.** Reliability curves (seed-averaged probabilities, equal-mass bins) for gradient boosting and the CNN, clean versus electrode motion at 12 and 0 dB. (`fig5_calibration.png`)

**Figure 6.** (a) Gradient-boosting feature importance (share of mean |SHAP|) on clean data. (b) Importance across electrode-motion SNR levels. (`fig6_shap.png`)
