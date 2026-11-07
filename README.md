# ecgrobust

A reusable framework for testing how wearable single-lead ECG classifiers hold up under real recorded noise,
with atrial fibrillation (AF) detection as the worked example.

Author: Tejas Menneni, Columbus State Community College.

Archived on Zenodo: [doi:10.5281/zenodo.24042752](https://doi.org/10.5281/zenodo.24042752) (all versions; the manuscript cites v1.0.2, [doi:10.5281/zenodo.24042753](https://doi.org/10.5281/zenodo.24042753)).

| Role | Dataset | What it gives the study |
|---|---|---|
| Development + internal test | PhysioNet/CinC 2017 (8,528 AliveCor single-lead ECGs, 300 Hz, 9–60 s) | Realistic consumer-device recordings |
| External test | PTB-XL v1.0.3, lead I only, resampled 500→300 Hz (21,799 ECGs, 18,869 patients) | Different device, hospital and population; patient IDs |
| Noise | MIT-BIH Noise Stress Test Database (bw, ma, em), resampled 360→300 Hz | Real baseline wander, muscle artifact and electrode motion |

The study compares four models: a single-threshold rule on beat irregularity (RMSSD / mean RR), logistic regression,
gradient boosting (LightGBM), and a 1-D residual CNN on the raw signal.

## Key results

Full run, 5 seeds; 95% bootstrap CIs are in `outputs/tables/`.

- Clean internal AUROC: CNN 0.984, gradient boosting 0.981, logistic regression 0.972, rule 0.848. On PTB-XL the learned models reach 0.987–0.990 and the rule 0.926.
- Under electrode motion at −6 dB, AUROC drops by 0.09–0.12 for the learned models and by 0.30 for the rule. The CNN and gradient boosting do not differ significantly.
- The models fail in opposite directions. Beat-based models lose specificity and raise false alarms (gradient boosting 0.98 → 0.82); the CNN loses sensitivity and misses AF (0.85 → 0.42).
- Noise augmentation costs nothing on clean data and adds up to 0.030 AUROC and 0.18 F1 under severe noise.
- Quality-gated abstention helps little, and a threshold set on training data withholds 52% of clean PTB-XL recordings.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q                                            # unit tests (metrics match scikit-learn, exact SNR, etc.)
python -m ecgrobust all --config configs/smoke.yaml --unlock   # 4-minute synthetic end-to-end check
```

Full study:

```bash
python scripts/download_data.py                       # ~1.8 GB from physionet.org
python -m ecgrobust prepare
python -m ecgrobust split                             # the 20% internal test set, made once
python -m ecgrobust features
python -m ecgrobust train                             # resumable; --models cnn to run one model
python -m ecgrobust unlock-test                       # only when model development is finished
python -m ecgrobust evaluate
python -m ecgrobust report
```

`scripts/finish_study.py --config <cfg> --train-log <log>` can instead wait for training to finish, check that every
CNN converged, and then run the last three stages unattended.

On Windows, keep the virtual environment on a short path (e.g. `C:\venvs\ecg`); pip fails on paths longer than 260 characters.

## Design decisions

**Labels.** The binary task is AF vs not-AF (Normal + Other), with the noisy class `~` dropped. The four-class
model and the official challenge score, mean(F1<sub>N</sub>, F1<sub>A</sub>, F1<sub>O</sub>), are there for comparison with
earlier work. The data ship with three label versions: `REFERENCE-original.csv` (5,154 N / 771 A / 2,557 O / 46 ~),
`REFERENCE.csv` (5,050 / 738 / 2,456 / 284, the version most papers report) and the post-challenge relabel
`REFERENCE-v3.csv` (5,076 / 758 / 2,415 / 279). The default is `REFERENCE.csv`; set `data.cinc2017.reference` to
rerun on v3 as a sensitivity analysis. The pipeline logs which file it loaded and its class counts.

**Internal test lock.** `split` stratifies on the four original classes, so the AF fraction matches, and writes
`splits.json` with a SHA-256 hash of the test IDs. It refuses to overwrite an existing split. Training code asserts that
no test ID reaches it, and `evaluate` refuses to run until `unlock-test` has been called. That call records a timestamp
and the list of frozen model files, which documents that the test set was used once.

**No patient IDs in CinC 2017.** CV folds and the hold-out split are at the record level, which the paper lists as a
limitation. External confidence intervals on PTB-XL use a patient-level cluster bootstrap.

**Features** (`src/ecgrobust/features.py`). Signals go through NeuroKit2 cleaning and automatic polarity correction,
then R-peak detection with artifact correction turned off, because correction would replace the irregular RR intervals
that define AF.
- RR irregularity: SDNN, RMSSD, pNN50, pNN20, CV, RMSSD/mean RR, Poincaré SD1/SD2/ratio, MAD, IQR, turning-point ratio, sample entropy, and COSEn (Lake & Moorman, designed for short AF series).
- Atrial activity: P-wave presence fraction, P/R amplitude ratio and its variability, PR-interval SD/CV (DWT delineation), and a delineation-free P-region template correlation.
- QRS consistency, mainly for the four-class task.
- No frequency-domain HRV, because LF/HF are not interpretable on 9–60 s recordings.
- Signal quality (`q_*`, not model inputs): NeuroKit2 averageQRS (used for abstention), Zhao 2018, kSQI, pSQI, basSQI.

**Tuning.** 5-fold CV on training data only, selected by AUPRC. `StratifiedGroupKFold` keeps the noisy copies of a
record in the same fold. All learners use class weighting. Platt scaling is fitted on the out-of-fold predictions; it is
monotone, so AUROC and AUPRC do not change, and it corrects the probability distortion from class weighting. The decision
threshold maximizes F1 on the calibrated out-of-fold predictions, so no threshold is ever set on test data.

**CNN.** Six residual blocks on z-scored 9 s windows. Each epoch uses one random crop per record, with random polarity
flips and gradient clipping, over a full 60-epoch one-cycle learning-rate schedule; the best epoch on an inner 15% split
is kept. At test time it averages probabilities over windows with a 3 s stride. The peak learning rate is chosen once by
5-fold CV (binary task, standard training, seed 0) and reused. Per-seed 5-fold out-of-fold prediction would cost about
6× more on a CPU, so the CNN's Platt calibration and threshold come from the inner 15% split
(`models.cnn.calibration: holdout`; set `oof` on a GPU).

Two earlier CNN versions were discarded before the test set was unlocked, and both are archived in `outputs/dev_archive/`.
The first (cosine schedule, 30 epochs) was still improving at its last epoch. In the second, early stopping fired during a
transient divergence near the peak learning rate. `scripts/check_cnn_converged.py` now refuses to unlock the test set
unless every CNN's validation curve has plateaued.

**Noise** (`src/ecgrobust/noise.py`). Power-based SNR ∈ {24, 18, 12, 6, 0, −6} dB. Each NSTDB record is split in time:
the first half is used only for training augmentation and the second half only for test corruption. For a given record,
noise type and realization, the same noise segment is used at every SNR with only its gain changed, and every model sees
the same corrupted signals, so comparisons are paired.

**Beat detection under noise.** R-peaks on the noisy signal are matched (±150 ms, AAMI EC57) to the peaks found on the
clean signal, which serve as a pseudo-reference because CinC has no beat annotations.

**Augmentation.** Feature models get one noisy copy of each training record (random noise type, SNR ~ U[−6, 24] dB),
generated from a deterministic table. The CNN gets on-the-fly corruption with p = 0.5 from the same noise partition.

**Abstention.** The report gives two views. The first is a coverage curve that discards the lowest-quality recordings
first within each condition. The second uses fixed thresholds set in advance at the 1st, 5th and 10th percentile of
training quality and reports how many recordings each noise level pushes into "inconclusive", along with accuracy on the
rest. A device designer would deploy the second kind.

**Statistics.** Five seeds per model. The point estimate is the mean of the per-seed metrics, ± their SD. The 95% CI
comes from 1,000 bootstrap replicates of the test records, averaging the per-seed metrics within each replicate. All
models and conditions share the same replicate indices, so differences (model vs model, noisy vs clean, augmented vs
standard) are paired bootstrap tests, Holm-adjusted within each family.

## Outputs

`outputs/tables/` holds the tables listed at the top of `src/ecgrobust/report.py`: cohorts, main results, failure modes
(sensitivity and specificity per condition), external subgroups, the four-class score, noise degradation, paired tests,
augmentation effect, beat detection, performance stratified by beat-detection quality, abstention, calibration, and SHAP
importance and rank stability, plus `summary.md`.

| File in `outputs/figures/` | Content |
|---|---|
| `fig1_noise_{auroc,auprc,f1}.png` | Metric vs SNR, one panel per noise type, 95% CI bands |
| `fig2_beat_detection.png` | NeuroKit2 R-peak F1 vs SNR; classifier AUROC vs beat-detection F1 |
| `fig3_augmentation_auroc.png` | Standard (solid) vs noise-augmented (dashed) training |
| `fig4_abstention.png` | F1 on retained recordings vs fraction abstained |
| `fig5_calibration.png` | Reliability curves, clean vs electrode motion |
| `fig6_shap.png` | Gradient-boosting SHAP importance, and how it shifts with noise |

## Compute

Measured on an 8-core laptop CPU (no GPU), plugged in:
- Feature extraction: 45 min.
- Feature models: ~45 min.
- CNN: 21 fits at ~25–37 min each, ≈ 11 h.
- Evaluation: 82 min.
- Report: 2.6 h.

A GPU should shorten the CNN stage considerably (not measured here). Laptops with Modern Standby pause the run whenever the
screen turns off; `scripts/keep_display_on.py` keeps the display on until the run finishes. Every stage is resumable, and
CNN cross-validation folds are checkpointed.

## Reproducibility

Every random process is seeded from `configs/default.yaml`. Exact package versions are in `requirements.txt` and
`pip-freeze.txt`, and each run writes `environment_*.json`. A tagged release archived on Zenodo (`.zenodo.json`,
`CITATION.cff`) gives the code a DOI.

## Data availability

All data are openly available from PhysioNet: the PhysioNet/Computing in Cardiology Challenge 2017
(https://doi.org/10.13026/D3D0-0F49), PTB-XL v1.0.3 (https://doi.org/10.13026/KFZX-AW45), and the MIT-BIH Noise
Stress Test Database (https://doi.org/10.13026/C2HS3T). No new data were collected.

The manuscript will be added here once the medRxiv preprint is posted. `scripts/build_manuscript_docx.py` and
`scripts/build_supplement_docx.py` build the submission Word files, `scripts/docx_to_pdf.ps1` exports them to PDF, and
`scripts/make_flow_diagram.py` draws the participant flow diagram (`fig0_flow.png`). The TRIPOD+AI reporting
checklist, the submission checklist and a medRxiv form guide are in `docs/`.
