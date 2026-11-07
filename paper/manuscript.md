# A real-noise stress-testing framework for wearable ECG classifiers: development and external validation with atrial fibrillation detection as a worked example

**Author:** Tejas Menneni, Columbus State Community College, Columbus, Ohio, USA

**Corresponding author:** Tejas Menneni (tejas.menneni@gmail.com)

> Draft status: complete first draft. All numbers were copied from `outputs/tables/` (run of 2026-11-07; test set unlocked
> 03:40 after all models were frozen). To regenerate a number, see the table named next to each result. Before submission:
> add the Zenodo DOI, confirm the funding and competing-interest statements, and confirm the abstract word limit for the
> target journal.

---

## Abstract

**Objective.** Wearable single-lead electrocardiograms (ECGs) are often corrupted by motion and muscle noise, yet atrial
fibrillation (AF) classifiers are usually evaluated on clean recordings. We propose a reproducible framework for
stress-testing wearable ECG classifiers with real recorded noise, demonstrated on AF detection.

**Methods.** Four classifiers were developed on 8,528 handheld recordings (PhysioNet/CinC 2017): an irregularity rule,
logistic regression and gradient boosting on beat-timing and P-wave features, and a 1-D convolutional neural network
(CNN). They were tested on a locked internal set and externally on PTB-XL lead I (21,799 recordings, 18,869 patients),
clean and corrupted with real noise from the MIT-BIH Noise Stress Test Database (24 to −6 dB). We also assessed beat
detection, noise augmentation and quality-based abstention (five seeds; paired bootstrap).

**Results.** Clean internal AUROC was 0.972–0.984 for the learned models versus 0.848 for the rule (external: 0.987–0.990).
Under electrode motion at −6 dB, AUROC fell by 0.09–0.12 for the learned models and by 0.30 for the rule, and the CNN and
gradient boosting did not differ significantly. The models failed in opposite directions. Beat-based models lost specificity
(gradient boosting 0.98→0.82), whereas the CNN lost sensitivity (0.85→0.42). Noise augmentation had no cost on clean
data and improved AUROC by up to 0.030 and F1 by up to 0.18 under severe noise. Quality-gated abstention helped little,
and a training-derived quality threshold rejected 52% of clean external recordings.

**Conclusion.** Aggregate discrimination hides clinically different failure modes under real noise. Robustness
evaluations should report error direction, detector accuracy and quality-gate transferability. Code and data are
openly available.

**Keywords:** atrial fibrillation; wearable ECG; robustness; noise stress test; external validation; abstention; TRIPOD+AI

---

## 1. Introduction

Smartwatches and handheld devices now record single-lead ECGs from millions of people, and AF detection is their flagship
clinical application [Perez 2019]. The recordings are short, taken by untrained users, and are often degraded by electrode
motion, muscle activity and baseline drift. In a clinical comparison of five consumer devices, the devices' own
algorithms labelled 17–26% of single-lead tracings inconclusive [Mannhart 2023]. A classifier that performs well on
clean data can still fail on these noisy recordings.

Most single-lead AF classifiers, including the large literature built on the PhysioNet/Computing in Cardiology (CinC)
Challenge 2017 [Clifford 2017], are evaluated on held-out recordings from the same device and population. Robustness to noise
is measured less often. Venton et al. added physiological noise to ECGs and found that a CNN trained on clean data lost up
to 0.18 F1, while training on noisy data limited the loss [Venton 2021]. Studies that use synthetic noise (white noise,
sinusoids) miss the non-stationary, morphology-mimicking character of real electrode motion. It is also rarely asked why a
model fails under noise, whether the failure can be reduced, and what a device should do when it cannot classify a recording
reliably.

We built a reusable evaluation framework to answer these questions and applied it to AF detection. It has six parts:

1. Real recorded noise from the MIT-BIH Noise Stress Test Database [Moody 1984], added at controlled SNR, with separate
   noise segments for training and testing.
2. Paired evaluation: every model sees the same corrupted signals, so model differences are tested on matched data.
3. A diagnostic that separates failures of the beat detector from failures of the classifier.
4. Noise-augmented training as a remedy.
5. Quality-gated abstention, reported as operating points a device designer can choose from.
6. External validation on a different device and population with patient identifiers [Wagner 2020].

The code and configuration are open source, and all data are public.

## 2. Methods

Reporting follows TRIPOD+AI [Collins 2024] (Supplementary Table S4).

### 2.1 Data sources

**Development data.** We used the 8,528 single-lead ECGs of the CinC 2017 training set [Clifford 2017], recorded with an
AliveCor handheld device at 300 Hz and lasting 9–60 s. We used the reference labels distributed with the data
(`REFERENCE.csv`; 5,050 normal, 738 AF, 2,456 other rhythm, 284 noisy), which are the labels used by most published work. The
public release contains no patient identifiers.

**External data.** PTB-XL v1.0.3 [Wagner 2020] contains 21,799 ten-second 12-lead ECGs from 18,869 patients. They were recorded at 500 Hz
with clinical devices. We used lead I only, the limb lead closest to what a wrist or handheld device measures, and resampled it to
300 Hz with a polyphase filter. A record was labelled AF if its cardiologist-assigned SCP statements included `AFIB`
(1,514 records). All other records, including atrial flutter, were labelled non-AF, mirroring the CinC labelling, in
which flutter belongs to the "other rhythm" class.

**Noise.** The MIT-BIH Noise Stress Test Database [Moody 1984] provides 30-minute, two-channel recordings of baseline
wander (bw), muscle artifact (ma) and electrode motion (em), captured from ambulatory ECG electrodes placed so that no
ECG was recorded. Records were resampled from 360 to 300 Hz.

### 2.2 Outcomes and tasks

The primary task was binary: AF versus non-AF (normal and other rhythm). Records labelled noisy were excluded because
signal corruption is the experimental variable of this study. For comparability with the CinC 2017 literature, we also
trained four-class models and report the official challenge score, the mean F1 over the normal, AF and other classes.

### 2.3 Internal hold-out and test-set lock

Before any modelling, 20% of the CinC recordings (n = 1,706; 148 AF) were set aside by stratified random sampling on the
four original classes. The split was written once together with a SHA-256 hash of the test identifiers. Training code
asserts that no test identifier reaches it, and evaluation code refuses to run until model development has been formally
frozen. The freeze is recorded with a timestamp and the list of model files that existed at that moment. Two design
revisions happened before the freeze, both based only on training and validation curves. First, the CNN schedule was
changed after training curves showed it was still improving at its epoch limit. Second, early stopping was replaced by
best-epoch selection over the full schedule, and gradient clipping was added, after two cross-validation folds stopped at a
transient divergence near the peak learning rate (Section 2.5). Before unlocking, every final CNN was required to show a
validation plateau (less than 0.01 AUPRC gain over its last 10 epochs).

### 2.4 Signal processing and features

Signals were cleaned with NeuroKit2 [Makowski 2021] after automatic polarity correction. R peaks were detected with the NeuroKit2
detector with artifact correction turned off, because correction methods [Lipponen 2019] replace irregular intervals and
would remove the irregularity that defines AF.

*Beat-timing irregularity.* We computed the following on RR intervals within 250–2000 ms, with successive differences taken
only between adjacent valid intervals: mean and median RR, heart rate, SDNN, RMSSD, pNN50, pNN20, coefficient of variation,
RMSSD normalised by mean RR, Poincaré SD1, SD2 and SD1/SD2, normalised median absolute deviation and interquartile range,
turning-point ratio, sample entropy (m = 2, r = 0.2·SD) [Richman 2000], and the coefficient of sample entropy (COSEn), which was
designed for short AF series [Lake 2011]. Frequency-domain heart rate variability measures were deliberately excluded because
they are not interpretable on recordings this short.

*Atrial activity.* Using discrete-wavelet delineation, we computed the fraction of beats with a detected P wave, the mean
and variability of the P-to-R amplitude ratio, and the standard deviation and coefficient of variation of the PR interval.
We also computed a delineation-free measure: the mean correlation of the 250–60 ms pre-R segment with its median template.

*Ventricular morphology* (mainly for the four-class task): mean QRS duration, QRS template correlation, fraction of beats
with correlation < 0.8, and R-amplitude variability.

*Signal quality* was computed for every recording but was not a model input: the NeuroKit2 average-QRS index (used for
abstention), the Zhao 2018 fuzzy index [Zhao 2018], kurtosis (kSQI), the relative QRS-band power (pSQI) and the baseline
power (basSQI) [Li 2008].

Features were undefined when fewer than five beats were detected. Logistic regression imputed missing values with
training medians and added missingness indicators; gradient boosting handled them natively; the rule used the training median.

### 2.5 Models

1. **Irregularity rule.** AF is flagged when RMSSD/mean RR exceeds a threshold. The feature was chosen *a priori*, and the
   threshold was set on training data.
2. **Logistic regression** with L1 or L2 penalty (saga solver), standardised inputs.
3. **Gradient boosting** (LightGBM [Ke 2017]), with randomized search over 40 configurations.
4. **1-D CNN**: six residual blocks (32–128 channels, kernel 9) on z-scored 9 s windows. Training used one random 9 s crop
   per recording per epoch, random polarity inversion, AdamW with gradient-norm clipping (1.0) and a 60-epoch one-cycle
   learning-rate schedule. The epoch with the best AUPRC on a stratified 15% inner validation split was retained. At inference, probabilities were averaged over 9 s windows with a 3 s
   stride covering the whole recording.

All learners used inverse-frequency class weights. Logistic-regression and gradient-boosting hyper-parameters were tuned
by stratified 5-fold cross-validation on the training set only, maximizing the area under the precision-recall curve (AUPRC).
The CNN's peak learning rate was chosen by 5-fold CV once (binary task, standard training, seed 0) and reused for the
other seeds.

**Calibration and thresholds.** Class weighting distorts predicted probabilities. Each model was therefore recalibrated with Platt scaling [Platt 1999]. For the
feature-based models, the scaling was fitted on out-of-fold training predictions. For the CNN, it was fitted on the inner
validation split, because per-seed out-of-fold CNN training was computationally prohibitive on CPU. Platt scaling is monotone and does not alter AUROC or AUPRC. The
decision threshold maximized F1 on the same calibration predictions. No threshold or calibration parameter was derived from test data.

### 2.6 Noise stress test

Each test recording was corrupted by adding a noise segment scaled to a target power-based SNR of 24, 18, 12, 6, 0 or −6 dB:
SNR = 10·log₁₀(P<sub>signal</sub>/P<sub>noise</sub>), with both powers computed after mean removal. Each NSTDB record was split in time.
The first half was reserved for training augmentation and the second half for test corruption, so no noise sample was
shared between training and evaluation. For a given recording and noise type, the *same* noise segment was used at every SNR
with only its gain changed, so differences across SNR levels reflect noise level rather than a different draw. All models
received identical corrupted signals. Features were re-extracted from every corrupted signal, so the feature-based models
experienced the full pipeline, including beat detection, under noise.

**Beat-detection accuracy.** R peaks detected on each corrupted signal were matched to those detected on the clean signal
within ±150 ms (ANSI/AAMI EC57 tolerance), giving sensitivity, positive predictive value and F1. Because CinC has no expert
beat annotations, clean-signal detections served as a pseudo-reference.

### 2.7 Noise augmentation

Each model was retrained with noise-augmented data. Each feature-based model received one corrupted copy of every training
recording. The noise type was drawn uniformly, the SNR uniformly from [−6, 24] dB, and the corruption came from a fixed,
seeded table using the training noise partition only. Cross-validation kept each recording and its noisy copy in the same
fold. The CNN was corrupted on the fly with probability 0.5 from the same distribution. Augmented models were calibrated on
the same clean-plus-noisy mixture.

### 2.8 Quality-based abstention

Models were allowed to return "inconclusive" for low-quality recordings, ranked by the NeuroKit2 average-QRS quality index.
Recordings for which no quality could be computed were treated as the lowest quality. We report two views:
(i) performance on the retained recordings as the lowest-quality 5–50% are discarded;
(ii) fixed quality thresholds set in advance at the 1st, 5th and 10th percentile of *training* quality. For these, we report the
abstention rate and the performance at each noise level. View (ii) mirrors how a deployed device operates.

### 2.9 Explainability and calibration under noise

For gradient boosting, we computed mean absolute SHAP values [Lundberg 2017] on clean and corrupted test features and
compared feature rankings with Kendall's τ. Calibration was summarised by the expected calibration error (10 equal-mass
bins), the calibration slope and intercept, and the Brier score, for clean and corrupted data.

### 2.10 Statistical analysis

Every experiment was repeated with five seeds, which change the CV folds, model initialisation, subsampling and the CNN's
data order; the test split was fixed. Point estimates are means over seeds, with the standard deviation across seeds. 95%
confidence intervals come from 1,000 bootstrap resamples of test recordings, averaging the per-seed metric within each
resample. For PTB-XL, resampling was by patient (cluster bootstrap). The same resampling indices were used for every
model and condition, so differences between models, between noise levels and clean, and between augmented and standard
training were evaluated with paired bootstrap tests. Two-sided p-values were Holm-adjusted within each family of
comparisons. We report sensitivity, specificity, positive and negative predictive value, F1, AUROC, AUPRC and Brier
score. Accuracy is not reported as a primary metric because AF prevalence is ~9%. To assess performance across
demographic groups, external performance was also reported by sex and by age (< 65 vs ≥ 65 years), with patient-level
bootstrap intervals. The CinC 2017 data contain no demographic information.

### 2.11 Software and reproducibility

Analyses used Python 3.13 with NeuroKit2 0.3.0, scikit-learn 1.9.1, LightGBM 4.7.0, PyTorch 2.15 and SHAP 0.54.1. All
random processes are seeded from a single configuration file. The code, configuration and exact package versions are
available at https://github.com/TSFIRE99-bot/wearable-ecg-noise-stress-test and archived at [Zenodo DOI].

## 3. Results

Values are means over five seeds (± SD across seeds) with 95% bootstrap confidence intervals. With 1,000 bootstrap
replicates, the smallest attainable p-value is 0.001, or 0.005 after Holm adjustment within a family of five comparisons.

### 3.1 Cohorts
*(Table 1 ← `table1_cohorts.csv`; participant flow: Supplementary Fig. S1 ← `fig0_flow.png`)* The development set contained 6,822 recordings, of which 6,595 were used for the binary task
(590 AF, 8.9%). The internal test set contained 1,706 recordings (1,649 binary; 148 AF, 9.0%). Most CinC recordings lasted
30 s. PTB-XL contributed 21,799 ten-second recordings from 18,869 patients (1,514 AF, 6.9%). The median age was 62 years (IQR 50–72), and 47.9% of
patients were female.

### 3.2 Performance on clean recordings
*(Table 2 ← `main_internal.csv`, `main_external.csv`; comparisons ← `paired_tests.csv`)* On the internal test set, the CNN
(AUROC 0.984 [0.977–0.990]), gradient boosting (0.981 [0.973–0.989]) and logistic regression (0.972 [0.955–0.986]) all
outperformed the irregularity rule (0.848 [0.821–0.871]). The paired AUROC differences were 0.124–0.136, the AUPRC differences
0.58–0.61, and the F1 differences 0.38–0.41 (all p < 0.005). Gradient boosting had the highest AUPRC (0.874 [0.824–0.920]) and F1
(0.814 [0.767–0.859], sensitivity 0.853, specificity 0.976). It exceeded logistic regression on AUPRC (+0.029
[0.007–0.058], p = 0.040) and did not differ significantly from the CNN on any metric (AUROC difference CNN − gradient boosting
+0.003 [−0.003 to 0.008]; AUPRC −0.032 [−0.070 to 0.005]).

Externally on PTB-XL, discrimination was maintained or higher: AUROC 0.990 [0.988–0.992] for gradient boosting, 0.987
[0.985–0.990] for the CNN, 0.987 [0.984–0.990] for logistic regression and 0.926 [0.922–0.930] for the rule. With the much
larger external sample, gradient boosting was slightly but significantly better than the CNN (AUROC +0.0025
[0.0007–0.0043], AUPRC +0.018 [0.006–0.031], F1 +0.022 [0.010–0.034]). External AUROC was similar by sex (gradient boosting
0.990 in both). It was marginally lower in patients aged ≥ 65 years (0.986 [0.983–0.989] vs 0.991 [0.986–0.995] for < 65), and the
gap was larger for the rule (0.894 vs 0.950) (Supplementary Table S2 ← `subgroups_external.csv`).

On the four-class task, the challenge score (mean F1 over normal, AF and other) was 0.803 [0.780–0.824] for the CNN (one seed),
0.799 (± 0.004) [0.776–0.821] for gradient boosting and 0.743 [0.718–0.769] for logistic regression (Supplementary Table S1 ←
`fourclass_challenge.csv`).

### 3.3 Degradation under real noise
*(Fig. 1 ← `fig1_noise_auroc.png`; Table 3 ← `noise_degradation.csv`)* Baseline wander had almost no effect at any SNR. The
largest AUROC change was −0.010, for the CNN at −6 dB, because NeuroKit2 cleaning and the CNN's window standardisation
remove most low-frequency power. Muscle artifact and electrode motion were harmless down to 12 dB, mildly harmful at 6 dB
and damaging at 0 and −6 dB. Electrode motion was the most damaging. At −6 dB, AUROC fell by 0.087 [0.063–0.111] for the
CNN, 0.097 [0.075–0.121] for gradient boosting, 0.117 [0.087–0.147] for logistic regression and 0.295 [0.255–0.333] for the
rule, which approached chance (0.553). The CNN and gradient boosting did not differ significantly in AUROC at any noise
level (largest difference +0.013 [−0.016 to 0.039] at −6 dB electrode motion, Holm-adjusted p = 0.37).

The models failed in opposite directions *(Table 4 ← `failure_modes.csv`)*. Under electrode motion at −6 dB, the
feature-based models produced false alarms. Specificity fell from 0.976 to 0.818 [0.800–0.835] for gradient boosting and
from 0.811 to 0.252 for the rule, while sensitivity was largely retained (gradient boosting 0.759; rule 0.905). Positive
predictive value collapsed to 0.296 and 0.107. The CNN instead missed AF: its specificity was preserved (0.984 [0.978–0.989])
but its sensitivity fell from 0.849 to 0.424 [0.359–0.499], with a PPV of 0.732. Muscle artifact at −6 dB showed the same pattern
(CNN sensitivity 0.40; feature-model specificity 0.95–0.97, rule 0.25). F1 therefore degraded more than AUROC for every
model. At −6 dB electrode motion it was 0.534 for the CNN, 0.424 for gradient boosting and 0.402 for logistic regression.

### 3.4 Beat detection and the feature models' failure
*(Fig. 2 ← `fig2_beat_detection.png`; `beat_detection.csv`; `performance_by_beat_quality.csv`)* Aggregate R-peak F1
stayed ≥ 0.99 for baseline wander at every level and for the other noise types down to 12 dB. It fell to 0.929 (electrode motion)
and 0.939 (muscle) at 0 dB, and to 0.829 and 0.861 at −6 dB. At −6 dB electrode motion, PPV (0.784) fell more than sensitivity (0.879): the
detector added spurious beats. The detector never failed outright (every recording had ≥ 5 detected beats), but per-recording
accuracy degraded widely: 62.0% of recordings had beat-detection F1 < 0.9 at −6 dB electrode motion, versus 2.6% at 12 dB.
Across conditions, the rule's AUROC tracked beat-detection accuracy almost linearly (Fig. 2b).

Pooling all corrupted internal recordings and stratifying by their own beat-detection F1 (descriptive; not bootstrapped),
the three learned models were indistinguishable when detection was accurate (F1 ≥ 0.95: AUROC 0.977–0.982; n = 24,970
recording-conditions). When detection was partly corrupted (F1 0.50–0.80; n = 1,228), the CNN retained more discrimination
than the feature-based models (0.786 vs 0.696 for gradient boosting and 0.701 for logistic regression). After augmentation the
gap widened (0.874 vs 0.758). In the few recordings with near-total detection failure (F1 < 0.50; n = 145), all models
performed poorly and estimates were unstable.

### 3.5 Noise augmentation
*(Fig. 3 ← `fig3_augmentation_auroc.png`; `augmentation_effect.csv`; unadjusted paired bootstrap p)* Augmented training
did not change clean internal performance for any model (|ΔAUROC| ≤ 0.002, |ΔF1| ≤ 0.003, all p ≥ 0.2). Under electrode
motion at −6 dB, it improved AUROC by 0.030 [0.019–0.040] for the CNN, 0.022 [0.011–0.034] for gradient boosting and 0.019
[0.012–0.026] for logistic regression (all p = 0.001). Gains in F1 were larger: +0.133 [0.099–0.169] for gradient boosting and
+0.072 [0.041–0.103] for the CNN. Under muscle artifact at −6 dB, CNN F1 rose by 0.175 [0.125–0.230]. Augmentation partly repaired each model's
failure mode. Gradient boosting's specificity at −6 dB electrode motion recovered from 0.818 to 0.930, and the CNN's sensitivity
rose from 0.424 to 0.526 under electrode motion and from 0.40 to 0.67 under muscle artifact. The rule has a single fixed feature, so augmentation
cannot change its discrimination by construction. Externally, augmentation slightly improved F1 for logistic regression
(+0.023 [0.015–0.031]) and the CNN (+0.008 [0.002–0.013]).

### 3.6 Abstention
*(Fig. 4 ← `fig4_abstention.png`; Table 5 ← `abstention_fixed_threshold.csv`; `abstention_curve.csv`)* Discarding the
lowest-quality recordings first improved performance only modestly. On clean data, gradient-boosting AUROC rose from
0.981 to 0.989 and F1 from 0.814 to 0.855 when 50% were withheld. Under electrode motion at −6 dB, AUROC rose from 0.884 to
0.903 at 50% abstention while F1 did not improve (0.424 → 0.419).

With a fixed threshold at the 5th percentile of training quality (averageQRS 0.614), the internal abstention rate rose with
noise: 5.2% on clean data, 7.6% at 6 dB, 13.0% at 0 dB and 24.4% at −6 dB electrode motion. AF recordings were withheld
disproportionately (7.4%, 14.2%, 20.9% and 32.4% of AF cases). Performance on the retained recordings barely changed (gradient
boosting at −6 dB: AUROC 0.897, F1 0.422; CNN: 0.908, 0.552). The same threshold applied to *clean* PTB-XL recordings withheld
51.5% of them, showing that a quality threshold calibrated on one device does not transfer to another.

### 3.7 Calibration and explanation under noise
*(Fig. 5 ← `fig5_calibration.png`, `calibration.csv`; Fig. 6 ← `fig6_shap.png`, `shap_rank_stability.csv`)* On clean
internal data, all learned models were well calibrated after Platt scaling (expected calibration error, ECE, ≤ 0.007). The CNN was slightly
overconfident (calibration slope 0.86). Under noise, the feature-based models became miscalibrated, overestimating AF
probability (ECE 0.053–0.056 at 0 dB and 0.118–0.126 at −6 dB electrode motion). The CNN stayed well calibrated (ECE 0.013 and
0.026). Externally, calibration slopes were 1.20 for gradient boosting (underconfident) and 0.82 for the CNN.

The most important gradient-boosting features on clean data were the P-region template correlation, pNN50, median RR
interval, RMSSD/mean RR and COSEn. The ranking was stable down to 6 dB (Kendall's τ vs clean ≥ 0.93) and shifted at 0 dB
(τ = 0.87) and −6 dB (τ = 0.78). At −6 dB, the fraction of beats with an atypical QRS morphology entered the top five, which
suggests that the model increasingly relied on a signature of noise-corrupted beats.

## 4. Discussion

We built a real-noise stress test for wearable ECG classifiers and applied it to AF detection with external validation.
On clean data, the three learned models (feature-based and end-to-end) performed almost identically and far better than
a single-feature rule, both internally and on an external clinical dataset. Under heavy real noise their AUROC also stayed
similar, but their errors went in opposite directions. Models built on detected beats produced false alarms, because
noise-induced spurious and missed R peaks look like an irregular rhythm. The CNN, which never uses detected beats, missed
AF. Noise-augmented training cost nothing on clean data and reduced both kinds of error. Quality-gated abstention helped
little, withheld AF recordings more often than others, and did not transfer to a second device.

**Error direction.** AUROC summarises ranking across all thresholds, but a deployed device uses one threshold, so it
matters which error grows. On a wearable used for screening, false alarms from beat-based models lead to unnecessary
clinical follow-up and erode user trust. Missed AF delays the diagnosis of a condition that raises stroke risk. Which error
is more acceptable depends on the use case. An evaluation that reported only AUROC, or only accuracy, would have rated these
models as equivalent, so we recommend reporting sensitivity and specificity at the deployed threshold for every noise
condition, as in Table 4.

**Beyond AF.** A classifier built on detected fiducial points (R peaks, P waves, QRS boundaries) can only be as robust as
its detector, and the framework measures how much that limits it. When we stratified performance by each recording's own
beat-detection accuracy, the feature-based models matched the CNN where detection was accurate and fell behind it where
detection was partly corrupted. The same protocol applies to other wearable ECG tasks, and to other signals for which
public real-noise recordings exist.

**Practical guidance.** (i) Test with electrode motion and muscle artifact. After standard cleaning, baseline wander had
almost no effect. (ii) Use noise augmentation by default. It cost nothing on clean data, internally or externally, and
was the only intervention that improved performance under severe noise. (iii) Do not assume that a signal-quality gate
fixes noise. The NeuroKit2 average-QRS index withheld up to a
quarter of severely corrupted recordings but barely improved accuracy on the rest. It preferentially withheld AF
recordings, which is consistent with earlier evidence that signal-quality indices perform worse during arrhythmias and
should be rhythm specific [Behar 2013]. A threshold set on training data would also have rejected half of the clean
recordings from a different device. Quality thresholds must be re-derived for each device and evaluated for their effect
on sensitivity, particularly since consumer devices already return inconclusive results for about one in four tracings
[Mannhart 2023].

**Comparison with prior work.** On the CinC 2017 data, Andreotti et al. found that a feature-based classifier and a CNN
reached similar cross-validated scores (72.0% and 72.1%) [Andreotti 2017], which matches the near-identical clean
performance of our feature-based models and CNN. Our four-class challenge scores (0.80 for the CNN and gradient boosting)
are slightly below the best scores on the challenge's hidden test set (0.83, a four-way tie [Clifford 2017]); those were
measured on a different test set and are not directly comparable to a single 20% hold-out. The binary AF results agree
with the literature in that RR-irregularity and P-wave features carry most of the information (COSEn, pNN50 and the
P-region correlation ranked highest), and the external results show that features designed on handheld recordings transfer
to clinical lead I. Our augmentation results agree with Venton et al., who found that training on noisy ECGs limited the
loss of CNN performance under noise [Venton 2021]. To our knowledge, real-noise evaluations of single-lead AF classifiers
that compare feature-based and end-to-end models, report the direction of errors, and test augmentation and abstention in one
paired design are rare.

### Limitations
1. The public CinC 2017 data have no patient identifiers. Splits and cross-validation are therefore at the recording level,
   and if individuals contributed several recordings, internal estimates may be optimistic. The external PTB-XL evaluation,
   with patient-level resampling, was designed to address this.
2. PTB-XL differs from the development data in more than device: lead I of a clinical 12-lead system, a fixed 10 s duration, and an older
   hospital population. External performance changes reflect these shifts together.
3. Noise was recorded from Holter chest electrodes and added to the signal. Artifacts that interact with the signal, such as
   saturation or loss of contact, are not represented.
4. SNR was power-based. The original NSTDB tool scales noise by QRS amplitude, so absolute SNR values are not
   directly comparable with studies using that convention.
5. Beat-detection accuracy was measured against clean-signal detections, not expert annotations.
6. The quality index used for abstention depends on beat detection and is itself degraded by noise. Results describe this
   practical gate, not an ideal one.
7. CNN hyper-parameter search was limited to the peak learning rate. The CNN was calibrated on an inner hold-out rather
   than out-of-fold predictions, and the four-class CNN used a single seed, all for computational reasons. The CNN
   training schedule was revised twice during development, before the test set was unlocked.
8. The stratification by beat-detection accuracy is descriptive. Bins are defined by an outcome of the noise itself and
   were not bootstrapped. The augmentation comparisons report unadjusted p-values.
9. A single noise realization per recording and condition was used. Variability across noise draws is not captured
   by the confidence intervals.

## 5. Conclusion
Under real recorded noise, single-lead AF classifiers with near-identical clean performance degraded to similar AUROC but
failed in opposite clinical directions. Beat-based models produced false alarms and an end-to-end CNN missed AF. Noise augmentation
reduced both failures at no cost on clean data, whereas quality-gated abstention offered little protection and did not transfer
across devices. We recommend that evaluations of wearable ECG classifiers routinely include real-noise stress tests with
paired comparisons, report sensitivity and specificity per noise condition, measure upstream detector accuracy, and validate
quality gates on the target device. The open-source code runs the full protocol from a single configuration file.

## Declarations

**Ethics.** This is a secondary analysis of de-identified data that are openly available on PhysioNet. No new data were
collected, and no additional ethical approval was required. [Check your institution's policy and cite each dataset's
original approval.]

**Data availability.** PhysioNet/Computing in Cardiology Challenge 2017 (https://doi.org/10.13026/D3D0-0F49); PTB-XL v1.0.3
(https://doi.org/10.13026/KFZX-AW45); MIT-BIH Noise Stress Test Database (https://doi.org/10.13026/C2HS3T).

**Code availability.** https://github.com/TSFIRE99-bot/wearable-ecg-noise-stress-test; archived at [Zenodo DOI]. MIT licence.

**Funding.** This research received no specific grant from any funding agency. [Confirm.]

**Competing interests.** The author declares no competing interests. [Confirm.]

**Author contributions (CRediT).** Tejas Menneni: Conceptualization, Methodology, Software, Formal analysis,
Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing.

**Declaration of generative AI use.** During the preparation of this work the author used the Anaconda Prompt AI
assistant and Claude (Anthropic) to help write the analysis code and draft the manuscript. The author reviewed and edited
all code and text and takes full responsibility for the content of the publication.

## References
*(Verify every entry against the original before submission; complete page ranges and DOIs.)*

- [Andreotti 2017] Andreotti F, Carr O, Pimentel MAF, Mahdi A, De Vos M. Comparing feature-based classifiers and convolutional neural networks to detect arrhythmia from short segments of ECG. *Computing in Cardiology* 2017;44.
- [Behar 2013] Behar J, Oster J, Li Q, Clifford GD. ECG signal quality during arrhythmia and its application to false alarm reduction. *IEEE Trans Biomed Eng* 2013;60(6):1660–1666. doi:10.1109/TBME.2013.2240452.
- [Clifford 2017] Clifford GD, Liu C, Moody B, et al. AF classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017. *Computing in Cardiology* 2017;44. doi:10.22489/CinC.2017.065-469.
- [Collins 2024] Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378.
- [Ke 2017] Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems* 30, 2017.
- [Lake 2011] Lake DE, Moorman JR. Accurate estimation of entropy in very short physiological time series: the problem of atrial fibrillation detection in implanted ventricular devices. *Am J Physiol Heart Circ Physiol* 2011;300:H319–H325.
- [Li 2008] Li Q, Mark RG, Clifford GD. Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter. *Physiol Meas* 2008;29:15–32.
- [Lipponen 2019] Lipponen JA, Tarvainen MP. A robust algorithm for heart rate variability time series artefact correction using novel beat classification. *J Med Eng Technol* 2019;43:173–181.
- [Lundberg 2017] Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* 30, 2017.
- [Makowski 2021] Makowski D, Pham T, Lau ZJ, et al. NeuroKit2: a Python toolbox for neurophysiological signal processing. *Behav Res Methods* 2021;53:1689–1696.
- [Mannhart 2023] Mannhart D, et al. Clinical validation of 5 direct-to-consumer wearable smart devices to detect atrial fibrillation: BASEL Wearable Study. *JACC Clin Electrophysiol* 2023;9(2):232–242. doi:10.1016/j.jacep.2022.09.011.
- [Moody 1984] Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors. *Computers in Cardiology* 1984;11:381–384.
- [Perez 2019] Perez MV, Mahaffey KW, Hedlin H, et al. Large-scale assessment of a smartwatch to identify atrial fibrillation. *N Engl J Med* 2019;381:1909–1917.
- [Platt 1999] Platt J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: *Advances in Large Margin Classifiers*. MIT Press; 1999.
- [Richman 2000] Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy. *Am J Physiol Heart Circ Physiol* 2000;278:H2039–H2049.
- [Venton 2021] Venton J, Harris PM, Sundar A, Smith NAS, Aston PJ. Robustness of convolutional neural networks to physiological electrocardiogram noise. *Phil Trans R Soc A* 2021;379(2212):20200262. doi:10.1098/rsta.2020.0262.
- [Wagner 2020] Wagner P, Strodthoff N, Bousseljot R-D, et al. PTB-XL, a large publicly available electrocardiography dataset. *Sci Data* 2020;7:154.
- [Zhao 2018] Zhao Z, Zhang Y. SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation. *Front Physiol* 2018;9:727. doi:10.3389/fphys.2018.00727.

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

**Supplementary Figure S1.** Participant flow: CinC 2017 split into development and locked internal test sets, exclusion of the noisy class for the binary task, PTB-XL external test set, and the separate training and test partitions of the MIT-BIH Noise Stress Test Database. (`fig0_flow.png`)
