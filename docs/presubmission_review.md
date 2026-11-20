# Pre-submission review: "Stress-testing wearable ECG atrial fibrillation classifiers with real recorded noise"

Target journal: Physiological Measurement (IOP). Review date: 2026-11-20. Three simulated reviewer perspectives, each
written as a critical referee would. Every point lists what was done about it. "Fixed" means the manuscript, supplement
or analysis changed; "Answered" means the existing text already handles it and the location is given; "Open" means it
needs the author.

## Reviewer 1: signal processing and measurement

**Summary judgement.** The noise model is well specified and the beat-detection analysis is now the paper's strongest
part. Main concerns were the realism of additive noise, the SNR definition and whether the CinC pseudo-reference is
valid.

| # | Issue | Severity | Resolution |
|---|---|---|---|
| 1.1 | Additive NSTDB noise on clean signals does not reproduce saturation, contact loss or noise that interacts with the signal. | Major | Answered and partly fixed: naturally noisy AFDB Holter recordings added (section 3.7); limitation states what additive noise cannot represent (4.1). |
| 1.2 | Power-based SNR differs from the QRS-amplitude scaling of the original NSTDB tool, so the −6 dB condition may not match the literature. | Major | Fixed: QRS-amplitude scaling and burst noise run as alternative generators (2.4, 3.2, table S10); conclusions unchanged. |
| 1.3 | Beat detection on CinC was measured against detections on the clean signal, not against annotations. | Major | Fixed: seven detectors scored against cardiologist annotations on the MIT-BIH Arrhythmia Database; pseudo-reference agrees with the expert reference within 0.004 F1 under heavy noise (3.4, table S11). |
| 1.4 | Results may be specific to the NeuroKit2 detector. | Major | Fixed: feature models refitted on Hamilton, Elgendi and Pan–Tompkins detections; same loss of specificity (3.5, table S12b). |
| 1.5 | Several detectors mark a point 50–65 ms after the annotated R peak; is the ±150 ms tolerance hiding errors? | Minor | Answered: constant offsets are reported (table S11) and do not change beat counts or RR intervals (supplementary methods). |
| 1.6 | Baseline wander: the CNN reads the unfiltered waveform, so "baseline wander is harmless" may not hold for it. | Major | Fixed: the text now reports that baseline wander left AUROC almost unchanged but lowered CNN sensitivity at the fixed threshold (0.849 to 0.696 internal; 0.863 to 0.766 PTB-XL), and the practical advice was corrected (3.2, 3.3, 4). |
| 1.7 | Noise was recorded from chest electrodes and applied to lead I and handheld recordings. | Minor | Answered in Limitations (4.1). |

## Reviewer 2: machine learning and validation

**Summary judgement.** Design and reporting are careful. Main concerns were post hoc hypotheses, resampling units,
multiplicity and whether the model comparison is fair.

| # | Issue | Severity | Resolution |
|---|---|---|---|
| 2.1 | Hypotheses were formed after seeing the internal results; testing them on the same data is circular. | Major | Fixed: internal primary comparisons use only the four fresh noise draws; table 1 labels every analysis as planned or post hoc; confirmatory claims rest on fresh draws, PTB-XL and AFDB (2.11, table 1, 4.1). |
| 2.2 | Bootstrap units: repeated measures of one recording across noise levels, seeds and draws must stay together. | Major | Answered: recordings (CinC) or patients (PTB-XL, MIT-BIH) are resampled with all their predictions; exactness of the call-rate estimator is stated (2.11). |
| 2.3 | Multiplicity across many conditions. | Major | Fixed: six designated primary comparisons, Holm-adjusted; everything else is labelled exploratory and interpreted by effect size (2.11, table 4). |
| 2.4 | Comparing sensitivity of two models at different thresholds may reflect threshold choice, not robustness. | Major | Fixed: the text shows both models start at almost the same clean operating point (3.3), and reports a sensitivity-target threshold (3.9). |
| 2.5 | How much of the variability comes from the noise segment, the training seed and the test sample? | Minor | Fixed: variance components reported; seed variability is largest (3.2, table S9, figure S3). |
| 2.6 | Leakage, duplicates and label definitions. | Major | Fixed: audit added (2.2, supplementary methods); duplicates within development set only; flutter exclusion changes metrics by < 0.01. |
| 2.7 | External AUROC may be inflated by an easy majority class. | Minor | Fixed: performance by rhythm statement added (3.1, table S8). |
| 2.8 | Only one CNN architecture; CNN extra draws limited to 0 and −6 dB; CNN ablation with three seeds. | Minor | Answered in Limitations (4.1) and Methods (2.4, 2.8). |
| 2.9 | Revision analyses ran under a different scikit-learn version. | Minor | Answered: stored models re-scored exactly; elastic-net setting disclosed (2.12, supplementary methods). |
| 2.10 | Claim that the CNN's failure is "mostly a threshold shift" (earlier draft). | Major | Fixed: noise-specific threshold recovers only part of sensitivity, AUROC falls, and regional noise locates the loss outside the QRS complexes (3.5, table S13). |

## Reviewer 3: clinical

**Summary judgement.** The clinical framing is now restrained. Main concerns were whether −6 dB is realistic, how
flutter is handled, and whether predictive values are presented responsibly.

| # | Issue | Severity | Resolution |
|---|---|---|---|
| 3.1 | How common are 0 and −6 dB conditions in real consumer recordings? | Major | Fixed: limitation added that the frequency is unknown and natural Holter noise was mostly milder, so the severest conditions are a stress test (4.1). |
| 3.2 | Atrial flutter is usually managed like AF; labelling it non-AF may distort results. | Minor | Answered: CinC labelling followed; flutter-only records reported separately and their exclusion changes metrics by < 0.01 (2.1, 3.1). |
| 3.3 | PPV depends on prevalence; test-set PPV misleads for screening. | Major | Fixed: hypothetical PPV and NPV at 0.5–10% prevalence, clearly labelled as illustrations (2.10, 3.9, table S18). |
| 3.4 | "Inconclusive" results: does abstention drop AF patients disproportionately, and does it differ by sex or age? | Major | Fixed: AF rejection ratio reported for seven measures; subgroup withholding by sex and age reported (3.8, table S16). |
| 3.5 | Which error is clinically worse depends on the setting; the paper should not decide it. | Minor | Fixed: Discussion states that the clinical judgement is outside the scope of a bench study (4). |
| 3.6 | No clinician involved. | Major | Open: the author plans to send the manuscript to cardiologists before submission (brief in docs/clinician_review_brief.md). Limitation states the absence (4.1). |
| 3.7 | Absolute language such as "costs nothing" or "can only be as robust as its detector". | Minor | Fixed: removed; claims are now conditional on the data shown. |

## Editorial and compliance checks

| Check | Status |
|---|---|
| Structured abstract (Objective, Approach, Main results, Significance), ≤ 250 words including headings | 248 words |
| Harvard references; every in-text citation listed and every listed reference cited | Checked by script after build |
| Double-anonymous version without author name, links or funding | Built by `--anonymous` |
| AI disclosure names every AI tool used | Unchanged |
| Novelty statement with comparison table | Introduction and table S5 |
| Data and code availability | All PhysioNet DOIs listed; code link present |

## Items the author must handle

1. **Code archive.** Done: release v1.0.6 with every revision script is archived at
   https://doi.org/10.5281/zenodo.24403832 and cited in the manuscript.
2. **Two references** could not be verified online and should be checked against the original sources: Elgendi et al
   (2010) and Engelse and Zeelenberg (1979). Moody and Mark (1983) follows PhysioNet's recommended citation for the
   AFDB.
3. **Clinical review** by cardiologists (see the brief); update the acknowledgements or author list afterwards.
