# TRIPOD+AI mapping

TRIPOD+AI (Collins et al., *BMJ* 2024;385:e078378) is the reporting guideline for studies that develop or evaluate
prediction models using regression or machine learning. The topic names below are paraphrased, so before submission,
copy the official checklist from tripod-statement.org, check the item numbering against it, and fill in the manuscript
page numbers.

| Section | Topic | Where / how addressed |
|---|---|---|
| Title | Identify as development + external evaluation of a prediction model; population; outcome | "…development and external validation…, single-lead AF detection…" (manuscript) |
| Abstract | Objective, data, models, validation, main results and conclusion (TRIPOD+AI for Abstracts items) | Manuscript abstract |
| Introduction | Clinical context: consumer-wearable AF screening; why noise robustness matters | Intro ¶1–2 |
| | Rationale for the models and the existing evidence (CinC 2017 literature) | Intro ¶3; four-class score for comparability (`fourclass_challenge.csv`) |
| | Health inequalities / fairness considerations | Discussion: PTB-XL age/sex subgroup performance (`subgroups_external.csv`) |
| Objectives | Development, internal + external evaluation, robustness, mechanism, augmentation, abstention; hypotheses H1–H4 | Intro final ¶; table 1 (planned vs post hoc) |
| Methods: data | Sources, dates, settings | Section 2.1 and supplementary methods (data versions): CinC 2017 (AliveCor, 2017), PTB-XL (Schiller devices, 1989–1996), NSTDB, MIT-BIH Arrhythmia Database, MIT-BIH AF Database |
| | Eligibility; exclusions | Binary task drops `~` class; PTB-XL `exclude_codes` (default none) |
| | Data preparation, pre-processing | `features.py` docstring; resampling in `data.py` |
| Outcome | Definition and how assessed | CinC: expert-reviewed v3 labels; PTB-XL: SCP code `AFIB` (cardiologist annotation) |
| | Blinding of outcome assessment | Labels were made independently of this study |
| Predictors | Definition, timing, measurement | Supplementary methods (feature list; `features.py`); raw signal for CNN |
| | Predictor assessment blind to outcome | Feature extraction never reads labels (`run_features`) |
| Sample size | Justification | Fixed public datasets: report events per split (`splits.json`), events per feature, and CI widths |
| Missing data | Handling | Features undefined when < 5 beats: median imputation + missingness indicators (LR), native NaN handling (LightGBM), median fill (rule) |
| Analysis | Model type, hyper-parameter tuning, internal validation | `models.py`; 5-fold CV on training only, AUPRC objective |
| | Model output (probabilities, thresholds) | Platt-calibrated probability; max-F1 threshold from out-of-fold training predictions |
| | Performance measures | Sensitivity, specificity, PPV, NPV, F1, AUROC, AUPRC, Brier, ECE, calibration slope/intercept |
| | Uncertainty | 5 seeds + 1,000 bootstrap replicates resampling recordings (CinC) or patients (PTB-XL, MIT-BIH); variance components (table S9); six primary comparisons Holm-adjusted (table 4); other comparisons exploratory |
| Class imbalance | Methods used and their effect on calibration | Class weights; Platt recalibration; calibration reported (`calibration.csv`, supplementary figure S5) |
| Fairness | Methods to assess | External performance by sex and age band, patient-level bootstrap (`subgroups_external.csv`); abstention by sex and age (table S16); no demographics in CinC 2017 |
| Training vs evaluation | Differences in data between development and evaluation | Device, lead, duration (9–60 s vs 10 s), population; discussed |
| Ethics | Approval | Secondary analysis of de-identified public data; cite each dataset's original ethics statement |
| Open science | Funding, conflicts, protocol, registration | Manuscript declarations; note no prospective registration |
| | Data sharing | Data availability statement (README) |
| | Code sharing | GitHub + Zenodo DOI; fixed seeds; `requirements.txt` |
| Patient involvement | PPI | State "none" if none |
| Results | Participant flow and characteristics | Supplementary table S4 (`table1_cohorts.csv`): counts, AF prevalence, PTB-XL age/sex; flow diagram in supplementary figure S1 (`fig0_flow.png`, made by `scripts/make_flow_diagram.py`) |
| | Model development details | `best_params` in each bundle; CV scores |
| | Full model specification | Code + archived model files |
| | Performance with CIs, internal and external | Tables 2–4; `main_internal.csv`, `main_external.csv`, `external_noise.csv` (table S7); natural noise `afdb_by_quality.csv` (table S14) |
| | Model updating | Augmentation analysis (`augmentation_effect.csv`) and ablation (`rev_augablation.csv`, table S15) |
| Discussion | Interpretation, including fairness | Discussion |
| | Limitations | Discussion, section 4.1 (Limitations) |
| | Usability, implementation | Operating points, confusion matrices, NPV, ≥90%-sensitivity threshold, hypothetical prevalence (tables S17–S18); quality-measure transfer (table S16) |
