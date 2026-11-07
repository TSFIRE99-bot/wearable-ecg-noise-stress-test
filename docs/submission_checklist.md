# Submission checklist (medRxiv preprint → Journal of Biomedical Informatics)

## Before posting anywhere
- [x] Author set to Tejas Menneni, Columbus State Community College (manuscript, `CITATION.cff`, `.zenodo.json`, `LICENSE`).
- [ ] Add the corresponding email to the manuscript; optionally create an ORCID (free) and add it to `CITATION.cff`.
- [ ] Review the funding, competing-interests and generative-AI declarations (drafted; marked [Confirm.]).
- [ ] Verify every reference against its source (entries marked *[verify]*: Zhao 2018; the CinC 2017 winning score ≈ 0.83).
- [ ] Add 2–4 citations of prior ECG-robustness / noise-stress work in the Discussion placeholder.
- [ ] Check the abstract word limit of the target journal (the current abstract is 255 words including section labels; trim if the limit is 250).
- [ ] Check the ethics wording against your institution's policy.
- [ ] Draw the participant flow diagram (8,528 → split → binary exclusions; PTB-XL counts).
- [ ] Re-read the limitations (9 items) and keep all of them.

## Code release (needed for the DOI cited in the paper)
- [x] Public GitHub repository created and pushed: https://github.com/TSFIRE99-bot/wearable-ecg-noise-stress-test
      (raw data, features and model files are excluded; results tables, figures, the split and the test-lock audit trail are included).
- [ ] Tag a release (e.g. `v1.0.0`) and connect the repository to Zenodo to mint a DOI; put the DOI into the manuscript,
      README and `CITATION.cff`.
- [ ] Optionally archive the 51 frozen model files (`outputs/models/`, about 110 MB) on Zenodo alongside the code.

## Figures and tables for the manuscript
| Manuscript | File |
|---|---|
| Table 1 | `outputs/tables/table1_cohorts.csv` |
| Table 2 | `outputs/tables/main_internal.csv`, `main_external.csv` |
| Table 3 | `outputs/tables/noise_degradation.csv` (select SNR 0 and −6 dB) |
| Table 4 | `outputs/tables/failure_modes.csv` |
| Table 5 | `outputs/tables/abstention_fixed_threshold.csv` |
| Fig. 1 | `outputs/figures/fig1_noise_auroc.png` (F1/AUPRC versions as supplementary) |
| Fig. 2 | `outputs/figures/fig2_beat_detection.png` |
| Fig. 3 | `outputs/figures/fig3_augmentation_auroc.png` |
| Fig. 4 | `outputs/figures/fig4_abstention.png` |
| Fig. 5 | `outputs/figures/fig5_calibration.png` |
| Fig. 6 | `outputs/figures/fig6_shap.png` |
| Suppl. S1 | `fourclass_challenge.csv` · S2 `subgroups_external.csv` · S3 `paired_tests.csv`, `augmentation_effect.csv` · S4 TRIPOD+AI (`docs/TRIPOD_AI_checklist.md`) |

## medRxiv specifics
- Subject area: Cardiovascular Medicine (or Health Informatics).
- Ethics statement: secondary analysis of de-identified, openly available PhysioNet data.
- Data availability: the three PhysioNet DOIs (already in the manuscript).
- Upload the TRIPOD+AI checklist as a supplementary file.

## Optional strengthening (reviewers may ask)
- Sensitivity analysis with the post-challenge labels (`data.cinc2017.reference: REFERENCE-v3.csv`).
- PTB-XL excluding atrial flutter (`data.ptbxl.exclude_codes: [AFLT]`).
- Noise stress test on PTB-XL too (`evaluation.external_noise: true`; adds ~1–2 h of features on 8 cores).
- More than one noise realization per recording (`noise.test_realizations: 3`).
