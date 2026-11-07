# medRxiv submission guide

Submit at https://submit.medrxiv.org (create a free account with tejas.menneni@gmail.com first).
medRxiv screens every preprint (usually a few days) for scope, completeness and the required statements;
it does not peer review. Everything below is already in the manuscript, so the form answers just repeat it.

## Before you start
- [ ] Zenodo DOI added (switch the repository on at zenodo.org → GitHub, then a new release; see `docs/submission_checklist.md`).
- [ ] Funding and competing-interests statements confirmed (remove "[Confirm.]" in `paper/manuscript.md`).
- [ ] Ethics sentence confirmed (remove the bracketed note).
- [ ] Rebuild the files: `python scripts/build_manuscript_docx.py`, then open `paper/submission/manuscript.docx` in Word and
      save it as PDF (or upload the .docx directly).

## Files to upload
| medRxiv field | File |
|---|---|
| Manuscript (PDF or Word, figures and tables included) | `paper/submission/manuscript.pdf` or `manuscript.docx` |
| Supplementary material (optional) | `docs/TRIPOD_AI_checklist.md` saved as PDF |

## Form answers
| Field | Answer |
|---|---|
| Article type | New Results |
| Subject area | Cardiovascular Medicine (alternative: Health Informatics) |
| Title | A real-noise stress-testing framework for wearable ECG classifiers: development and external validation with atrial fibrillation detection as a worked example |
| Author | Tejas Menneni, Columbus State Community College, Columbus, Ohio, USA (corresponding; tejas.menneni@gmail.com; add ORCID if you have one) |
| Abstract | Paste the abstract from the manuscript as plain text |
| Competing interests | "The author has declared no competing interest." (only if true) |
| Funding statement | "This study did not receive any funding." (only if true) |
| Clinical trial | No / not applicable |
| Ethics: IRB/oversight body | Choose the option for studies that use **only openly available human data**, and give the sources: https://doi.org/10.13026/D3D0-0F49 (CinC 2017), https://doi.org/10.13026/KFZX-AW45 (PTB-XL), https://doi.org/10.13026/C2HS3T (MIT-BIH NSTDB) |
| Consent | Not applicable: de-identified, openly available data |
| Reporting guideline (EQUATOR) | Yes: TRIPOD+AI (Collins et al., BMJ 2024) |
| Data availability | All data are openly available from PhysioNet (the three DOIs above); code at https://github.com/TSFIRE99-bot/wearable-ecg-noise-stress-test (plus the Zenodo DOI) |
| Generative AI | Disclose as in the manuscript: the Anaconda Prompt AI assistant and Claude (Anthropic) were used for code and drafting |
| License | CC-BY 4.0 is the usual choice and allows later journal submission |

## After posting
- medRxiv gives a DOI (10.1101/...). Add it to the README and `CITATION.cff`, and mention it in the Journal of
  Biomedical Informatics cover letter. Elsevier journals generally accept submissions that were preprinted, but check
  JBI's current policy page before submitting.
- If medRxiv declines on scope, the usual alternative is arXiv (eess.SP or cs.LG).
