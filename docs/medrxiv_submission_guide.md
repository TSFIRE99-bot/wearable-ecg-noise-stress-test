# medRxiv submission guide

Submit at https://submit.medrxiv.org (create a free account with tejas.menneni@gmail.com first).
medRxiv screens every preprint (usually a few days) for scope, completeness and the required statements;
it does not peer review. Everything below is already in the manuscript, so the form answers just repeat it.

## Before you start
- [x] Zenodo DOI added: https://doi.org/10.5281/zenodo.24042753.
- [x] Funding and competing-interests statements confirmed (none and none).
- [x] Ethics statement finalized.
- [x] Files built. To rebuild after any edit: `python scripts/build_manuscript_docx.py`, `python scripts/build_supplement_docx.py`,
      then `powershell -ExecutionPolicy Bypass -File scripts/docx_to_pdf.ps1`.

## Files to upload
| medRxiv field | File |
|---|---|
| Manuscript (PDF or Word, figures and tables included) | `paper/submission/manuscript.pdf` or `manuscript.docx` |
| Supplementary material (separate file, required here) | `paper/submission/supplementary.pdf` (Fig. S1, Tables S1–S3) |

## Form answers
| Field | Answer |
|---|---|
| Article type | New Results; category: **Study on diagnostic tools or medical equipment** |
| Subject area | Cardiovascular Medicine (alternative: Health Informatics) |
| Title | Stress-testing wearable ECG atrial fibrillation classifiers with real recorded noise |
| Author | Tejas Menneni, Columbus State Community College, Columbus, Ohio, USA (corresponding; tejas.menneni@gmail.com; add ORCID if you have one) |
| Abstract | Paste the abstract from the manuscript as plain text |
| Competing interests | "The author has declared no competing interest." |
| Funding statement | "This study did not receive any funding." |
| Clinical trial | No / not applicable |
| Ethics: IRB/oversight body | Choose the option for studies that use **only openly available human data**, and give the sources: https://doi.org/10.13026/D3D0-0F49 (CinC 2017), https://doi.org/10.13026/KFZX-AW45 (PTB-XL), https://doi.org/10.13026/C2HS3T (MIT-BIH NSTDB) |
| Consent | Not applicable: de-identified, openly available data |
| Reporting guideline (EQUATOR) | Yes: TRIPOD+AI (Collins et al., BMJ 2024) |
| Data availability | All data are openly available from PhysioNet (the three DOIs above); code at https://github.com/TSFIRE99-bot/wearable-ecg-noise-stress-test, archived at https://doi.org/10.5281/zenodo.24042753 |
| Generative AI | Disclose as in the manuscript: the Anaconda Prompt AI assistant and Claude (Anthropic) were used for code and drafting |
| License | CC-BY 4.0 is the usual choice and allows later journal submission |

## After posting
- medRxiv gives a DOI (10.1101/...). Add it to the README and `CITATION.cff`, and mention it in the Journal of
  Biomedical Informatics cover letter. Elsevier journals generally accept submissions that were preprinted, but check
  JBI's current policy page before submitting.
- If medRxiv declines on scope, the usual alternative is arXiv (eess.SP or cs.LG).
