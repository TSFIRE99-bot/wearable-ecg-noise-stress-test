# Data, label and leakage audit

Produced by `scripts/audit_data.py`; nothing in this audit changes a model or a result.

## 1. Duplicate recordings in CinC 2017

- Identical signals: 2 groups (4 recordings); groups crossing the train/test split: 0.
- Identical first 3 s (near-duplicates): 2 groups (4 recordings); crossing the split: 0 groups (0 recordings).
- CinC 2017 (AliveCor, 300 Hz) and PTB-XL (clinical 12-lead, 500 Hz) come from different devices and institutions, so no recording can appear in both.

## 2. Leakage paths in the stored models

- binary\augmented\cnn_seed0.pkl: 1980 calibration rows, test ids among them: 0
- binary\augmented\cnn_seed1.pkl: 1980 calibration rows, test ids among them: 0
- binary\augmented\cnn_seed2.pkl: 1980 calibration rows, test ids among them: 0
- binary\augmented\cnn_seed3.pkl: 1980 calibration rows, test ids among them: 0
- binary\augmented\cnn_seed4.pkl: 1980 calibration rows, test ids among them: 0
- binary\augmented\lgbm_seed0.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\lgbm_seed1.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\lgbm_seed2.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\lgbm_seed3.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\lgbm_seed4.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\logreg_seed0.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\logreg_seed1.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\logreg_seed2.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\logreg_seed3.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\logreg_seed4.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\rule_seed0.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\rule_seed1.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\rule_seed2.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\rule_seed3.pkl: 13190 calibration rows, test ids among them: 0
- binary\augmented\rule_seed4.pkl: 13190 calibration rows, test ids among them: 0
- binary\standard\cnn_seed0.pkl: 990 calibration rows, test ids among them: 0
- binary\standard\cnn_seed1.pkl: 990 calibration rows, test ids among them: 0
- binary\standard\cnn_seed2.pkl: 990 calibration rows, test ids among them: 0
- binary\standard\cnn_seed3.pkl: 990 calibration rows, test ids among them: 0
- binary\standard\cnn_seed4.pkl: 990 calibration rows, test ids among them: 0
- binary\standard\lgbm_seed0.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\lgbm_seed1.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\lgbm_seed2.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\lgbm_seed3.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\lgbm_seed4.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\logreg_seed0.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\logreg_seed1.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\logreg_seed2.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\logreg_seed3.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\logreg_seed4.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\rule_seed0.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\rule_seed1.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\rule_seed2.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\rule_seed3.pkl: 6595 calibration rows, test ids among them: 0
- binary\standard\rule_seed4.pkl: 6595 calibration rows, test ids among them: 0
- fourclass\standard\cnn_seed0.pkl: 1024 calibration rows, test ids among them: 0
- fourclass\standard\lgbm_seed0.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\lgbm_seed1.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\lgbm_seed2.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\lgbm_seed3.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\lgbm_seed4.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\logreg_seed0.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\logreg_seed1.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\logreg_seed2.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\logreg_seed3.pkl: 6822 calibration rows, test ids among them: 0
- fourclass\standard\logreg_seed4.pkl: 6822 calibration rows, test ids among them: 0
- feature-model augmentation copies: 6595, made from test records: 0

Other checks, by construction in the code:
- The test split is stored with a SHA-256 hash and re-verified on every load; training code raises an error if a test identifier reaches it.
- Logistic regression imputes and scales inside the cross-validated pipeline (training folds only); LightGBM uses no population statistics; the rule's missing-value fill is the training median.
- The CNN z-scores each 9 s window on its own (no population statistics) and draws its inner validation split by recording, so windows of one recording never sit on both sides.
- Thresholds and Platt calibrators are fitted on the calibration rows listed above, all from training records.

## 3. Noise separation

- bw: samples 0-270832 for training augmentation, 270833-541666 for test corruption; ma: samples 0-270832 for training augmentation, 270833-541666 for test corruption; em: samples 0-270832 for training augmentation, 270833-541666 for test corruption.

## 4. PTB-XL labels

- AF (statement AFIB): 1514 of 21799 records (6.9%). AFIB likelihood values: {0.0: 1466, 100.0: 48} (PTB-XL gives rhythm statements likelihood 0 by convention, so every AFIB statement was counted).
- AFIB and AFLT both coded: 17 records (labelled AF). AFLT without AFIB: 56 records (labelled non-AF, as atrial flutter is 'other rhythm' in CinC 2017).

Share of recordings called AF (mean over five seeds), standard training. Categories other than AFIB exclude records that also carry AFIB:

|                |   ('external__clean', 'cnn') |   ('external__clean', 'lgbm') |   ('external__clean', 'logreg') |   ('external__clean', 'rule') |   ('external__em__-6dB__r0', 'cnn') |   ('external__em__-6dB__r0', 'lgbm') |   ('external__em__-6dB__r0', 'logreg') |   ('external__em__-6dB__r0', 'rule') |
|:---------------|-----------------------------:|------------------------------:|--------------------------------:|------------------------------:|------------------------------------:|-------------------------------------:|---------------------------------------:|-------------------------------------:|
| ('AFIB', 1514) |                        0.863 |                         0.842 |                           0.807 |                         0.666 |                               0.497 |                                0.725 |                                  0.693 |                                0.84  |
| ('AFLT', 56)   |                        0.282 |                         0.186 |                           0.161 |                         0.161 |                               0.436 |                                0.168 |                                  0.204 |                                0.411 |
| ('BIGU', 75)   |                        0.141 |                         0.056 |                           0.125 |                         0.973 |                               0.229 |                                0.44  |                                  0.395 |                                0.987 |
| ('PAC', 395)   |                        0.124 |                         0.105 |                           0.111 |                         0.585 |                               0.137 |                                0.341 |                                  0.327 |                                0.803 |
| ('PACE', 290)  |                        0.122 |                         0.049 |                           0.044 |                         0.2   |                               0.041 |                                0.197 |                                  0.186 |                                0.755 |
| ('PSVT', 23)   |                        0.043 |                         0.009 |                           0.043 |                         0.13  |                               0.217 |                                0.061 |                                  0.139 |                                0.435 |
| ('PVC', 938)   |                        0.072 |                         0.058 |                           0.07  |                         0.738 |                               0.077 |                                0.293 |                                  0.289 |                                0.88  |
| ('SARRH', 772) |                        0.039 |                         0.023 |                           0.021 |                         0.209 |                               0.044 |                                0.256 |                                  0.23  |                                0.72  |
| ('SBRAD', 637) |                        0.012 |                         0.002 |                           0.003 |                         0.248 |                               0.026 |                                0.203 |                                  0.281 |                                0.917 |
| ('SR', 16748)  |                        0.007 |                         0.004 |                           0.005 |                         0.064 |                               0.009 |                                0.129 |                                  0.135 |                                0.613 |
| ('STACH', 826) |                        0.038 |                         0.032 |                           0.035 |                         0.093 |                               0.055 |                                0.074 |                                  0.091 |                                0.293 |
| ('SVARR', 157) |                        0.048 |                         0.032 |                           0.025 |                         0.159 |                               0.037 |                                0.149 |                                  0.153 |                                0.662 |
| ('SVTAC', 26)  |                        0.115 |                         0.085 |                           0.077 |                         0.231 |                               0.292 |                                0.092 |                                  0.192 |                                0.769 |
| ('TRIGU', 20)  |                        0.22  |                         0.08  |                           0.22  |                         0.95  |                               0.27  |                                0.29  |                                  0.42  |                                1     |

## 5. Label sensitivity: atrial flutter excluded

| variant   | model   | condition              | set                     |     n |   auroc |   sensitivity |   specificity |   ppv |
|:----------|:--------|:-----------------------|:------------------------|------:|--------:|--------------:|--------------:|------:|
| standard  | rule    | external__clean        | all records             | 21799 |   0.926 |         0.666 |         0.912 | 0.36  |
| standard  | rule    | external__clean        | atrial flutter excluded | 21726 |   0.927 |         0.669 |         0.912 | 0.36  |
| standard  | rule    | external__em__-6dB__r0 | all records             | 21799 |   0.633 |         0.84  |         0.38  | 0.092 |
| standard  | rule    | external__em__-6dB__r0 | atrial flutter excluded | 21726 |   0.635 |         0.843 |         0.379 | 0.091 |
| standard  | logreg  | external__clean        | all records             | 21799 |   0.987 |         0.807 |         0.989 | 0.852 |
| standard  | logreg  | external__clean        | atrial flutter excluded | 21726 |   0.988 |         0.809 |         0.99  | 0.857 |
| standard  | logreg  | external__em__-6dB__r0 | all records             | 21799 |   0.863 |         0.693 |         0.853 | 0.27  |
| standard  | logreg  | external__em__-6dB__r0 | atrial flutter excluded | 21726 |   0.864 |         0.694 |         0.853 | 0.269 |
| standard  | lgbm    | external__clean        | all records             | 21799 |   0.99  |         0.842 |         0.99  | 0.866 |
| standard  | lgbm    | external__clean        | atrial flutter excluded | 21726 |   0.99  |         0.845 |         0.991 | 0.871 |
| standard  | lgbm    | external__em__-6dB__r0 | all records             | 21799 |   0.893 |         0.725 |         0.861 | 0.283 |
| standard  | lgbm    | external__em__-6dB__r0 | atrial flutter excluded | 21726 |   0.894 |         0.728 |         0.861 | 0.282 |
| standard  | cnn     | external__clean        | all records             | 21799 |   0.987 |         0.863 |         0.984 | 0.804 |
| standard  | cnn     | external__clean        | atrial flutter excluded | 21726 |   0.988 |         0.863 |         0.985 | 0.81  |
| standard  | cnn     | external__em__-6dB__r0 | all records             | 21799 |   0.936 |         0.497 |         0.983 | 0.7   |
| standard  | cnn     | external__em__-6dB__r0 | atrial flutter excluded | 21726 |   0.937 |         0.496 |         0.984 | 0.713 |
