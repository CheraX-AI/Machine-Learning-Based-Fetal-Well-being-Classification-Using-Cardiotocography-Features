# Machine-Learning-Based-Fetal-Well-being-Classification-Using-Cardiotocography-Features

EC5203 Machine Learning mini project, Group 24, University of Ruhuna.

Binary classification of the expert FIGO label (`figo_label`: 0 = normal, 1 = abnormal) from the
103 signal-only CTG features of the JNU-CTG dataset, using classical supervised ML (no neural networks),
a patient-grouped train / validation / test split, feature selection, tuning and evaluation.

For academic research and education only. Not for clinical use.

## Project structure

```
├── data/
│   ├── raw/metadata.csv            # original JNU-CTG table (never changed, not committed)
│   └── processed/                  # shared patient-grouped split (train / val / test + manifest)
├── notebooks/                      # run in order 01 → 04
│   ├── 01_data_understanding.ipynb # look at the raw table (nothing is changed)
│   ├── 02_preprocessing.ipynb      # 103 features, checks, 70/15/15 split by patient, outlier tests
│   ├── 03_feature_selection.ipynb  # correlation analysis (+ planned MI / RF top-k)
│   └── 04_model_training.ipynb     # LR, RF, XGBoost, LightGBM, RF tuning (validation set only)
├── results/                        # saved metric tables (one flat folder)
├── archive/                        # old notebooks, src/, config/, tests/ (kept for reference)
├── requirements.txt
└── README.md
```

## How to run

1. Put `metadata.csv` in `data/raw/`.
2. Install the packages: `pip install -r requirements.txt`
3. Open the notebooks folder and run the notebooks **in order** (01 → 04).

Every notebook starts with a **Settings** cell (paths, column names, random seed 42) and finds the project
folder automatically, so it works on every team member's computer.

## Rules

- `record_id` and `patient_id` are never features; all recordings of one patient stay in one set.
- Anything learned from data (scaler, imputer, feature selection, thresholds) is fitted on the **training set only**.
- Models are compared on the **validation set**. The **test set is used once**, at the final evaluation.
- The files in `data/processed/` are shared by the team and are never overwritten.
