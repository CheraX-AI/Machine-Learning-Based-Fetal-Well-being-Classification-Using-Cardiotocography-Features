# Processed Dataset — Group 24

This folder contains the **shared train / validation / test split** that all four team members must use.
Everyone must use exactly these files, so our results can be compared fairly.

Created on 29 Oct 2026 by `notebooks/02_data_preprocessing.ipynb` (seed = 42).

---

## 1. Files

| File | Rows | Patients | Normal (0) | Abnormal (1) | Use it for |
|---|---|---|---|---|---|
| `train.csv` | 14,540 (70.0%) | 8,742 | 6,719 | 7,821 | Fitting models, imputer, scaler, feature selection |
| `val.csv` | 3,115 (15.0%) | 1,904 | 1,439 | 1,676 | Tuning hyperparameters, choosing k, comparing models |
| `test.csv` | 3,114 (15.0%) | 1,960 | 1,440 | 1,674 | **Final evaluation only, once, together as a team** |
| `split_manifest.csv` | 20,769 | 12,606 | – | – | Shows which set every `record_id` belongs to |

Abnormal rate is **53.8%** in all three sets.

---

## 2. Columns (106 in each CSV)

| Column | Role |
|---|---|
| `record_id` | Recording ID. For traceability only. **Not a feature.** |
| `patient_id` | Pregnancy ID. Used for grouping only. **Not a feature.** |
| 103 feature columns (`F_*`, `U_*`, `FU_*`) | The predictors (X) |
| `figo_label` | The target (y): 0 = normal, 1 = abnormal |

Notes:
- The column `F_Med.1` from the raw file was renamed to **`F_MedAD`** (median absolute deviation of FHR).
- `NetFea_0`–`NetFea_4` and all clinical / delivery / neonatal columns were removed.
- There are **no missing values and no infinite values** in the 103 features.
- **No scaling, imputation or feature selection has been done.** These must be fitted on `train.csv` only, inside your model pipeline.

---

## 3. How the split was made

- All recordings of a patient are in **one set only** (patient-level split, to avoid data leakage).
  Many patients have 2–10 recordings, and 3,006 patients have both normal and abnormal recordings.
- `StratifiedGroupKFold` (20 parts, seed 42) was used: parts 0–2 → test, 3–5 → validation, 6–19 → train.
- Checks that passed: no patient in two sets, no rows lost (20,769 in = 20,769 out),
  both classes in every set, no duplicate rows, no identical feature rows across sets.

---

## 4. How to get the files

**Option A — pull from GitHub (recommended).** The 4 CSV files are in this folder in the repository:
```
git pull
```

**Option B — create them yourself** (only needed if you want to re-check the split):
1. Put `metadata.csv` in `data/raw/`.
2. Run all cells of `notebooks/02_data_preprocessing.ipynb`.
3. Compare your `split_manifest.csv` with the shared one — it must be identical.

### Check you have the correct files

The MD5 fingerprints of the correct files are:

| File | MD5 |
|---|---|
| `train.csv` | `97ea03cb89fecc8e3490529519385de0` |
| `val.csv` | `821efe5e0e70b11cd5ef2cfd2b6a5dca` |
| `test.csv` | `bc987f63bbcd4e03d7b7142f76858444` |
| `split_manifest.csv` | `d16c4c98886d6cbd407616b8c397c2bb` |

Check on Windows (PowerShell), from the project folder:
```powershell
Get-FileHash data\processed\*.csv -Algorithm MD5
```

If you created the files yourself with Option B and the MD5 is different (for example because of a different pandas version), check that the **row counts in Section 1 match** and that your `split_manifest.csv` gives every `record_id` the same split.

---

## 5. How to load the data in a notebook

```python
import sys
sys.path.append('..')

import pandas as pd
from config import config
from src.data_preprocessing import get_feature_columns

# Load the three sets
train_df = pd.read_csv(config.PROCESSED_TRAIN_FILE)
val_df = pd.read_csv(config.PROCESSED_VAL_FILE)

# Find the 103 feature columns (columns starting with F_, U_ or FU_)
feature_cols = get_feature_columns(train_df)

# Features (X) and target (y)
X_train = train_df[feature_cols]
y_train = train_df["figo_label"]

X_val = val_df[feature_cols]
y_val = val_df["figo_label"]

print("Train:", X_train.shape, " Validation:", X_val.shape)
```

Expected output:
```
Number of feature columns found: 103
Train: (14540, 103)  Validation: (3115, 103)
```

`test.csv` is loaded the same way (`config.PROCESSED_TEST_FILE`), but **only at the final evaluation step**.

---

## 6. Team rules

1. **Never edit these CSV files.** If something looks wrong, tell the team.
2. **Never use `record_id` or `patient_id` as features.** Use `feature_cols` from Section 5.
3. **Fit everything on `train.csv` only** — imputer, scaler, correlation filter, mutual information, Random Forest ranking.
4. **Tune and compare models on `val.csv`.**
5. **Do not look at `test.csv` results during development.** The test set is used **once**, at the end, for the final chosen model.
6. Use **seed 42** (`config.RANDOM_STATE`) everywhere.

---

## 7. Source and licence

These files are derived from the **JNU-CTG dataset**:

> J. Bai, "JNU-CTG: A large-scale continuous cardiotocography dataset for fetal well-being assessment," 2026.

JNU-CTG is released under the **Creative Commons Attribution 4.0 International licence (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

**Changes we made to the original `metadata.csv`:**
- Kept only `record_id`, `patient_id`, the 103 signal features and `figo_label`
  (removed clinical, delivery, neonatal and `NetFea_*` columns).
- Renamed `F_Med.1` to `F_MedAD`.
- Split the recordings into train / validation / test by patient (seed 42).

No feature values or labels were changed.

These files are for research and education only and must not be used to guide patient care.
