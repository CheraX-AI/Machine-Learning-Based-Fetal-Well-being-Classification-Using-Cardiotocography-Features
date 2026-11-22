"""
Run all seven classifiers on the patient-grouped split, VALIDATION SET ONLY.

Order of steps:
  1. Load train.csv and val.csv (test.csv is NOT opened).
  2. Pick the 103 signal features (F_, U_, FU_ columns).
  3. Build the selected feature sets from TRAINING data only.
  4. For each feature set and each model:
       fit imputer (+ scaler for LR, KNN, SVM) on training data only,
       fit the model on training data, score on validation data.
  5. Save every score to seven_models_validation.csv.

Feature selection (training data only):
  - absolute Pearson correlation filter at 0.90 (columns taken alphabetically)
  - rank remaining features by mutual information and by Random Forest importance
  - keep the INTERSECTION of the top-k of both rankings, k = 10, 20, 30, 50
Models use fixed, mostly default settings (no tuning) so the comparison is consistent.

Usage: python run_seven_models.py <project_root>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
import lightgbm
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

SEED = 42
PROJECT_ROOT = Path(sys.argv[1])
OUT_DIR = PROJECT_ROOT / "presentation" / "supporting"
PREFIXES = ("F_", "U_", "FU_")
TARGET = "figo_label"

# 1. Load training and validation data only (the test file is never opened)
train_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val.csv")

# 2. Feature columns
all_features = []
for col in train_df.columns:
    if col.startswith(PREFIXES):
        all_features.append(col)
if len(all_features) != 103:
    raise ValueError("Expected 103 features, found " + str(len(all_features)))

y_train = train_df[TARGET].astype(int)
y_val = val_df[TARGET].astype(int)
print("Train", train_df.shape, "Val", val_df.shape, "features", len(all_features))
print("NaN in train features:", int(train_df[all_features].isna().sum().sum()),
      " inf:", int(np.isinf(train_df[all_features].to_numpy()).sum()))

# Fill missing values with TRAINING medians (only used for the selection step)
train_features = train_df[all_features].replace([np.inf, -np.inf], np.nan)
train_medians = train_features.median()
train_filled = train_features.fillna(train_medians)


def correlation_filter(df, threshold):
    """Keep a column only if |corr| with every already-kept column is <= threshold."""
    corr = df.corr().abs()
    kept = []
    for col in sorted(df.columns):
        keep_it = True
        for other in kept:
            if corr.loc[col, other] > threshold:
                keep_it = False
                break
        if keep_it:
            kept.append(col)
    return kept


def ranked_names(names, scores):
    """Sort by score (high to low); ties are broken by feature name."""
    pairs = []
    for i in range(len(names)):
        pairs.append((-float(scores[i]), names[i]))
    pairs.sort()
    result = []
    for pair in pairs:
        result.append(pair[1])
    return result


# 3. Selection rankings (TRAINING data only)
kept_after_corr = correlation_filter(train_filled, 0.90)
print("Features after correlation filter (0.90):", len(kept_after_corr))

mi_scores = mutual_info_classif(train_filled[kept_after_corr], y_train, random_state=SEED)
rf_rank_model = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf_rank_model.fit(train_filled[kept_after_corr], y_train)
rf_scores = rf_rank_model.feature_importances_

mi_rank = ranked_names(kept_after_corr, mi_scores)
rf_rank = ranked_names(kept_after_corr, rf_scores)

feature_sets = {"all_103": all_features}
for k in (10, 20, 30, 50):
    top_mi = set(mi_rank[:k])
    top_rf = set(rf_rank[:k])
    common = top_mi.intersection(top_rf)
    feature_sets["top_" + str(k)] = sorted(common)
    print("k =", k, "-> intersection size", len(common))

selection_info = {
    "after_corr_filter": len(kept_after_corr),
    "mi_rank_top20": mi_rank[:20],
    "rf_rank_top20": rf_rank[:20],
    "feature_sets": feature_sets,
}
(OUT_DIR / "feature_selection_info.json").write_text(json.dumps(selection_info, indent=1))

# 4. Models
models = {
    "Logistic Regression": (True, LogisticRegression(max_iter=1000, random_state=SEED)),
    "KNN": (True, KNeighborsClassifier(n_neighbors=5)),
    "SVM": (True, SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
    "Decision Tree": (False, DecisionTreeClassifier(random_state=SEED)),
    "Random Forest": (False, RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
    "XGBoost": (False, XGBClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, eval_metric="logloss")),
    "LightGBM": (False, LGBMClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, verbose=-1)),
}

rows = []
for set_name, cols in feature_sets.items():
    X_train = train_df[cols].replace([np.inf, -np.inf], np.nan)
    X_val = val_df[cols].replace([np.inf, -np.inf], np.nan)
    for model_name, (needs_scaling, estimator) in models.items():
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if needs_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        pipe = Pipeline(steps)

        start = time.time()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        if model_name == "SVM":
            score = pipe.decision_function(X_val)
        else:
            score = pipe.predict_proba(X_val)[:, 1]
        seconds = time.time() - start

        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        row = {
            "feature_set": set_name,
            "n_features": len(cols),
            "model": model_name,
            "accuracy": accuracy_score(y_val, pred),
            "precision": precision_score(y_val, pred, pos_label=1),
            "recall": recall_score(y_val, pred, pos_label=1),
            "f1": f1_score(y_val, pred, pos_label=1),
            "roc_auc": roc_auc_score(y_val, score),
            "balanced_accuracy": balanced_accuracy_score(y_val, pred),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "seconds": round(seconds, 1),
        }
        rows.append(row)
        print(set_name, model_name, "F1 %.4f AUC %.4f (%.1fs)" % (row["f1"], row["roc_auc"], seconds), flush=True)

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / "seven_models_validation.csv", index=False)

versions = {
    "python": sys.version.split()[0],
    "sklearn": sklearn.__version__,
    "xgboost": xgboost.__version__,
    "lightgbm": lightgbm.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "seed": SEED,
}
(OUT_DIR / "package_versions.json").write_text(json.dumps(versions, indent=1))

# Sanity check for the Logistic Regression result: strongest scaled coefficients
lr_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
])
lr_pipe.fit(train_df[all_features].replace([np.inf, -np.inf], np.nan), y_train)
coefs = lr_pipe.named_steps["model"].coef_[0]
order = np.argsort(-np.abs(coefs))[:10]
print("Top 10 |coef| features (scaled LR):")
for i in order:
    print("  ", all_features[i], round(float(coefs[i]), 3))
print("DONE")
