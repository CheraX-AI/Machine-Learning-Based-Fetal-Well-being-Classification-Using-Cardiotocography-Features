"""
Make ROC-curve points and feature-scale statistics for the presentation.

Uses TRAINING data to fit and VALIDATION data to score (test file is never opened).
Same models and settings as run_seven_models.py, all 103 features.

Usage: python run_roc_curves.py <project_root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.metrics import roc_curve, roc_auc_score

SEED = 42
ROOT = Path(sys.argv[1])
OUT_DIR = ROOT / "presentation" / "supporting"

train_df = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
val_df = pd.read_csv(ROOT / "data" / "processed" / "val.csv")

features = []
for col in train_df.columns:
    if col.startswith(("F_", "U_", "FU_")):
        features.append(col)

y_train = train_df["figo_label"].astype(int)
y_val = val_df["figo_label"].astype(int)
X_train = train_df[features].replace([np.inf, -np.inf], np.nan)
X_val = val_df[features].replace([np.inf, -np.inf], np.nan)

# Feature-scale statistics (training data only)
stds = X_train.std()
nonzero = stds[stds > 0]
scale_info = {
    "largest_std_feature": str(nonzero.idxmax()),
    "largest_std": float(nonzero.max()),
    "smallest_std_feature": str(nonzero.idxmin()),
    "smallest_std": float(nonzero.min()),
    "zero_std_features": int((stds == 0).sum()),
    "prefix_counts": {
        "F_": sum(1 for c in features if c.startswith("F_")),
        "U_": sum(1 for c in features if c.startswith("U_")),
        "FU_": sum(1 for c in features if c.startswith("FU_")),
    },
}
print(scale_info)

models = {
    "Logistic Regression": (True, LogisticRegression(max_iter=1000, random_state=SEED)),
    "KNN": (True, KNeighborsClassifier(n_neighbors=5)),
    "SVM": (True, SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
    "Decision Tree": (False, DecisionTreeClassifier(random_state=SEED)),
    "Random Forest": (False, RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
    "XGBoost": (False, XGBClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, eval_metric="logloss")),
    "LightGBM": (False, LGBMClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, verbose=-1)),
}

curves = {}
for name, (needs_scaling, estimator) in models.items():
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    if name == "SVM":
        score = pipe.decision_function(X_val)
    else:
        score = pipe.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, score)
    # keep at most about 90 evenly spaced points so the file stays small
    keep = np.unique(np.linspace(0, len(fpr) - 1, 90).astype(int))
    curves[name] = {
        "fpr": [round(float(x), 4) for x in fpr[keep]],
        "tpr": [round(float(x), 4) for x in tpr[keep]],
        "auc": round(float(roc_auc_score(y_val, score)), 4),
    }
    print(name, "AUC", curves[name]["auc"], "points", len(keep), flush=True)

(OUT_DIR / "roc_curves.json").write_text(json.dumps({"curves": curves, "scale": scale_info}))
print("DONE")
