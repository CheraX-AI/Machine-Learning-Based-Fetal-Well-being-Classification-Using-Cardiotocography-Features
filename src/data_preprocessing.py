"""
Data preprocessing for the JNU-CTG project.

This file prepares the processed dataset in these steps:
  1. Find the 103 feature columns.
  2. Keep only the ID columns, the feature columns and the target column.
  3. Check that all features are numbers and the target is only 0 or 1.
  4. Change +inf / -inf values to NaN (missing).
  5. Split the data into train (70%), validation (15%) and test (15%) by patient.
  6. Check the split: no patient in two sets, no rows lost, both classes
     in every set, and no identical feature rows crossing between sets.
  7. Save the three sets and a split manifest as CSV files in data/processed/.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

TARGET_COL = "figo_label"
GROUP_COL = "patient_id"
ID_COLS = ["record_id", "patient_id"]
NUMBER_OF_FEATURES = 103

# All signal features start with one of these:
#   "F_"  = fetal heart rate (FHR) features
#   "U_"  = uterine contraction (UC) features
#   "FU_" = FHR-UC relationship features
# Clinical, delivery, neonatal and NetFea_* columns do not start with these,
# so they are left out automatically.
FEATURE_PREFIXES = ["F_", "U_", "FU_"]


def get_feature_columns(df):
    """Return a list with the names of the 103 signal feature columns."""
    feature_cols = []
    for col in df.columns:
        for prefix in FEATURE_PREFIXES:
            if col.startswith(prefix):
                feature_cols.append(col)
                break

    print("Number of feature columns found:", len(feature_cols))

    # Stop if we did not get exactly 103 features
    if len(feature_cols) != NUMBER_OF_FEATURES:
        raise ValueError(
            "Expected 103 features but found " + str(len(feature_cols)) + "."
        )

    return feature_cols


def select_columns(df, feature_cols):
    """Keep only the ID columns, the feature columns and the target column."""
    columns_to_keep = ID_COLS + feature_cols + [TARGET_COL]
    new_df = df[columns_to_keep].copy()

    print("Shape after selecting columns:", new_df.shape)
    return new_df


def check_features_are_numeric(df, feature_cols):
    """Stop with an error if any feature column contains text instead of numbers."""
    text_columns = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            text_columns.append(col)

    if len(text_columns) > 0:
        print("These feature columns are not numeric:", text_columns)
        raise ValueError("Some feature columns are not numeric. Inspect them before continuing.")

    print("Check passed: all feature columns are numeric.")


def check_target_values(df):
    """Stop with an error if figo_label has missing values or values other than 0 and 1."""
    print("figo_label counts:")
    print(df[TARGET_COL].value_counts(dropna=False))

    missing_labels = df[TARGET_COL].isna().sum()
    if missing_labels > 0:
        raise ValueError(str(missing_labels) + " rows have no figo_label.")

    for value in df[TARGET_COL].unique():
        if value not in [0, 1]:
            raise ValueError("figo_label has an unexpected value: " + str(value))

    print("Check passed: figo_label only contains 0 and 1.")


def replace_inf_with_nan(df, feature_cols):
    """
    Change +inf and -inf values to NaN.

    The median imputer used later cannot handle infinity,
    but it can fill NaN values.
    """
    new_df = df.copy()

    # Count how many infinite values there are before replacing them
    inf_count = 0
    for col in feature_cols:
        inf_count = inf_count + np.isinf(new_df[col]).sum()

    # Replace them with NaN
    for col in feature_cols:
        new_df[col] = new_df[col].replace([np.inf, -np.inf], np.nan)

    print("Infinite values changed to NaN:", inf_count)
    return new_df


def split_by_patient(df, seed=42):
    """
    Split the data into train (70%), validation (15%) and test (15%).

    All recordings of one patient go into the same set.

    How it works:
      - StratifiedGroupKFold cuts the data into 20 parts (about 5% each).
      - Every patient stays inside one part, and each part keeps a
        similar normal / abnormal balance.
      - Parts 0 to 2  -> test        (3 parts  x 5% = 15%)
      - Parts 3 to 5  -> validation  (3 parts  x 5% = 15%)
      - Parts 6 to 19 -> train       (14 parts x 5% = 70%)
    """
    # Make a new copy with row numbers 0, 1, 2, ... so we can label each row
    df = df.reset_index(drop=True)
    df["part"] = -1

    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    labels = df[TARGET_COL]
    patients = df[GROUP_COL]

    # Give every row the number of the part it belongs to
    part_number = 0
    for other_rows, rows_in_this_part in splitter.split(df, labels, patients):
        df.loc[rows_in_this_part, "part"] = part_number
        part_number = part_number + 1

    # Make the three sets from the part numbers
    test_df = df[df["part"] <= 2]
    val_df = df[(df["part"] >= 3) & (df["part"] <= 5)]
    train_df = df[df["part"] >= 6]

    # The "part" column was only a helper, so remove it
    train_df = train_df.drop(columns=["part"])
    val_df = val_df.drop(columns=["part"])
    test_df = test_df.drop(columns=["part"])

    return train_df, val_df, test_df


def check_patient_split(train_df, val_df, test_df):
    """Stop with an error if any patient appears in more than one set."""
    train_patients = set(train_df[GROUP_COL])
    val_patients = set(val_df[GROUP_COL])
    test_patients = set(test_df[GROUP_COL])

    in_train_and_val = train_patients.intersection(val_patients)
    in_train_and_test = train_patients.intersection(test_patients)
    in_val_and_test = val_patients.intersection(test_patients)

    print("Patients in both train and validation:", len(in_train_and_val))
    print("Patients in both train and test:", len(in_train_and_test))
    print("Patients in both validation and test:", len(in_val_and_test))

    if len(in_train_and_val) > 0 or len(in_train_and_test) > 0 or len(in_val_and_test) > 0:
        raise ValueError("Some patients appear in more than one set. The split is not valid.")

    print("Check passed: every patient is in only one set.")


def check_all_rows_assigned(df, train_df, val_df, test_df):
    """Stop with an error if any row was lost, added, or put into two sets."""
    rows_after_split = len(train_df) + len(val_df) + len(test_df)
    print("Rows before split:", len(df))
    print("Rows after split:", rows_after_split)

    if rows_after_split != len(df):
        raise ValueError("The number of rows changed during the split.")

    # Every record_id must appear exactly once across the three sets
    all_record_ids = list(train_df["record_id"]) + list(val_df["record_id"]) + list(test_df["record_id"])
    if len(set(all_record_ids)) != len(all_record_ids):
        raise ValueError("Some record_id values appear in more than one set.")
    if set(all_record_ids) != set(df["record_id"]):
        raise ValueError("The record_id values after the split do not match the original data.")

    print("Check passed: every row is in exactly one set.")


def check_both_classes(train_df, val_df, test_df):
    """Stop with an error if any set does not contain both normal (0) and abnormal (1)."""
    for name, set_df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        classes_in_set = set(set_df[TARGET_COL].unique())
        if classes_in_set != {0, 1}:
            raise ValueError(name + " set does not contain both classes: " + str(classes_in_set))

    print("Check passed: all three sets contain both classes.")


def check_patterns_across_sets(train_df, val_df, test_df, feature_cols):
    """
    Count identical feature rows that appear in more than one set.

    This does NOT stop the code. It prints a warning so you can investigate.
    Returns the number of feature patterns that cross between sets.
    """
    # Put the three sets together, remembering which set each row came from
    train_part = train_df[feature_cols].copy()
    train_part["set"] = "train"
    val_part = val_df[feature_cols].copy()
    val_part["set"] = "val"
    test_part = test_df[feature_cols].copy()
    test_part["set"] = "test"
    combined = pd.concat([train_part, val_part, test_part])

    # Look only at rows whose features are repeated somewhere
    is_repeated = combined.duplicated(subset=feature_cols, keep=False)
    repeated = combined[is_repeated]

    if len(repeated) == 0:
        print("Check passed: no identical feature rows cross between sets.")
        return 0

    # For each repeated pattern, count in how many different sets it appears
    sets_per_pattern = repeated.groupby(feature_cols, dropna=False)["set"].nunique()
    crossing_patterns = (sets_per_pattern > 1).sum()

    if crossing_patterns > 0:
        print("WARNING:", crossing_patterns, "identical feature patterns appear in more than one set.")
        print("Investigate these before training (see print_duplicate_summary).")
    else:
        print("Check passed: no identical feature rows cross between sets.")

    return crossing_patterns


def print_split_summary(train_df, val_df, test_df):
    """Print and return a table with records, patients and class counts for each set."""
    total_records = len(train_df) + len(val_df) + len(test_df)
    total_patients = train_df[GROUP_COL].nunique() + val_df[GROUP_COL].nunique() + test_df[GROUP_COL].nunique()

    rows = []
    for name, set_df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        records = len(set_df)
        patients = set_df[GROUP_COL].nunique()
        normal = (set_df[TARGET_COL] == 0).sum()
        abnormal = (set_df[TARGET_COL] == 1).sum()

        rows.append({
            "Dataset": name,
            "Records": records,
            "Unique Patients": patients,
            "Normal (0)": normal,
            "Abnormal (1)": abnormal,
            "Records %": round(records / total_records * 100, 1),
            "Patients %": round(patients / total_patients * 100, 1),
            "Abnormal %": round(abnormal / records * 100, 1),
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary


def make_split_manifest(train_df, val_df, test_df):
    """
    Make a small table that says which set every recording belongs to.

    Columns: record_id, patient_id, split.
    Team members can use it to check they all have the same split.
    """
    train_part = train_df[ID_COLS].copy()
    train_part["split"] = "train"
    val_part = val_df[ID_COLS].copy()
    val_part["split"] = "val"
    test_part = test_df[ID_COLS].copy()
    test_part["split"] = "test"

    manifest = pd.concat([train_part, val_part, test_part])
    return manifest


def save_splits(train_df, val_df, test_df, out_dir):
    """
    Save train.csv, val.csv, test.csv and split_manifest.csv.

    If any of these files already exists, nothing is saved,
    so earlier results are never overwritten by mistake.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    manifest_path = out_dir / "split_manifest.csv"

    # Check all paths first, so we never save only some of the files
    for path in [train_path, val_path, test_path, manifest_path]:
        if path.exists():
            raise FileExistsError(str(path) + " already exists. Delete or rename it first.")

    manifest = make_split_manifest(train_df, val_df, test_df)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    print("Saved:", train_path)
    print("Saved:", val_path)
    print("Saved:", test_path)
    print("Saved:", manifest_path)


def prepare_processed_dataset(raw_df, out_dir, seed=42):
    """
    Run all preprocessing steps in order and save the result.

    raw_df should come from load_dataset() in data_loader.py.
    Returns train_df, val_df and test_df.
    """
    print("Step 1: find feature columns")
    feature_cols = get_feature_columns(raw_df)

    print("\nStep 2: keep only the needed columns")
    df = select_columns(raw_df, feature_cols)

    print("\nStep 3: check features and target")
    check_features_are_numeric(df, feature_cols)
    check_target_values(df)

    print("\nStep 4: change infinite values to NaN")
    df = replace_inf_with_nan(df, feature_cols)

    print("\nStep 5: split by patient (70 / 15 / 15)")
    train_df, val_df, test_df = split_by_patient(df, seed)
    print_split_summary(train_df, val_df, test_df)

    print("\nStep 6: check the split")
    check_patient_split(train_df, val_df, test_df)
    check_all_rows_assigned(df, train_df, val_df, test_df)
    check_both_classes(train_df, val_df, test_df)
    check_patterns_across_sets(train_df, val_df, test_df, feature_cols)

    print("\nStep 7: save the three sets and the split manifest")
    save_splits(train_df, val_df, test_df, out_dir)

    return train_df, val_df, test_df
