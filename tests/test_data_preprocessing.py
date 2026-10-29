"""
Tests for src.data_preprocessing, src.data_inspection and src.data_loader.

All tests use small FAKE datasets made inside this file.
The real data (and especially the held-out test set) is never used here.

Run from the project folder with:
    python -m pytest tests/test_data_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import check_required_columns
from src.data_inspection import (
    count_exact_duplicate_rows,
    count_duplicate_record_ids,
    count_recordings_per_patient,
    find_repeated_feature_patterns,
)
from src.data_preprocessing import (
    get_feature_columns,
    select_columns,
    check_target_values,
    split_by_patient,
    check_patient_split,
    check_all_rows_assigned,
    check_both_classes,
    check_patterns_across_sets,
)


# ---------------------------------------------------------------------------
# Helper: build a small fake dataset that looks like metadata.csv
# ---------------------------------------------------------------------------

def make_feature_names():
    """Make 103 fake feature names that start with F_, U_ or FU_ (like the real data)."""
    names = []
    for i in range(56):
        names.append("F_feat" + str(i))
    for i in range(23):
        names.append("U_feat" + str(i))
    for i in range(24):
        names.append("FU_feat" + str(i))
    return names


def make_fake_dataset(n_patients=200, seed=0):
    """
    Make a fake dataset.

    Each patient has 1 to 3 recordings. Features are random numbers, so no two
    rows are identical. It also has columns that must NOT become features:
    a clinical column and the NetFea_* columns.
    """
    rng = np.random.default_rng(seed)
    feature_names = make_feature_names()

    rows = []
    record_number = 0
    for patient in range(n_patients):
        number_of_recordings = rng.integers(1, 4)
        for r in range(number_of_recordings):
            row = {}
            row["record_id"] = "rec_" + str(record_number)
            row["patient_id"] = patient
            row["gravidity"] = 1
            for name in feature_names:
                row[name] = rng.normal()
            for i in range(5):
                row["NetFea_" + str(i)] = rng.normal()
            row["figo_label"] = int(rng.integers(0, 2))
            rows.append(row)
            record_number = record_number + 1

    return pd.DataFrame(rows)


def make_ready_dataset(n_patients=200, seed=0):
    """Fake dataset after column selection (record_id, patient_id, 103 features, figo_label)."""
    raw_df = make_fake_dataset(n_patients, seed)
    feature_cols = get_feature_columns(raw_df)
    df = select_columns(raw_df, feature_cols)
    return df, feature_cols


# ---------------------------------------------------------------------------
# 1. Exact duplicate detection
# ---------------------------------------------------------------------------

def test_no_duplicates_in_clean_data():
    df, feature_cols = make_ready_dataset()
    assert count_exact_duplicate_rows(df) == 0
    assert count_duplicate_record_ids(df) == 0


def test_exact_duplicate_row_is_found():
    df, feature_cols = make_ready_dataset()

    # Copy the first row but give it a new record_id
    copied_row = df.iloc[[0]].copy()
    copied_row["record_id"] = "rec_copy"
    df_with_copy = pd.concat([df, copied_row], ignore_index=True)

    assert count_exact_duplicate_rows(df_with_copy) == 1


def test_duplicate_record_id_is_found():
    df, feature_cols = make_ready_dataset()

    # Give the second row the same record_id as the first row
    df.loc[1, "record_id"] = df.loc[0, "record_id"]

    assert count_duplicate_record_ids(df) == 1


# ---------------------------------------------------------------------------
# 2. Repeated patient recordings are NOT treated as duplicates
# ---------------------------------------------------------------------------

def test_recordings_per_patient_are_counted():
    df = pd.DataFrame({"patient_id": [1, 1, 1, 2, 3, 3]})
    counts = count_recordings_per_patient(df)

    assert counts[1] == 3
    assert counts[2] == 1
    assert counts[3] == 2


def test_different_recordings_of_same_patient_are_not_flagged():
    df, feature_cols = make_ready_dataset()

    # Many patients have several recordings, but their features differ
    assert (count_recordings_per_patient(df) > 1).sum() > 0

    repeated_rows, pattern_summary = find_repeated_feature_patterns(df, feature_cols)
    assert len(repeated_rows) == 0
    assert len(pattern_summary) == 0


def test_repeated_patterns_are_described_correctly():
    df, feature_cols = make_ready_dataset()

    # Pattern A: copy of row 0 for the SAME patient and same label
    same_patient_copy = df.iloc[[0]].copy()
    same_patient_copy["record_id"] = "copy_same_patient"

    # Pattern B: copy of row 5 for a DIFFERENT patient, same label
    other_patient_copy = df.iloc[[5]].copy()
    other_patient_copy["record_id"] = "copy_other_patient"
    other_patient_copy["patient_id"] = 99999

    # Pattern C: copy of row 10 with the OTHER label
    conflicting_copy = df.iloc[[10]].copy()
    conflicting_copy["record_id"] = "copy_conflicting"
    conflicting_copy["figo_label"] = 1 - conflicting_copy["figo_label"]

    df = pd.concat([df, same_patient_copy, other_patient_copy, conflicting_copy], ignore_index=True)

    repeated_rows, pattern_summary = find_repeated_feature_patterns(df, feature_cols)
    types = list(pattern_summary["type"])

    assert len(pattern_summary) == 3
    assert types.count("same patient") == 1
    assert types.count("different patients") == 1
    assert types.count("conflicting labels") == 1


# ---------------------------------------------------------------------------
# 3. Patient separation across splits
# ---------------------------------------------------------------------------

def test_no_patient_in_two_sets():
    df, feature_cols = make_ready_dataset()
    train_df, val_df, test_df = split_by_patient(df, seed=42)

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0

    # The check function should also pass without an error
    check_patient_split(train_df, val_df, test_df)


def test_patient_overlap_is_detected():
    df, feature_cols = make_ready_dataset()
    train_df, val_df, test_df = split_by_patient(df, seed=42)

    # Put one training row into the test set on purpose
    bad_test_df = pd.concat([test_df, train_df.iloc[[0]]])

    with pytest.raises(ValueError):
        check_patient_split(train_df, val_df, bad_test_df)


def test_split_sizes_are_close_to_70_15_15():
    df, feature_cols = make_ready_dataset(n_patients=400)
    train_df, val_df, test_df = split_by_patient(df, seed=42)

    total = len(df)
    assert abs(len(train_df) / total - 0.70) < 0.05
    assert abs(len(val_df) / total - 0.15) < 0.05
    assert abs(len(test_df) / total - 0.15) < 0.05


# ---------------------------------------------------------------------------
# 4. Reproducible splitting with seed 42
# ---------------------------------------------------------------------------

def test_same_seed_gives_same_split():
    df, feature_cols = make_ready_dataset()

    train_1, val_1, test_1 = split_by_patient(df, seed=42)
    train_2, val_2, test_2 = split_by_patient(df, seed=42)

    assert list(train_1["record_id"]) == list(train_2["record_id"])
    assert list(val_1["record_id"]) == list(val_2["record_id"])
    assert list(test_1["record_id"]) == list(test_2["record_id"])


def test_different_seed_gives_different_split():
    df, feature_cols = make_ready_dataset()

    train_1, val_1, test_1 = split_by_patient(df, seed=42)
    train_2, val_2, test_2 = split_by_patient(df, seed=7)

    assert set(test_1["record_id"]) != set(test_2["record_id"])


# ---------------------------------------------------------------------------
# 5. Correct predictor count
# ---------------------------------------------------------------------------

def test_exactly_103_features_are_found():
    raw_df = make_fake_dataset()
    feature_cols = get_feature_columns(raw_df)

    assert len(feature_cols) == 103


def test_forbidden_columns_are_not_features():
    raw_df = make_fake_dataset()
    feature_cols = get_feature_columns(raw_df)

    forbidden = ["record_id", "patient_id", "figo_label", "gravidity",
                 "NetFea_0", "NetFea_1", "NetFea_2", "NetFea_3", "NetFea_4"]
    for col in forbidden:
        assert col not in feature_cols


def test_wrong_feature_count_stops_the_code():
    raw_df = make_fake_dataset()
    raw_df = raw_df.drop(columns=["F_feat0"])

    with pytest.raises(ValueError):
        get_feature_columns(raw_df)


# ---------------------------------------------------------------------------
# 6. Missing required columns
# ---------------------------------------------------------------------------

def test_required_columns_present_passes():
    raw_df = make_fake_dataset()
    check_required_columns(raw_df)


def test_missing_patient_id_stops_the_code():
    raw_df = make_fake_dataset()
    raw_df = raw_df.drop(columns=["patient_id"])

    with pytest.raises(ValueError):
        check_required_columns(raw_df)


def test_missing_target_stops_the_code():
    raw_df = make_fake_dataset()
    raw_df = raw_df.drop(columns=["figo_label"])

    with pytest.raises(ValueError):
        check_required_columns(raw_df)


# ---------------------------------------------------------------------------
# 7. Invalid target labels
# ---------------------------------------------------------------------------

def test_valid_labels_pass():
    df, feature_cols = make_ready_dataset()
    check_target_values(df)


def test_label_other_than_0_or_1_stops_the_code():
    df, feature_cols = make_ready_dataset()
    df.loc[0, "figo_label"] = 2

    with pytest.raises(ValueError):
        check_target_values(df)


def test_missing_label_stops_the_code():
    df, feature_cols = make_ready_dataset()
    df["figo_label"] = df["figo_label"].astype(float)
    df.loc[0, "figo_label"] = np.nan

    with pytest.raises(ValueError):
        check_target_values(df)


# ---------------------------------------------------------------------------
# 8. Every row is assigned to exactly one set
# ---------------------------------------------------------------------------

def test_every_row_is_assigned_once():
    df, feature_cols = make_ready_dataset()
    train_df, val_df, test_df = split_by_patient(df, seed=42)

    check_all_rows_assigned(df, train_df, val_df, test_df)


def test_lost_row_is_detected():
    df, feature_cols = make_ready_dataset()
    train_df, val_df, test_df = split_by_patient(df, seed=42)

    # Remove one row from the training set on purpose
    smaller_train_df = train_df.iloc[1:]

    with pytest.raises(ValueError):
        check_all_rows_assigned(df, smaller_train_df, val_df, test_df)


def test_split_does_not_change_original_data():
    df, feature_cols = make_ready_dataset()
    df_before = df.copy()

    split_by_patient(df, seed=42)

    assert df.equals(df_before)
    assert "part" not in df.columns


def test_identical_rows_in_two_sets_are_reported():
    df, feature_cols = make_ready_dataset()
    train_df, val_df, test_df = split_by_patient(df, seed=42)
    assert check_patterns_across_sets(train_df, val_df, test_df, feature_cols) == 0

    # Copy one training row into the test set (as a different patient)
    copied_row = train_df.iloc[[0]].copy()
    copied_row["patient_id"] = 99999
    copied_row["record_id"] = "copy"
    test_with_copy = pd.concat([test_df, copied_row])

    assert check_patterns_across_sets(train_df, val_df, test_with_copy, feature_cols) == 1


# ---------------------------------------------------------------------------
# 9. Small or unusually imbalanced patient groups
# ---------------------------------------------------------------------------

def test_patient_with_many_recordings_stays_together():
    df, feature_cols = make_ready_dataset()

    # Give patient 0 ten extra recordings (like the real maximum of 10)
    extra_rows = []
    for i in range(10):
        new_row = df[df["patient_id"] == 0].iloc[[0]].copy()
        new_row["record_id"] = "extra_" + str(i)
        # Give the new recording its own random feature values (all 103 at once)
        new_row[feature_cols] = np.random.default_rng(i).normal(size=(1, len(feature_cols)))
        extra_rows.append(new_row)
    # .copy() stores the combined table neatly in memory (avoids a pandas warning)
    df = pd.concat([df] + extra_rows, ignore_index=True).copy()

    train_df, val_df, test_df = split_by_patient(df, seed=42)

    sets_with_patient_0 = 0
    for set_df in [train_df, val_df, test_df]:
        if 0 in set(set_df["patient_id"]):
            sets_with_patient_0 = sets_with_patient_0 + 1

    assert sets_with_patient_0 == 1
    check_all_rows_assigned(df, train_df, val_df, test_df)


def test_imbalanced_data_still_has_both_classes():
    df, feature_cols = make_ready_dataset(n_patients=400)

    # Make the data imbalanced: only about 20% of patients are abnormal
    rng = np.random.default_rng(1)
    for patient in df["patient_id"].unique():
        label = 1 if rng.random() < 0.2 else 0
        df.loc[df["patient_id"] == patient, "figo_label"] = label

    train_df, val_df, test_df = split_by_patient(df, seed=42)

    check_both_classes(train_df, val_df, test_df)
    check_patient_split(train_df, val_df, test_df)


def test_set_with_one_class_is_detected():
    df, feature_cols = make_ready_dataset()
    train_df, val_df, test_df = split_by_patient(df, seed=42)

    # Keep only normal rows in the test set on purpose
    only_normal_test_df = test_df[test_df["figo_label"] == 0]

    with pytest.raises(ValueError):
        check_both_classes(train_df, val_df, only_normal_test_df)
