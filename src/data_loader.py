"""
Load the raw JNU-CTG dataset.

This file does these steps:
  1. Read data/raw/metadata.csv into a DataFrame.
  2. Rename the badly named column F_Med.1 to F_MedAD.
  3. Check that the ID columns and the target column exist.

The raw file itself is never changed.
"""

import pandas as pd
from config.config import RAW_DATA_FILE, RECORD_ID_COLUMN, PATIENT_ID_COLUMN, TARGET_COLUMN


def load_raw_data(file_path=RAW_DATA_FILE):
    """Read the raw CSV file and return it as a DataFrame."""
    # encoding="utf-8-sig" removes a hidden character (BOM) at the start of the file.
    # Without it, the first column is read as "﻿record_id" instead of "record_id".
    # low_memory=False makes pandas look at the whole column before choosing its type,
    # instead of guessing from small pieces (this is what causes the mixed-type warning).
    df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)

    print("Loaded:", file_path)
    print("Shape (rows, columns):", df.shape)
    return df


def rename_columns(df):
    """
    Rename F_Med.1 to F_MedAD.

    F_Med.1 is the median absolute deviation of FHR. Its name looks like a
    copy of F_Med, so we give it a clear name.
    """
    new_df = df.copy()

    if "F_Med.1" in new_df.columns:
        new_df = new_df.rename(columns={"F_Med.1": "F_MedAD"})
        print("Renamed column: F_Med.1 -> F_MedAD")

    return new_df


def check_required_columns(df):
    """Stop with an error if the ID columns or the target column are missing."""
    required_columns = [RECORD_ID_COLUMN, PATIENT_ID_COLUMN, TARGET_COLUMN]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError("Required column is missing: " + col)

    print("Check passed: record_id, patient_id and figo_label are present.")


def load_dataset():
    """Run all loading steps and return the raw DataFrame ready for preprocessing."""
    df = load_raw_data()
    df = rename_columns(df)
    check_required_columns(df)
    return df
