from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Displays the first n rows of a 1-based range of columns as a table for quick DataFrame inspection
from IPython.display import display

def print_column_values(df, columns, start, end, n=10):
    """
    Display selected columns as a table.

    start and end are 1-based column positions.
    """

    selected_columns = columns[start - 1:end]

    print(f"\nShowing columns {start} to {end}")

    table = df[selected_columns].head(n)

    display(table)
    
"""Check the values in a specific column."""
def check_column_values(df, column):
    print(f"Column: {column}")
    print(f"Available values: {df[column].dropna().unique()}")
    print(f"Missing values: {df[column].isna().sum()}")
    print("\nCategory counts:")
    print(df[column].value_counts(dropna=False))
    
    
"""Plot a histogram for a specific feature in the DataFrame."""

def plot_feature_histogram(df, column, bins=30):

    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return

    # Convert values to numeric
    data = pd.to_numeric(df[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)

    # Calculate statistics
    total = len(data)
    available = data.notna().sum()
    missing = data.isna().sum()
    unique = data.nunique()

    print(f"Feature: {column}")
    print(f"Total records: {total}")
    print(f"Available values: {available}")
    print(f"Missing/invalid values: {missing}")
    print(f"Unique values: {unique}")

    if available == 0:
        print("No numerical values available.")
        return

    plt.figure(figsize=(10, 5))

    # Use discrete bins for features with few unique values
    if unique <= 15:
        sns.histplot(data.dropna(), discrete=True)
    else:
        sns.histplot(data.dropna(), bins=bins, kde=True)

    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Duplicate and repeated-recording checks.
# These functions only REPORT. They never delete or change any rows.
# ---------------------------------------------------------------------------

RECORD_ID_COLUMN = "record_id"
PATIENT_ID_COLUMN = "patient_id"
TARGET_COLUMN = "figo_label"


def count_exact_duplicate_rows(df):
    """
    Count rows that are an exact copy of an earlier row.

    record_id is left out of the comparison, because every recording has its
    own record_id. If we included it, we would always find 0 duplicates.
    """
    columns_to_compare = []
    for col in df.columns:
        if col != RECORD_ID_COLUMN:
            columns_to_compare.append(col)

    duplicate_flags = df.duplicated(subset=columns_to_compare, keep="first")
    return duplicate_flags.sum()


def count_duplicate_record_ids(df):
    """Count record_id values that appear more than once (each should be unique)."""
    if RECORD_ID_COLUMN not in df.columns:
        print("record_id column not found, skipping this check.")
        return 0

    duplicate_flags = df[RECORD_ID_COLUMN].duplicated(keep="first")
    return duplicate_flags.sum()


def count_recordings_per_patient(df):
    """Return a Series that shows how many recordings each patient has."""
    return df[PATIENT_ID_COLUMN].value_counts()


def find_repeated_feature_patterns(df, feature_cols):
    """
    Find rows whose feature values are exactly the same as another row.

    Returns two tables:
      repeated_rows   - every row that shares its feature values with another row.
                        Rows with the same "pattern_id" are identical.
      pattern_summary - one row per pattern:
                          rows     = how many rows share this pattern
                          patients = how many different patients have it
                          labels   = how many different figo_label values it has
                          type     = what kind of repeat it is (in words)
    """
    # keep=False marks ALL copies, not only the second one
    is_repeated = df.duplicated(subset=feature_cols, keep=False)
    repeated_rows = df[is_repeated].copy()

    if len(repeated_rows) == 0:
        empty_summary = pd.DataFrame(columns=["rows", "patients", "labels", "type"])
        return repeated_rows, empty_summary

    # Give every group of identical rows the same number
    repeated_rows["pattern_id"] = repeated_rows.groupby(feature_cols, dropna=False).ngroup()

    # Count rows, patients and labels for each pattern
    pattern_summary = repeated_rows.groupby("pattern_id").agg(
        rows=(PATIENT_ID_COLUMN, "size"),
        patients=(PATIENT_ID_COLUMN, "nunique"),
        labels=(TARGET_COLUMN, "nunique"),
    )

    # Describe each pattern in words
    pattern_types = []
    for index, row in pattern_summary.iterrows():
        if row["labels"] > 1:
            pattern_types.append("conflicting labels")
        elif row["patients"] > 1:
            pattern_types.append("different patients")
        else:
            pattern_types.append("same patient")
    pattern_summary["type"] = pattern_types

    return repeated_rows, pattern_summary


def print_duplicate_summary(df, feature_cols):
    """
    Print a summary of duplicates and repeated recordings. Nothing is removed.

    Returns repeated_rows and pattern_summary so you can look at them in detail.
    """
    recordings_per_patient = count_recordings_per_patient(df)
    repeated_rows, pattern_summary = find_repeated_feature_patterns(df, feature_cols)

    same_patient = pattern_summary[pattern_summary["type"] == "same patient"]
    different_patients = pattern_summary[pattern_summary["type"] == "different patients"]
    conflicting = pattern_summary[pattern_summary["type"] == "conflicting labels"]

    print("Total records:", len(df))
    print("Unique patients:", df[PATIENT_ID_COLUMN].nunique())
    print("Exact duplicate rows (all columns except record_id):", count_exact_duplicate_rows(df))
    print("Duplicate record_id values:", count_duplicate_record_ids(df))
    print("Patients with more than one recording:", (recordings_per_patient > 1).sum())
    print("Most recordings for one patient:", recordings_per_patient.max())
    print()
    print("Repeated feature patterns (all features identical):", len(pattern_summary))
    print("  - same patient, same label (likely accidental copy):", len(same_patient))
    print("  - different patients, same label (suspicious):", len(different_patients))
    print("  - conflicting labels (suspicious):", len(conflicting))
    print("Rows involved in repeated patterns:", len(repeated_rows))

    return repeated_rows, pattern_summary