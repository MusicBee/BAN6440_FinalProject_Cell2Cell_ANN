#!/usr/bin/env python
"""
Cell2Cell data preparation and quality-audit script.

This script performs conservative, reproducible source-level checks before model
training. It does NOT impute missing predictor values or scale/encode features:
those operations must be fitted on the training partition only inside the model
pipeline to prevent data leakage.

Usage:
    python cell2cell_data_prep.py --input cell2celltrain.csv --output-dir outputs/data_prep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COLUMN = "CustomerID"
POTENTIAL_LEAKAGE_COLUMNS = [
    "RetentionCalls",
    "RetentionOffersAccepted",
    "MadeCallToRetentionTeam",
]
TARGET_MAP = {"yes": 1, "no": 0, "1": 1, "0": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and normalize Cell2Cell source data.")
    parser.add_argument("--input", default="cell2celltrain.csv", help="Path to labelled Cell2Cell CSV.")
    parser.add_argument("--output-dir", default="outputs/data_prep", help="Directory for output artifacts.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    raw = pd.read_csv(input_path, low_memory=False)
    if raw.empty:
        raise ValueError("Input dataset is empty.")

    initial_rows, initial_columns = raw.shape
    initial_duplicate_rows = int(raw.duplicated().sum())

    # Normalize column names and string values conservatively.
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names after trimming: {duplicates}")

    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].map(lambda x: x.strip() if isinstance(x, str) else x)
        df[column] = df[column].replace("", np.nan)

    required = {TARGET, ID_COLUMN}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Remove exact duplicates, but do not deduplicate by ID automatically.
    df = df.drop_duplicates().copy()
    duplicate_ids = int(df[ID_COLUMN].duplicated().sum())

    # Normalize the target and reject unknown labels rather than guessing.
    normalized_target = df[TARGET].astype("string").str.strip().str.lower().map(TARGET_MAP)
    if normalized_target.isna().any():
        unexpected = sorted(
            df.loc[normalized_target.isna(), TARGET].astype("string").dropna().unique().tolist()
        )
        raise ValueError(f"Unexpected or missing target labels found: {unexpected}")
    df[TARGET] = normalized_target.astype("int64")

    if df[TARGET].nunique() != 2:
        raise ValueError("Target must contain both classes after normalization.")

    missing_by_column = df.isna().sum().sort_values(ascending=False)
    missing_by_column = missing_by_column[missing_by_column > 0]
    missing_total = int(df.isna().sum().sum())

    churn_counts = df[TARGET].value_counts().sort_index()
    churn_distribution = {
        "No (0)": int(churn_counts.get(0, 0)),
        "Yes (1)": int(churn_counts.get(1, 0)),
    }
    churn_rate = float(df[TARGET].mean())

    # Save normalized staging data. Missing predictors remain missing intentionally.
    cleaned_path = output_dir / "cell2cell_normalized_staging.csv"
    df.to_csv(cleaned_path, index=False)

    missing_path = output_dir / "missing_values_by_column.csv"
    missing_by_column.rename_axis("column").reset_index(name="missing_count").to_csv(
        missing_path, index=False
    )

    excluded_present = [c for c in [ID_COLUMN, *POTENTIAL_LEAKAGE_COLUMNS] if c in df.columns]
    summary = {
        "input_file": str(input_path.resolve()),
        "rows_before_duplicate_removal": int(initial_rows),
        "columns_before_normalization": int(initial_columns),
        "exact_duplicate_rows_found": initial_duplicate_rows,
        "exact_duplicate_rows_removed": initial_duplicate_rows,
        "rows_after_duplicate_removal": int(len(df)),
        "duplicate_customer_ids_after_exact_deduplication": duplicate_ids,
        "total_missing_cells_after_normalization": missing_total,
        "columns_with_missing_values": int(len(missing_by_column)),
        "target": TARGET,
        "target_mapping": {"No": 0, "Yes": 1},
        "target_distribution": churn_distribution,
        "churn_rate": churn_rate,
        "identifier_and_potential_leakage_columns_to_exclude_from_primary_features": excluded_present,
        "missing_value_policy": (
            "Predictor imputation must occur after splitting and be fitted on training data only. "
            "This script intentionally does not impute missing values."
        ),
        "outlier_policy": (
            "No automatic outlier deletion or capping is performed. Inspect ranges and data dictionary; "
            "correct or exclude only values shown to be invalid by documented rules."
        ),
        "normalized_staging_file": str(cleaned_path),
        "missing_values_file": str(missing_path),
    }
    summary_path = output_dir / "data_quality_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("CELL2CELL DATA PREPARATION SUMMARY")
    print("---------------------------------")
    print(f"Input: {input_path.resolve()}")
    print(f"Rows before duplicate removal: {initial_rows:,}")
    print(f"Exact duplicate rows removed: {initial_duplicate_rows:,}")
    print(f"Rows after duplicate removal: {len(df):,}")
    print(f"Duplicate CustomerIDs remaining: {duplicate_ids:,}")
    print(f"Missing cells after normalization: {missing_total:,}")
    print(f"Columns with missing values: {len(missing_by_column):,}")
    print(f"Churn distribution: {churn_distribution}")
    print(f"Normalized staging CSV: {cleaned_path}")
    print(f"Missing-values CSV: {missing_path}")
    print(f"Quality summary JSON: {summary_path}")
    print("\nNote: imputation, scaling, and encoding remain in the model pipeline to avoid leakage.")


if __name__ == "__main__":
    main()
