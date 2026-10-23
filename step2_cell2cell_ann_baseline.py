from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
CLASSIFICATION_THRESHOLD = 0.50

ID_COLUMNS = ["CustomerID"]
POTENTIAL_LEAKAGE_COLUMNS = [
    "RetentionCalls",
    "RetentionOffersAccepted",
    "MadeCallToRetentionTeam",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a baseline ANN using Cell2Cell churn data."
    )
    parser.add_argument(
        "--data",
        default="cell2celltrain.csv",
        help="Path to labelled Cell2Cell training CSV (default: cell2celltrain.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cell2cell_baseline",
        help="Directory for model, metrics, and evaluation artifacts.",
    )
    return parser.parse_args()


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_and_validate_data(data_path: Path) -> tuple[pd.DataFrame, dict]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}\n"
            "Place cell2celltrain.csv in the project directory or pass --data with its path."
        )

    dataset = pd.read_csv(data_path, low_memory=False)
    dataset.columns = dataset.columns.str.strip()

    required_columns = {"Churn", "CustomerID"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {dataset.columns.tolist()}"
        )
    if dataset.empty:
        raise ValueError("The dataset contains no records.")

    initial_rows = len(dataset)
    duplicate_count = int(dataset.duplicated().sum())
    if duplicate_count:
        dataset = dataset.drop_duplicates().copy()

    # Trim string values and turn blank strings into missing values.
    object_columns = dataset.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        dataset[column] = dataset[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        dataset[column] = dataset[column].replace("", np.nan)

    # Normalize target labels.
    target = dataset["Churn"].astype("string").str.strip().str.lower()
    target_map = {"yes": 1, "no": 0, "1": 1, "0": 0}
    normalized_target = target.map(target_map)
    if normalized_target.isna().any():
        unexpected = sorted(target[normalized_target.isna()].dropna().unique().tolist())
        raise ValueError(
            "Churn must contain Yes/No or 1/0 values. "
            f"Unexpected target values: {unexpected}"
        )
    dataset["Churn"] = normalized_target.astype("int64")

    if dataset["Churn"].nunique() != 2:
        raise ValueError("Churn target must contain both classes (0 and 1).")

    missing_by_column = dataset.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

    # Report the excluded columns that are present in this version of the file.
    present_id_columns = [c for c in ID_COLUMNS if c in dataset.columns]
    present_leakage_columns = [c for c in POTENTIAL_LEAKAGE_COLUMNS if c in dataset.columns]
    excluded_columns = present_id_columns + present_leakage_columns

    print("\nDATA QUALITY SUMMARY")
    print("--------------------")
    print(f"Input file: {data_path.resolve()}")
    print(f"Rows before duplicate removal: {initial_rows:,}")
    print(f"Exact duplicate rows removed: {duplicate_count:,}")
    print(f"Rows available for modelling: {len(dataset):,}")
    print(f"Original columns: {dataset.shape[1]:,}")
    print(f"Total missing cells before preprocessing: {int(dataset.isna().sum().sum()):,}")
    print("Missing values by column:")
    if missing_by_column.empty:
        print("  None")
    else:
        print(missing_by_column.to_string())
    print(
        "Churn distribution:\n"
        + dataset["Churn"].value_counts().sort_index()
        .rename(index={0: "No (0)", 1: "Yes (1)"}).to_string()
    )
    print(f"Excluded identifier columns: {present_id_columns or 'None'}")
    print(f"Excluded potential leakage columns: {present_leakage_columns or 'None'}")
    print(
        "Preprocessing note: predictor imputers, scaling, and encoding are fitted "
        "on the training split only."
    )

    config = {
        "input_file": str(data_path.resolve()),
        "input_rows": int(initial_rows),
        "rows_after_duplicate_removal": int(len(dataset)),
        "duplicate_rows_removed": int(duplicate_count),
        "columns_in_input": int(dataset.shape[1]),
        "total_missing_cells_before_preprocessing": int(dataset.isna().sum().sum()),
        "missing_values_by_column": {str(k): int(v) for k, v in missing_by_column.items()},
        "churn_counts": {str(k): int(v) for k, v in dataset["Churn"].value_counts().items()},
        "excluded_identifier_columns": present_id_columns,
        "excluded_potential_leakage_columns": present_leakage_columns,
    }
    return dataset, config


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X_train.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_baseline_ann(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="customer_features"),
            tf.keras.layers.Dense(32, activation="relu", name="hidden_layer_1"),
            tf.keras.layers.Dense(16, activation="relu", name="hidden_layer_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="churn_probability"),
        ],
        name="cell2cell_baseline_ann",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
    }


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seeds()
    dataset, data_quality = load_and_validate_data(data_path)

    excluded = [
        column for column in ID_COLUMNS + POTENTIAL_LEAKAGE_COLUMNS
        if column in dataset.columns
    ]
    X = dataset.drop(columns=["Churn"] + excluded)
    y = dataset["Churn"].astype("int64")

    # Stratified 70% train / 30% temporary split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Split temporary data equally into 15% validation and 15% test.
    relative_test_size = TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=SEED,
        stratify=y_temp,
    )

    preprocessor = make_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train).astype("float32")
    X_val_processed = preprocessor.transform(X_val).astype("float32")
    X_test_processed = preprocessor.transform(X_test).astype("float32")

    print("\nEXPERIMENT DESIGN")
    print("-----------------")
    print(f"Train/validation/test rows: {len(X_train):,}/{len(X_val):,}/{len(X_test):,}")
    print(f"Predictor columns before encoding: {X_train.shape[1]:,}")
    print(f"Encoded feature count: {X_train_processed.shape[1]:,}")
    print(f"Architecture: input -> 32 -> 16 -> 1")
    print(f"Optimizer: Adam; learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}; maximum epochs: {MAX_EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Classification threshold: {CLASSIFICATION_THRESHOLD:.2f}")
    print("Test data is held back until final evaluation.")

    model = build_baseline_ann(X_train_processed.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train_processed,
        y_train.to_numpy(),
        validation_data=(X_val_processed, y_val.to_numpy()),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )

    probabilities = model.predict(X_test_processed, verbose=0).ravel()
    metrics = compute_metrics(y_test.to_numpy(), probabilities)
    report = classification_report(
        y_test,
        (probabilities >= CLASSIFICATION_THRESHOLD).astype(int),
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test,
        (probabilities >= CLASSIFICATION_THRESHOLD).astype(int),
        labels=[0, 1],
    )

    # Persist all artifacts needed to reproduce and document the experiment.
    model.save(output_dir / "baseline_ann.keras")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(
        matrix,
        index=["actual_no_churn", "actual_churn"],
        columns=["predicted_no_churn", "predicted_churn"],
    ).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(
        {
            "actual_churn": y_test.to_numpy(),
            "churn_probability": probabilities,
            "predicted_churn": (probabilities >= CLASSIFICATION_THRESHOLD).astype(int),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    with (output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (output_dir / "data_quality_summary.json").open("w", encoding="utf-8") as f:
        json.dump(data_quality, f, indent=2)

    run_config = {
        **data_quality,
        "seed": SEED,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "predictor_columns_before_encoding": int(X_train.shape[1]),
        "encoded_features": int(X_train_processed.shape[1]),
        "architecture": [32, 16, 1],
        "hidden_activation": "relu",
        "output_activation": "sigmoid",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "binary_crossentropy",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "epochs_trained": int(len(history.history.get("loss", []))),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "metrics": metrics,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\nBASELINE TEST RESULTS")
    print("---------------------")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nTEST CLASSIFICATION REPORT")
    print("--------------------------")
    print(
        classification_report(
            y_test,
            (probabilities >= CLASSIFICATION_THRESHOLD).astype(int),
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )
    print("\nTEST CONFUSION MATRIX (rows=actual, columns=predicted)")
    print(matrix)
    print(f"\nArtifacts saved to: {output_dir.resolve()}")
    print("Do not report results from the previous dataset as Cell2Cell results.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
