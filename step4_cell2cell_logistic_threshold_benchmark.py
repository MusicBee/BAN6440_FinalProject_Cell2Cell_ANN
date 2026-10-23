from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ID_COLUMNS = ["CustomerID"]
LEAKAGE_COLUMNS = [
    "RetentionCalls",
    "RetentionOffersAccepted",
    "MadeCallToRetentionTeam",
]


def calculate_metrics(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_score": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).tolist(),
    }


def save_plot(fig, plot_dir, filename):
    fig.tight_layout()
    fig.savefig(plot_dir / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Cell2Cell logistic regression benchmark and generate charts."
    )
    parser.add_argument("--data", default="cell2celltrain.csv")
    parser.add_argument(
        "--output-dir", default="outputs/cell2cell_logistic_benchmark"
    )
    # Supports notebook execution by ignoring IPython's injected arguments.
    args, _unknown = parser.parse_known_args()

    data_path = Path(args.data).expanduser()
    out = Path(args.output_dir).expanduser()
    plot_dir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()
    if df.empty or "Churn" not in df.columns:
        raise ValueError("Dataset must be non-empty and contain a 'Churn' column.")

    rows_before = len(df)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        df[col] = df[col].replace("", np.nan)

    target = df["Churn"].astype("string").str.strip().str.lower()
    y = target.map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Churn must contain both classes, represented as Yes/No or 1/0.")
    y = y.astype("int64")

    excluded = [c for c in ID_COLUMNS + LEAKAGE_COLUMNS if c in df.columns]
    X = df.drop(columns=["Churn"] + excluded)

    # Same stratified split proportions and seed as the ANN experiments.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=None,
        random_state=SEED,
    )
    model.fit(X_train_p, y_train)

    val_prob = model.predict_proba(X_val_p)[:, 1]
    test_prob = model.predict_proba(X_test_p)[:, 1]

    # Select the threshold on validation data only, maximizing F1.
    threshold_rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        result = calculate_metrics(y_val.to_numpy(), val_prob, float(threshold))
        result.pop("confusion_matrix")
        threshold_rows.append(result)
    validation_results = pd.DataFrame(threshold_rows)
    best = validation_results.sort_values(
        ["f1_score", "precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    selected_threshold = float(best["threshold"])
    validation_results.to_csv(out / "validation_threshold_analysis.csv", index=False)

    test_050 = calculate_metrics(y_test.to_numpy(), test_prob, 0.50)
    test_selected = calculate_metrics(
        y_test.to_numpy(), test_prob, selected_threshold
    )

    # Threshold metric curves.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(validation_results["threshold"], validation_results["precision"], label="Precision")
    ax.plot(validation_results["threshold"], validation_results["recall"], label="Recall")
    ax.plot(validation_results["threshold"], validation_results["f1_score"], label="F1-score", linewidth=2)
    ax.axvline(selected_threshold, linestyle="--", label=f"Selected threshold = {selected_threshold:.2f}")
    ax.set_title("Logistic Regression: Validation Metrics by Threshold")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    save_plot(fig, plot_dir, "01_validation_threshold_metrics.png")

    # ROC and PR curves use test predictions for final evaluation.
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {test_050['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance baseline")
    ax.set_title("Logistic Regression: Test ROC Curve")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (recall)")
    ax.legend()
    save_plot(fig, plot_dir, "02_test_roc_curve.png")

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_prob)
    prevalence = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall_curve, precision_curve, label=f"Average precision = {test_050['average_precision']:.3f}")
    ax.axhline(prevalence, linestyle="--", label=f"Test churn prevalence = {prevalence:.3f}")
    ax.set_title("Logistic Regression: Test Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    save_plot(fig, plot_dir, "03_test_precision_recall_curve.png")

    # Confusion matrices at both thresholds.
    for suffix, result, label in [
        ("050", test_050, "Threshold 0.50"),
        ("selected", test_selected, f"Validation-selected threshold {selected_threshold:.2f}"),
    ]:
        cm = np.asarray(result["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(f"Logistic Regression: Test Confusion Matrix ({label})")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1], ["No churn", "Churn"])
        ax.set_yticks([0, 1], ["No churn", "Churn"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_plot(fig, plot_dir, f"04_confusion_matrix_{suffix}.png")

    results = {
        "model": "LogisticRegression",
        "input_file": str(data_path.resolve()),
        "rows_before_duplicate_removal": int(rows_before),
        "duplicate_rows_removed": int(duplicate_count),
        "rows_after_duplicate_removal": int(len(df)),
        "split_rows": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "excluded_columns": excluded,
        "predictor_columns_before_encoding": int(X_train.shape[1]),
        "encoded_feature_count": int(X_train_p.shape[1]),
        "class_weight": None,
        "threshold_selection": (
            "Validation-only grid 0.05-0.95 in 0.01 increments; maximize F1; "
            "tie-break by precision, then higher threshold."
        ),
        "selected_threshold": selected_threshold,
        "validation_metrics_at_selected_threshold": {
            key: float(value) for key, value in best.to_dict().items()
        },
        "test_metrics_at_0.50": test_050,
        "test_metrics_at_validation_selected_threshold": test_selected,
        "test_churn_prevalence": prevalence,
    }

    joblib.dump(preprocessor, out / "preprocessor.joblib")
    joblib.dump(model, out / "logistic_regression.joblib")
    pd.DataFrame(
        {
            "actual_churn": y_test.to_numpy(),
            "churn_probability": test_prob,
            "predicted_at_050": (test_prob >= 0.50).astype(int),
            "predicted_at_validation_threshold": (
                test_prob >= selected_threshold
            ).astype(int),
        }
    ).to_csv(out / "test_predictions.csv", index=False)
    (out / "benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    cm_050 = np.asarray(test_050["confusion_matrix"])
    cm_selected = np.asarray(test_selected["confusion_matrix"])
    report = [
        "CELL2CELL LOGISTIC REGRESSION BENCHMARK — INTERPRETATION",
        "=" * 61,
        "",
        f"Validation-selected threshold: {selected_threshold:.2f}.",
        "The threshold was selected using validation data only by maximizing F1-score.",
        "",
        "At test threshold 0.50:",
        (
            f"Accuracy={test_050['accuracy']:.4f}, precision={test_050['precision']:.4f}, "
            f"recall={test_050['recall']:.4f}, F1={test_050['f1_score']:.4f}, "
            f"ROC-AUC={test_050['roc_auc']:.4f}, average precision={test_050['average_precision']:.4f}."
        ),
        (
            f"Confusion matrix: TN={cm_050[0,0]}, FP={cm_050[0,1]}, "
            f"FN={cm_050[1,0]}, TP={cm_050[1,1]}."
        ),
        "",
        f"At validation-selected threshold {selected_threshold:.2f}:",
        (
            f"Accuracy={test_selected['accuracy']:.4f}, precision={test_selected['precision']:.4f}, "
            f"recall={test_selected['recall']:.4f}, F1={test_selected['f1_score']:.4f}, "
            f"ROC-AUC={test_selected['roc_auc']:.4f}, average precision={test_selected['average_precision']:.4f}."
        ),
        (
            f"Confusion matrix: TN={cm_selected[0,0]}, FP={cm_selected[0,1]}, "
            f"FN={cm_selected[1,0]}, TP={cm_selected[1,1]}."
        ),
        "",
        (
            "The ROC-AUC and average precision do not change between classification thresholds "
            "because threshold changes alter class labels, not probability ranking."
        ),
        (
            "Compare this benchmark with the ANN using the same test split and matching threshold "
            "policy. Do not claim either model is better until the real outputs are compared."
        ),
        "",
        f"Charts saved in: {plot_dir.resolve()}",
    ]
    (plot_dir / "chart_interpretation.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nLOGISTIC REGRESSION VALIDATION-SELECTED THRESHOLD:", selected_threshold)
    print("VALIDATION METRICS:", results["validation_metrics_at_selected_threshold"])
    print("TEST METRICS AT 0.50:", test_050)
    print("TEST METRICS AT VALIDATION-SELECTED THRESHOLD:", test_selected)
    print(f"\nCharts saved to: {plot_dir.resolve()}")
    print(f"Interpretation saved to: {(plot_dir / 'chart_interpretation.txt').resolve()}")
    print(f"Artifacts saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
