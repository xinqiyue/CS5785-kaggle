"""
TIDE - Final AI vs Human Text Classifier
========================================

This file trains and evaluates only the selected final model:
Random Forest using the 11 original linguistic features, MATTR,
bigram repetition, and trigram repetition.

Input files:
    train.csv  : required; columns text, label (0 = human, 1 = AI)
    test.csv   : optional; columns text and optionally id

Run:
    python tide_final.py

Install:
    pip install numpy pandas scikit-learn matplotlib joblib

All evaluation metrics and plots are saved in the "results" directory.
The model is also fitted on the complete labeled training set and saved.
"""

import json
import re
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5
MATTR_WINDOW = 50
CLASSIFICATION_THRESHOLD = 0.50

TRAIN_FILE = Path(r"..\data\train.csv")
TEST_FILE = Path(r"..\data\test.csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

FEATURE_NAMES = [
    "word_count",
    "unique_word_ratio",
    "avg_word_length",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_std",
    "comma_ratio",
    "semicolon_ratio",
    "question_ratio",
    "exclamation_ratio",
    "parenthesis_ratio",
    "mattr",
    "bigram_repetition",
    "trigram_repetition",
]


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def tokenize(text):
    """Return lowercase word tokens from a passage."""
    if pd.isna(text):
        return []
    return WORD_PATTERN.findall(str(text).lower())


def calculate_mattr(words, window_size=MATTR_WINDOW):
    """Calculate Moving-Average Type-Token Ratio."""
    if not words:
        return 0.0

    if len(words) < window_size:
        return len(set(words)) / len(words)

    scores = [
        len(set(words[i:i + window_size])) / window_size
        for i in range(len(words) - window_size + 1)
    ]
    return float(np.mean(scores))


def calculate_ngram_repetition(words, n):
    """
    Return the proportion of n-gram occurrences that belong to
    n-gram types appearing more than once in the passage.
    """
    if len(words) < n:
        return 0.0

    ngrams = [
        tuple(words[i:i + n])
        for i in range(len(words) - n + 1)
    ]
    counts = Counter(ngrams)
    repeated_count = sum(count for count in counts.values() if count > 1)

    return repeated_count / len(ngrams)


def extract_features(text):
    """Extract the final 14-feature representation for one passage."""
    text = "" if pd.isna(text) else str(text)
    words = tokenize(text)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    word_count = len(words)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    denominator = max(word_count, 1)

    return {
        "word_count": word_count,
        "unique_word_ratio": len(set(words)) / denominator,
        "avg_word_length": (
            float(np.mean([len(word) for word in words])) if words else 0.0
        ),
        "sentence_count": len(sentences),
        "avg_sentence_length": word_count / max(len(sentences), 1),
        "sentence_length_std": (
            float(np.std(sentence_lengths)) if sentence_lengths else 0.0
        ),
        "comma_ratio": text.count(",") / denominator,
        "semicolon_ratio": text.count(";") / denominator,
        "question_ratio": text.count("?") / denominator,
        "exclamation_ratio": text.count("!") / denominator,
        "parenthesis_ratio": (
            text.count("(") + text.count(")")
        ) / denominator,
        "mattr": calculate_mattr(words),
        "bigram_repetition": calculate_ngram_repetition(words, 2),
        "trigram_repetition": calculate_ngram_repetition(words, 3),
    }


def make_feature_matrix(texts):
    """Convert a collection of passages into a numeric feature matrix."""
    matrix = pd.DataFrame(
        [extract_features(text) for text in texts],
        columns=FEATURE_NAMES,
    )
    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ---------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------

def build_model():
    """Create the selected Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------
# Evaluation and saved outputs
# ---------------------------------------------------------------------

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_model(X, y):
    """
    Generate out-of-fold predictions using stratified 5-fold CV.
    Each passage is predicted by a model that did not train on it.
    """
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.zeros(len(y), dtype=float)
    fold_results = []

    print("\nCross-validation")
    print("----------------")

    for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
        model = build_model()
        model.fit(X.iloc[train_idx], y[train_idx])

        probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof_probabilities[valid_idx] = probabilities

        fold_auc = roc_auc_score(y[valid_idx], probabilities)
        fold_results.append({
            "fold": fold_number,
            "auroc": fold_auc,
        })
        print(f"Fold {fold_number}: AUROC = {fold_auc:.6f}")

    fold_results = pd.DataFrame(fold_results)
    fold_results.to_csv(OUTPUT_DIR / "fold_scores.csv", index=False)

    predictions = (
        oof_probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    mean_auc = float(fold_results["auroc"].mean())
    std_auc = float(fold_results["auroc"].std(ddof=0))
    pooled_auc = float(roc_auc_score(y, oof_probabilities))

    metrics = {
        "model": "RandomForestClassifier",
        "mean_cv_auroc": mean_auc,
        "std_cv_auroc": std_auc,
        "pooled_oof_auroc": pooled_auc,
        "accuracy_at_0_50": float(accuracy_score(y, predictions)),
        "precision_at_0_50": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall_at_0_50": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_at_0_50": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "number_of_samples": int(len(y)),
        "number_of_folds": N_SPLITS,
    }

    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "metrics.csv", index=False
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Human", "AI"],
        digits=4,
        zero_division=0,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    oof = pd.DataFrame({
        "actual_label": y,
        "ai_probability": oof_probabilities,
        "predicted_label": predictions,
    })
    oof["incorrect"] = oof["actual_label"] != oof["predicted_label"]
    oof.to_csv(OUTPUT_DIR / "out_of_fold_predictions.csv", index=False)

    print("\nFinal cross-validation results")
    print("------------------------------")
    print(f"Mean AUROC:   {mean_auc:.6f} +/- {std_auc:.6f}")
    print(f"Pooled OOF:   {pooled_auc:.6f}")
    print(f"Accuracy:     {metrics['accuracy_at_0_50']:.4f}")
    print(f"Precision:    {metrics['precision_at_0_50']:.4f}")
    print(f"Recall:       {metrics['recall_at_0_50']:.4f}")
    print(f"F1-score:     {metrics['f1_at_0_50']:.4f}")
    print("\nClassification report\n")
    print(report)

    # Confusion matrix: threshold-dependent, unlike AUROC.
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Human", "AI"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Out-of-fold confusion matrix")
    save_plot("confusion_matrix.png")

    # ROC curve from out-of-fold probabilities.
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y, oof_probabilities
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"Random Forest (AUROC = {pooled_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC curve")
    ax.legend(loc="lower right")
    save_plot("roc_curve.png")

    # AUROC for each fold.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        fold_results["fold"].astype(str),
        fold_results["auroc"],
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUROC")
    ax.set_title("Cross-validation AUROC by fold")
    ax.set_ylim(0.0, 1.0)
    save_plot("fold_auroc.png")

    return metrics, oof


def save_feature_importance(model, X, y):
    """
    Calculate permutation importance on the full training data.
    Treat this as descriptive; cross-validation AUROC is the evaluation.
    """
    from sklearn.inspection import permutation_importance

    importance = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_table = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    importance_table.to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False
    )

    plot_data = importance_table.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_data["feature"], plot_data["importance_mean"])
    ax.set_xlabel("Mean decrease in AUROC after permutation")
    ax.set_title("Permutation feature importance")
    save_plot("feature_importance.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_FILE}. Place it beside this script."
        )

    train_data = pd.read_csv(TRAIN_FILE)

    required_columns = {"text", "label"}
    if not required_columns.issubset(train_data.columns):
        raise ValueError(
            "train.csv must contain the columns 'text' and 'label'."
        )

    if train_data["text"].isna().any() or train_data["label"].isna().any():
        raise ValueError("Training text and labels must not contain missing values.")

    y = train_data["label"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Labels must contain 0 (human) and 1 (AI).")

    print(f"Training passages: {len(train_data):,}")
    print("Extracting final feature set...")
    X = make_feature_matrix(train_data["text"])
    print(f"Features extracted: {len(FEATURE_NAMES)}")

    print("\nSelected model parameters")
    print("-------------------------")
    print("Algorithm: Random Forest")
    print("n_estimators: 400")
    print("max_depth: 10")
    print("min_samples_leaf: 5")
    print("class_weight: balanced")
    print(f"random_state: {RANDOM_STATE}")
    print("n_jobs: -1")
    print(f"Cross-validation: {N_SPLITS}-fold stratified")

    metrics, oof = evaluate_model(X, y)

    # Fit the selected model on all labeled training passages.
    print("\nFitting final model on all labeled training data...")
    final_model = build_model()
    final_model.fit(X, y)

    joblib.dump(
        {
            "model": final_model,
            "feature_names": FEATURE_NAMES,
            "mattr_window": MATTR_WINDOW,
        },
        OUTPUT_DIR / "tide_final_model.joblib",
    )

    save_feature_importance(final_model, X, y)

    # Predict test data if test.csv is available.
    if TEST_FILE.exists():
        test_data = pd.read_csv(TEST_FILE)
        if "text" not in test_data.columns:
            raise ValueError("test.csv must contain a 'text' column.")

        test_X = make_feature_matrix(test_data["text"])
        test_probabilities = final_model.predict_proba(test_X)[:, 1]

        submission = pd.DataFrame({
            "id": (
                test_data["id"].to_numpy()
                if "id" in test_data.columns
                else np.arange(len(test_data))
            ),
            "label": test_probabilities,
        })
        submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
        print(f"Submission saved: {OUTPUT_DIR / 'submission.csv'}")

    # Save the exact configuration used for this run.
    configuration = {
        "model": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 400,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "mattr_window": MATTR_WINDOW,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "features": FEATURE_NAMES,
        "validation_metrics": metrics,
    }

    (OUTPUT_DIR / "model_configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    print(f"\nAll files saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
