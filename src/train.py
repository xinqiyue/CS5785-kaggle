
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.special import expit

# ==========================================
# 1. Load Data
# ==========================================

os.makedirs("results", exist_ok=True)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X = train["text"].fillna("")
y = train["label"]

print(f"Training examples: {len(train)}")
print(f"Test examples: {len(test)}")
print(f"Label distribution:\n{y.value_counts()}")

# ==========================================
# 2. Train/Validation Split
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 3. Define Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=2000
    ),
    "Linear SVM": LinearSVC(
        C=1.0,
        random_state=42
    ),
    "Naive Bayes": MultinomialNB(
        alpha=1.0
    )
}

def create_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=100000,
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

# ==========================================
# 4. Train and Evaluate
# ==========================================

results = []
best_auc = -1
best_name = None

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = create_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    scores = get_scores(pipeline, X_val)

    auc = roc_auc_score(y_val, scores)
    accuracy = accuracy_score(
        y_val, pipeline.predict(X_val)
    )

    print(f"AUROC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    results.append({
        "Model": name,
        "AUROC": auc,
        "Accuracy": accuracy
    })

    if auc > best_auc:
        best_auc = auc
        best_name = name

# ==========================================
# 5. Save Comparison Results
# ==========================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "AUROC", ascending=False
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print(f"\nBest Model: {best_name}")
print(f"Best Validation AUROC: {best_auc:.4f}")

# ==========================================
# 6. Retrain Best Model on Full Dataset
# ==========================================

print("\nRetraining best model on full dataset...")

best_model = create_pipeline(models[best_name])
best_model.fit(X, y)

joblib.dump(best_model, "results/best_model.joblib")

# ==========================================
# 7. Generate Kaggle Predictions
# ==========================================

X_test = test["text"].fillna("")

if hasattr(best_model, "predict_proba"):
    probabilities = best_model.predict_proba(X_test)[:, 1]
else:
    # Sigmoid preserves SVM rankings for AUROC.
    # These are not calibrated probabilities.
    probabilities = expit(best_model.decision_function(X_test))

submission = pd.DataFrame({
    "id": test["id"],
    "label": probabilities
})

# Verify Kaggle submission format
assert len(submission) == 3000
assert submission["id"].is_unique
assert submission["label"].between(0, 1).all()
assert submission.columns.tolist() == ["id", "label"]
assert set(submission["id"]) == set(test["id"])

submission.to_csv("submission.csv", index=False)

print("\nSubmission saved to submission.csv")
print(submission.head())
print("\nTraining complete!")
