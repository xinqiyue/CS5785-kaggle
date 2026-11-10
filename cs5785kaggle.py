import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy.special import expit

# ==========================================
# 0. Config
# ==========================================

SEED = 42
DATA_DIR = "data"
RESULTS_DIR = "results"

# Set to a model name to override automatic selection (None = best validation AUROC)
FINAL_MODEL_OVERRIDE = None

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. Load Data
# ==========================================

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

unlabeled_path = os.path.join(DATA_DIR, "unlabeled.csv")
unlabeled = pd.read_csv(unlabeled_path) if os.path.exists(unlabeled_path) else None

X = train["text"].fillna("")
y = train["label"]

print(f"Device: {DEVICE}")
print(f"Training examples: {len(train)}")
print(f"Test examples: {len(test)}")
print(f"Unlabeled examples: {0 if unlabeled is None else len(unlabeled)}")
print(f"Label distribution:\n{y.value_counts()}")

# ==========================================
# 2. Preprocessing
# ==========================================

URL_PATTERN = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
URL_PATTERN_WWW = r"www\.[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"

def preprocess_text(text):
    # Lowercase, replace URLs with <url>; tokenization (split on whitespace) is done by TF-IDF
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(URL_PATTERN, "<url>", text)
    text = re.sub(URL_PATTERN_WWW, "<url>", text)
    return text

def create_vectorizer():
    # Word TF-IDF on whitespace tokens (punctuation kept), min_df=2
    # return TfidfVectorizer(
    #     preprocessor=preprocess_text,
    #     token_pattern=r"\S+",
    #     min_df=2
    # )
    return TfidfVectorizer(
        preprocessor=preprocess_text,
        # token_pattern=r"\S+",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=100000,
        sublinear_tf=True
    )

# ==========================================
# 3. Train/Validation Split
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

y_train = y_train.values
y_val = y_val.values

print(f"\nTrain split: {len(X_train)}, Validation split: {len(X_val)}")

# ==========================================
# 4. Helpers
# ==========================================

def create_pipeline(classifier):
    return Pipeline([
        ("tfidf", create_vectorizer()),
        ("classifier", classifier)
    ])

def get_probs(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Sigmoid preserves SVM rankings for AUROC (threshold 0 -> 0.5).
    # These are not calibrated probabilities.
    return expit(model.decision_function(X))

def safe_auc(y_true, probs):
    return roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan

# ---------- MLP ----------

class TfidfMLP(nn.Module):
    # TF-IDF -> Linear(hidden) -> ReLU -> Dropout -> Linear(1)
    def __init__(self, n_features, hidden=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def sparse_batches(X, y=None, batch_size=128, shuffle=False):
    # Yield dense torch batches from a sparse matrix (only one batch dense at a time)
    idx = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        xb = torch.tensor(X[b].toarray(), dtype=torch.float32, device=DEVICE)
        if y is None:
            yield xb
        else:
            yield xb, torch.tensor(y[b], dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def mlp_predict(model, X):
    model.eval()
    probs = [torch.sigmoid(model(xb)).cpu().numpy() for xb in sparse_batches(X, batch_size=512)]
    return np.concatenate(probs)

def train_mlp(X_tr, y_tr, X_va=None, y_va=None, hidden=256, dropout=0.5, lr=1e-3,
              weight_decay=1e-5, max_epochs=20, patience=3):
    # With validation data: early stopping on val AUROC.
    # Without: train for exactly max_epochs (used for the final full-data retrain).
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = TfidfMLP(X_tr.shape[1], hidden, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_state, bad_epochs, history = 0.0, max_epochs, None, 0, []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in sparse_batches(X_tr, y_tr, shuffle=True):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        train_loss = total / X_tr.shape[0]

        if X_va is None:
            history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": np.nan})
            continue

        val_auc = roc_auc_score(y_va, mlp_predict(model, X_va))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auc})
        print(f"  Epoch {epoch:2d}  train loss {train_loss:.4f}  val AUROC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, best_epoch, bad_epochs = val_auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, best_epoch, history

# ==========================================
# 5. Hyperparameter Search (sklearn models)
# ==========================================

SEARCH_SPACE = {
    "Logistic Regression": {
        "param": "C",
        "values": [0.1, 0.3, 1, 3, 10, 30, 100],
        "build": lambda v: LogisticRegression(C=v, max_iter=2000, random_state=SEED)
    },
    "Linear SVM": {
        "param": "C",
        "values": [0.01, 0.03, 0.1, 0.3, 1, 3],
        "build": lambda v: LinearSVC(C=v, random_state=SEED)
    },
    "Naive Bayes": {
        "param": "alpha",
        "values": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        "build": lambda v: MultinomialNB(alpha=v)
    }
}

# TF-IDF is identical for all candidates, so fit it once for the search
vectorizer = create_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print(f"\nTF-IDF features: {X_train_vec.shape[1]}")

search_rows = []
best_params = {}

for name, space in SEARCH_SPACE.items():

    print(f"\nSearching {space['param']} for {name}...")

    for value in space["values"]:
        clf = space["build"](value)
        clf.fit(X_train_vec, y_train)
        auc = roc_auc_score(y_val, get_probs(clf, X_val_vec))
        print(f"  {space['param']}={value}: val AUROC {auc:.4f}")
        search_rows.append({
            "Model": name,
            "Param": space["param"],
            "Value": value,
            "Val AUROC": auc
        })

    model_rows = [r for r in search_rows if r["Model"] == name]
    best_params[name] = max(model_rows, key=lambda r: r["Val AUROC"])["Value"]
    print(f"Best {space['param']} for {name}: {best_params[name]}")

search_df = pd.DataFrame(search_rows)
search_df.to_csv(os.path.join(RESULTS_DIR, "hyperparameter_search.csv"), index=False)

# ==========================================
# 6. Hyperparameter Search (MLP)
# ==========================================

MLP_SEARCH_SPACE = {
    "hidden": [128, 256, 512],
    "dropout": [0.3, 0.5]
}

mlp_rows = []
mlp_histories = []
best_mlp = None

for hidden in MLP_SEARCH_SPACE["hidden"]:
    for dropout in MLP_SEARCH_SPACE["dropout"]:

        print(f"\nTraining MLP with hidden={hidden}, dropout={dropout}...")

        start = time.time()
        model, auc, best_epoch, history = train_mlp(
            X_train_vec, y_train, X_val_vec, y_val,
            hidden=hidden, dropout=dropout
        )
        elapsed = time.time() - start

        mlp_rows.append({
            "hidden": hidden,
            "dropout": dropout,
            "best_epoch": best_epoch,
            "Val AUROC": auc,
            "Train time (s)": elapsed
        })
        for h in history:
            mlp_histories.append({"hidden": hidden, "dropout": dropout, **h})

        if best_mlp is None or auc > best_mlp["auc"]:
            best_mlp = {"model": model, "auc": auc, "hidden": hidden, "dropout": dropout,
                        "best_epoch": best_epoch, "time": elapsed}

pd.DataFrame(mlp_rows).to_csv(os.path.join(RESULTS_DIR, "mlp_search.csv"), index=False)
pd.DataFrame(mlp_histories).to_csv(os.path.join(RESULTS_DIR, "mlp_history.csv"), index=False)

print(f"\nBest MLP: hidden={best_mlp['hidden']}, dropout={best_mlp['dropout']}, "
      f"epochs={best_mlp['best_epoch']}, val AUROC {best_mlp['auc']:.4f}")

# ==========================================
# 7. Model Comparison (best hyperparameters)
# ==========================================

results = []
val_probs = {}
fitted_models = {}

for name, space in SEARCH_SPACE.items():

    print(f"\nTraining {name} ({space['param']}={best_params[name]})...")

    clf = space["build"](best_params[name])
    start = time.time()
    clf.fit(X_train_vec, y_train)
    elapsed = time.time() - start

    train_probs = get_probs(clf, X_train_vec)
    val_probs[name] = get_probs(clf, X_val_vec)
    fitted_models[name] = clf

    results.append({
        "Model": name,
        "Best params": f"{space['param']}={best_params[name]}",
        "Train AUROC": roc_auc_score(y_train, train_probs),
        "Val AUROC": roc_auc_score(y_val, val_probs[name]),
        "Val Accuracy": accuracy_score(y_val, (val_probs[name] >= 0.5).astype(int)),
        "Train time (s)": elapsed
    })

mlp_train_probs = mlp_predict(best_mlp["model"], X_train_vec)
val_probs["MLP"] = mlp_predict(best_mlp["model"], X_val_vec)

results.append({
    "Model": "MLP",
    "Best params": f"hidden={best_mlp['hidden']}, dropout={best_mlp['dropout']}, epochs={best_mlp['best_epoch']}",
    "Train AUROC": roc_auc_score(y_train, mlp_train_probs),
    "Val AUROC": roc_auc_score(y_val, val_probs["MLP"]),
    "Val Accuracy": accuracy_score(y_val, (val_probs["MLP"] >= 0.5).astype(int)),
    "Train time (s)": best_mlp["time"]
})

results_df = pd.DataFrame(results).sort_values("Val AUROC", ascending=False)
results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

print("\nModel Comparison:")
print(results_df.round(4).to_string(index=False))

# ROC curves
plt.figure(figsize=(8, 6))
for name, probs in val_probs.items():
    fpr, tpr, _ = roc_curve(y_val, probs)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUROC = {roc_auc_score(y_val, probs):.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC Curves")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==========================================
# 8. Final Model Selection
# ==========================================

best_name = FINAL_MODEL_OVERRIDE or results_df.iloc[0]["Model"]
best_auc = roc_auc_score(y_val, val_probs[best_name])

print(f"\nBest Model: {best_name}")
print(f"Best Validation AUROC: {best_auc:.4f}")

# ==========================================
# 9. Effect of Training Data Size
# ==========================================
# Logistic Regression (best C); TF-IDF re-fit on each subset, same validation set.

print("\nComputing learning curve...")

sizes = [500, 1000, 2000, 5000, len(X_train)]
curve = []

for n in sizes:
    if n == len(X_train):
        X_sub, y_sub = X_train, y_train
    else:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=n, stratify=y_train, random_state=SEED
        )

    pipeline = create_pipeline(SEARCH_SPACE["Logistic Regression"]["build"](best_params["Logistic Regression"]))
    pipeline.fit(X_sub, y_sub)

    curve.append({
        "Training rows": n,
        "Train AUROC": roc_auc_score(y_sub, get_probs(pipeline, X_sub)),
        "Val AUROC": roc_auc_score(y_val, get_probs(pipeline, X_val))
    })
    print(f"  n={n}: val AUROC {curve[-1]['Val AUROC']:.4f}")

curve_df = pd.DataFrame(curve)
curve_df.to_csv(os.path.join(RESULTS_DIR, "learning_curve.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(curve_df["Training rows"], curve_df["Train AUROC"], "o-", label="Train AUROC")
plt.plot(curve_df["Training rows"], curve_df["Val AUROC"], "s-", label="Validation AUROC")
plt.xscale("log")
plt.xticks(sizes, sizes)
plt.xlabel("Training rows")
plt.ylabel("AUROC")
plt.title("Learning Curve (Logistic Regression)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(os.path.join(RESULTS_DIR, "learning_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# ==========================================
# 11. Error Analysis (final model, validation set)
# ==========================================

err_df = train.loc[X_val.index, ["id", "text", "label"]].copy()
err_df["pred_prob"] = val_probs[best_name]
err_df["pred_label"] = (err_df["pred_prob"] >= 0.5).astype(int)
err_df["n_words"] = err_df["text"].str.split().str.len()

err_df.to_csv(os.path.join(RESULTS_DIR, "val_predictions.csv"), index=False)

# ---------- 11.1 Performance by text length ----------
# Bins by word count: <150, 150-249, >=250

err_df["length_bin"] = pd.cut(
    err_df["n_words"], [0, 150, 250, np.inf], right=False, labels=["<150", "150-249", ">=250"]
)

length_rows = []
for bin_name, g in err_df.groupby("length_bin", observed=True):
    length_rows.append({
        "Length bin": bin_name,
        "Count": len(g),
        "AI fraction": g["label"].mean(),
        "Val AUROC": safe_auc(g["label"], g["pred_prob"]),
        "Val Accuracy": accuracy_score(g["label"], g["pred_label"])
    })

length_df = pd.DataFrame(length_rows)
length_df.to_csv(os.path.join(RESULTS_DIR, "performance_by_length.csv"), index=False)

print(f"\nPerformance by text length ({best_name}):")
print(length_df.round(4).to_string(index=False))

# ---------- 11.2 False positives / false negatives ----------
# Threshold 0.5. FP = human predicted AI, FN = AI predicted human; most confident errors first.

fp = err_df[(err_df["label"] == 0) & (err_df["pred_label"] == 1)].sort_values("pred_prob", ascending=False).head(5)
fn = err_df[(err_df["label"] == 1) & (err_df["pred_label"] == 0)].sort_values("pred_prob").head(5)

errors = pd.concat([fp.assign(error="FP"), fn.assign(error="FN")])
errors["snippet"] = errors["text"].str.replace(r"\s+", " ", regex=True).str[:200]
errors[["id", "error", "pred_prob", "n_words", "snippet"]].to_csv(
    os.path.join(RESULTS_DIR, "error_examples.csv"), index=False
)

print(f"\nFalse positives: {int(((err_df['label'] == 0) & (err_df['pred_label'] == 1)).sum())}, "
      f"False negatives: {int(((err_df['label'] == 1) & (err_df['pred_label'] == 0)).sum())}")

# ---------- 11.3 Failure patterns ----------
# Average simple style statistics per outcome (TP / FN / TN / FP)

err_df["outcome"] = np.where(
    err_df["label"] == 1,
    np.where(err_df["pred_label"] == 1, "TP", "FN"),
    np.where(err_df["pred_label"] == 1, "FP", "TN")
)
err_df["sent_len_std"] = err_df["text"].apply(
    lambda t: np.std([len(s.split()) for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()] or [0])
)
err_df["punct_per_word"] = err_df["text"].str.count(r"[,;:()\"'!?-]") / err_df["n_words"]
err_df["has_newline"] = err_df["text"].str.contains("\n")

patterns_df = err_df.groupby("outcome")[["n_words", "sent_len_std", "punct_per_word", "has_newline"]].mean()
patterns_df.insert(0, "Count", err_df["outcome"].value_counts())
patterns_df.to_csv(os.path.join(RESULTS_DIR, "failure_patterns.csv"))

print("\nFailure patterns:")
print(patterns_df.round(3).to_string())

# ==========================================
# 12. Retrain Best Model on Full Dataset
# ==========================================

print(f"\nRetraining {best_name} on full dataset...")

X_test = test["text"].fillna("")

if best_name == "MLP":
    # No validation set left for early stopping: train for the best epoch count found above
    full_vectorizer = create_vectorizer()
    X_full_vec = full_vectorizer.fit_transform(X)
    final_mlp, _, _, _ = train_mlp(
        X_full_vec, y.values,
        hidden=best_mlp["hidden"], dropout=best_mlp["dropout"],
        max_epochs=best_mlp["best_epoch"]
    )
    joblib.dump(full_vectorizer, os.path.join(RESULTS_DIR, "best_model_vectorizer.joblib"))
    torch.save(final_mlp.state_dict(), os.path.join(RESULTS_DIR, "best_model_mlp.pt"))
    probabilities = mlp_predict(final_mlp, full_vectorizer.transform(X_test))
else:
    best_model = create_pipeline(SEARCH_SPACE[best_name]["build"](best_params[best_name]))
    best_model.fit(X, y)
    joblib.dump(best_model, os.path.join(RESULTS_DIR, "best_model.joblib"))
    probabilities = get_probs(best_model, X_test)

# ==========================================
# 13. Generate Kaggle Predictions
# ==========================================

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