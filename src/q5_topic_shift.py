"""
Q5 improvement experiment: does removing topic information help on unseen topics?

Run from the repository root:
    python src/q5_topic_shift.py   (run src/topic_subgroups.py first to fix K)

The leave-one-topic-out evaluation uses ONLY the 8,000-row training split,
so the fixed 2,000-row validation set is never used for training or for choosing the
configuration. Topics are the same K-topic clustering as src/topic_subgroups.py (Q4.2):
fitted on the 8,000 training texts with the same settings and seed.

This script is self-contained; it does not touch the baseline outputs (src/baseline.py).
It only writes:
    results/q5_topic_shift/val_comparison.csv  fixed 2,000-row validation set (reporting only)
    results/q5_topic_shift/topic_holdout.csv   leave-one-topic-out on the 8,000 training rows
    results/q5_topic_shift/topic_clusters.csv  size and label balance of each topic cluster
    submission.csv                             final Kaggle submission (selected configuration)

Configurations (all Linear SVM, C=1, same TF-IDF settings as src/baseline.py):
    baseline       word TF-IDF, default sklearn tokenizer (= main pipeline)
    ws_word        word TF-IDF, whitespace tokens (punctuation kept)
    masked         content words replaced by "w"; only the TOP_K most frequent
                   training words (mostly function words) are kept; whitespace tokens, 1-3 grams
    masked+ws_word features of "masked" and "ws_word" stacked side by side
"""

import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# ==========================================
# 0. Config
# ==========================================

SEED = 42
DATA_DIR = "data"
OUT_DIR = os.path.join("results", "q5_topic_shift")
SUBMISSION_PATH = "submission.csv"

K_SELECTION_PATH = os.path.join("results", "topic_analysis", "k_selection.csv")
DEFAULT_N_TOPICS = 11  # used only if src/topic_subgroups.py has not been run
TOP_K = 300           # words kept by the masking step
SVM_C = 1             # best C for Linear SVM in the main hyperparameter search

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 1. Data (identical split to src/baseline.py)
# ==========================================

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

X = train["text"].fillna("")
y = train["label"].values
X_test = test["text"].fillna("")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ==========================================
# 2. Preprocessing (same as src/baseline.py) + masking
# ==========================================

URL_PATTERN = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
URL_PATTERN_WWW = r"www\.[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
WORD = re.compile(r"[a-z']+")


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(URL_PATTERN, "<url>", text)
    text = re.sub(URL_PATTERN_WWW, "<url>", text)
    return text


def build_keep_set(texts):
    # TOP_K most frequent words, counted on the TRAINING texts only (no validation/test leakage)
    counts = Counter(w for t in texts for w in WORD.findall(preprocess_text(t)))
    return {w for w, _ in counts.most_common(TOP_K)}


def mask_texts(texts, keep):
    # Replace every word outside `keep` with "w"; punctuation, digits and spacing are unchanged
    return [WORD.sub(lambda m: m.group(0) if m.group(0) in keep else "w", preprocess_text(t))
            for t in texts]


def word_vectorizer(whitespace):
    kwargs = dict(preprocessor=preprocess_text, analyzer="word", ngram_range=(1, 2),
                  min_df=2, max_df=0.95, max_features=100000, sublinear_tf=True)
    if whitespace:
        kwargs["token_pattern"] = r"\S+"
    return TfidfVectorizer(**kwargs)


def masked_vectorizer():
    # Input is already preprocessed + masked
    return TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 3), min_df=2,
                           max_df=0.95, max_features=100000, sublinear_tf=True)


CONFIGS = {
    "baseline": {"word": False, "ws": False, "masked": False},
    "ws_word": {"word": True, "ws": True, "masked": False},
    "masked": {"word": False, "ws": False, "masked": True},
    "masked+ws_word": {"word": True, "ws": True, "masked": True},
}


def fit_predict(config, X_fit, y_fit, X_eval_list):
    """Fit vectorizer(s) + Linear SVM on X_fit; return P(AI) scores for each set in X_eval_list."""
    c = CONFIGS[config]
    blocks_fit, transforms = [], []

    if c["word"] or not c["masked"]:
        vec = word_vectorizer(whitespace=c["ws"])
        blocks_fit.append(vec.fit_transform(X_fit))
        transforms.append(lambda s, v=vec: v.transform(s))

    if c["masked"]:
        keep = build_keep_set(X_fit)
        mvec = masked_vectorizer()
        blocks_fit.append(mvec.fit_transform(mask_texts(X_fit, keep)))
        transforms.append(lambda s, v=mvec, k=keep: v.transform(mask_texts(s, k)))

    clf = LinearSVC(C=SVM_C, random_state=SEED)
    start = time.time()
    clf.fit(hstack(blocks_fit).tocsr(), y_fit)
    fit_time = time.time() - start

    outs = []
    for X_eval in X_eval_list:
        feats = hstack([tf(X_eval) for tf in transforms]).tocsr()
        outs.append(expit(clf.decision_function(feats)))
    return outs, fit_time


# ==========================================
# 3. Fixed 2,000-row validation set (reporting only, not used for selection)
# ==========================================

print("Fixed validation split:")
val_rows = []
for name in CONFIGS:
    start = time.time()
    (p_tr, p_va), fit_time = fit_predict(name, X_train, y_train, [X_train, X_val])
    val_rows.append({
        "Config": name,
        "Train AUROC": roc_auc_score(y_train, p_tr),
        "Val AUROC": roc_auc_score(y_val, p_va),
        "SVM fit time (s)": fit_time,
        "Total time (s)": time.time() - start,
    })
    print(f"  {name:15s} val AUROC {val_rows[-1]['Val AUROC']:.4f}")
pd.DataFrame(val_rows).to_csv(os.path.join(OUT_DIR, "val_comparison.csv"), index=False)

# ==========================================
# 4. Leave-one-topic-out on the 8,000 training rows only
# ==========================================
# Same topic clustering as src/topic_subgroups.py: TF-IDF (stop words removed) -> 100-d LSA
# -> L2-normalise -> KMeans, fitted on the 8,000 training texts. Labels are not used.
# For each topic k: train on the other topics, evaluate AUROC on topic k.
# The 2,000 validation rows are not used anywhere in this section.

if os.path.exists(K_SELECTION_PATH):
    k_sel = pd.read_csv(K_SELECTION_PATH)
    N_TOPICS = int(k_sel.loc[k_sel["Chosen"], "K"].iloc[0])
else:
    N_TOPICS = DEFAULT_N_TOPICS
print(f"\nUsing K = {N_TOPICS} topics")

X_tr = X_train.reset_index(drop=True)
y_tr = y_train

topic_vec = TfidfVectorizer(preprocessor=preprocess_text, stop_words="english",
                            min_df=5, max_df=0.5)
T_tr = topic_vec.fit_transform(X_tr)
terms = np.array(topic_vec.get_feature_names_out())
Z = TruncatedSVD(n_components=100, random_state=SEED).fit_transform(T_tr)
Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
topics = KMeans(n_clusters=N_TOPICS, random_state=SEED, n_init=10).fit_predict(Z)

cluster_rows = []
for k in range(N_TOPICS):
    idx = topics == k
    mean_w = np.asarray(T_tr[idx].mean(axis=0)).ravel()
    cluster_rows.append({
        "Topic": k,
        "Count (train)": int(idx.sum()),
        "AI fraction": y_tr[idx].mean(),
        "Top terms": ", ".join(terms[np.argsort(mean_w)[::-1][:6]]),
    })
pd.DataFrame(cluster_rows).to_csv(os.path.join(OUT_DIR, "topic_clusters.csv"), index=False)

print("Leave-one-topic-out (training split only):")
holdout_rows = []
for name in CONFIGS:
    aucs = []
    for k in range(N_TOPICS):
        held = topics == k
        (p_held,), _ = fit_predict(name, X_tr[~held], y_tr[~held], [X_tr[held]])
        aucs.append(roc_auc_score(y_tr[held], p_held))
    row = {"Config": name, "Mean AUROC": np.mean(aucs), "Min AUROC": np.min(aucs)}
    row.update({f"Topic {k}": a for k, a in enumerate(aucs)})
    holdout_rows.append(row)
    print(f"  {name:15s} mean {row['Mean AUROC']:.4f}  min {row['Min AUROC']:.4f}")
holdout_df = pd.DataFrame(holdout_rows)
holdout_df.to_csv(os.path.join(OUT_DIR, "topic_holdout.csv"), index=False)

# Select the configuration with the best mean leave-one-topic-out AUROC
FINAL_CONFIG = holdout_df.loc[holdout_df["Mean AUROC"].idxmax(), "Config"]
print(f"Selected configuration: {FINAL_CONFIG}")

# ==========================================
# 5. Final model on all 10,000 rows -> submission.csv
# ==========================================

print(f"\nRetraining '{FINAL_CONFIG}' on all labeled data...")
(probabilities,), _ = fit_predict(FINAL_CONFIG, X, y, [X_test])

submission = pd.DataFrame({"id": test["id"], "label": probabilities})
assert len(submission) == 3000
assert submission["id"].is_unique
assert submission["label"].between(0, 1).all()
assert set(submission["id"]) == set(test["id"])
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved {SUBMISSION_PATH}")
