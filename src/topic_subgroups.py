"""
Generalization and error analysis: validation performance by topic subgroup.

Run from the repository root:
    python src/topic_subgroups.py

Model analysed: the Linear SVM selected in Q1 (C=1, word TF-IDF with the same settings as
cs5785kaggle.py), trained on the fixed 8,000-row training split.

Topics are found without labels:
    TF-IDF (stop words removed) -> 100-d LSA (TruncatedSVD) -> L2-normalise -> KMeans.
Topics are fitted on the 8,000 training passages only; validation passages are then
assigned to the nearest topic, so the validation set is never used to form the topics.
We try K in {5, 7, 9, 11, 13, 15} and keep the K with the highest silhouette score (cosine) on the
training passages.

Outputs (new folder; nothing from the main pipeline is overwritten):
    results/topic_analysis/k_selection.csv       silhouette score for each K
    results/topic_analysis/topic_subgroups.csv   Subgroup / Count / Validation AUROC for the chosen K
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# ==========================================
# 0. Config
# ==========================================

SEED = 42
DATA_DIR = "data"
OUT_DIR = os.path.join("results", "topic_analysis")
K_CANDIDATES = [5, 7, 9, 11, 13, 15]
N_COMPONENTS = 100     # LSA dimensions used for clustering
N_TOP_TERMS = 6        # terms used to describe each topic
SVM_C = 1              # best C for Linear SVM in the main hyperparameter search

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# 1. Data (identical split to cs5785kaggle.py)
# ==========================================

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
X = train["text"].fillna("")
y = train["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ==========================================
# 2. Preprocessing (same as cs5785kaggle.py)
# ==========================================

URL_PATTERN = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"
URL_PATTERN_WWW = r"www\.[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)"


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(URL_PATTERN, "<url>", text)
    text = re.sub(URL_PATTERN_WWW, "<url>", text)
    return text


# ==========================================
# 3. Final selected model -> validation scores
# ==========================================

vectorizer = TfidfVectorizer(
    preprocessor=preprocess_text, analyzer="word", ngram_range=(1, 2),
    min_df=2, max_df=0.95, max_features=100000, sublinear_tf=True
)
svm = LinearSVC(C=SVM_C, random_state=SEED)
svm.fit(vectorizer.fit_transform(X_train), y_train)
val_scores = svm.decision_function(vectorizer.transform(X_val))
print(f"Overall validation AUROC: {roc_auc_score(y_val, val_scores):.4f}")

# ==========================================
# 4. Topic clustering (labels not used)
# ==========================================

topic_vec = TfidfVectorizer(preprocessor=preprocess_text, stop_words="english",
                            min_df=5, max_df=0.5)
T_train = topic_vec.fit_transform(X_train)
T_val = topic_vec.transform(X_val)
terms = np.array(topic_vec.get_feature_names_out())

svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)


def normalise(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


Z_train = normalise(svd.fit_transform(T_train))
Z_val = normalise(svd.transform(T_val))

k_rows, models = [], {}
for k in K_CANDIDATES:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z_train)
    sil = silhouette_score(Z_train, km.labels_, metric="cosine")
    models[k] = km
    k_rows.append({"K": k, "Silhouette (cosine)": sil, "Inertia": km.inertia_})
    print(f"K={k:2d}: silhouette {sil:.4f}")

k_df = pd.DataFrame(k_rows)
best_k = int(k_df.loc[k_df["Silhouette (cosine)"].idxmax(), "K"])
k_df["Chosen"] = k_df["K"] == best_k
k_df.to_csv(os.path.join(OUT_DIR, "k_selection.csv"), index=False)
print(f"Chosen K = {best_k}")

# ==========================================
# 5. Validation AUROC per topic
# ==========================================

km = models[best_k]
train_topic = km.labels_
val_topic = km.predict(Z_val)

rows = []
for k in range(best_k):
    # Describe each topic by its highest-weight terms in the TRAINING passages
    mean_w = np.asarray(T_train[train_topic == k].mean(axis=0)).ravel()
    top_terms = ", ".join(terms[np.argsort(mean_w)[::-1][:N_TOP_TERMS]])

    idx = val_topic == k
    labels = y_val[idx]
    auc = roc_auc_score(labels, val_scores[idx]) if len(np.unique(labels)) > 1 else np.nan
    rows.append({
        "Subgroup": f"Topic {k}: {top_terms}",
        "Count": int(idx.sum()),
        "Validation AUROC": auc,
    })

sub_df = pd.DataFrame(rows).sort_values("Validation AUROC", ascending=False)
sub_df.to_csv(os.path.join(OUT_DIR, "topic_subgroups.csv"), index=False)

print("\nValidation AUROC by topic:")
print(sub_df.round(4).to_string(index=False))
