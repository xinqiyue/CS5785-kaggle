# CS5785 Kaggle: Human or AI?

Binary classification of text passages as human-written (0) or AI-generated (1), evaluated by AUROC.

## Approach

1. **Baseline (`src/baseline.py`).** Word TF-IDF (unigrams + bigrams, 100k features) with four classifiers:
   Naive Bayes, Logistic Regression, Linear SVM and an MLP. Hyperparameters are tuned on a fixed
   stratified 2,000-row validation split. The selected model is a Linear SVM (C=1, validation AUROC 0.9422).
2. **Topic analysis (`src/topic_subgroups.py`).** Training texts are clustered into topics
   (TF-IDF → LSA → KMeans, K chosen by silhouette score; K=11), and validation AUROC is reported per topic.
3. **Final model (`src/q5_topic_shift.py`).** Linear SVM on two concatenated feature blocks:
   whitespace-token word TF-IDF (content) and a masked representation that keeps only the 300 most
   frequent words plus punctuation (style). Configurations are selected by leave-one-topic-out AUROC
   on the 8,000 training rows only.

| Model | Validation AUROC | Leave-one-topic-out mean AUROC |
|---|---|---|
| Baseline (Linear SVM, word TF-IDF) | 0.9423 | 0.8931 |
| Final (masked + whitespace word TF-IDF) | 0.9670 | 0.9301 |

## Repository structure

```
data/                      train.csv, test.csv, unlabeled.csv, sample_submission.csv, train_starter_2000.csv
src/
  baseline.py              baseline pipeline: model comparison, learning curve, error analysis
  topic_subgroups.py       topic clustering and validation AUROC per topic
  q5_topic_shift.py        improvement experiment and final model
results/                   tables and figures from baseline.py
  topic_analysis/          outputs of topic_subgroups.py
  q5_topic_shift/          outputs of q5_topic_shift.py
submission_baseline.csv    Kaggle predictions from the baseline
submission.csv             final Kaggle predictions
Harsh/                     a teammate's separate experiments (Random Forest on linguistic features)
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

Run all commands from the repository root:

```bash
python src/baseline.py          # -> submission_baseline.csv, results/
python src/topic_subgroups.py   # -> results/topic_analysis/
python src/q5_topic_shift.py    # -> submission.csv, results/q5_topic_shift/
```

`src/q5_topic_shift.py` reads the chosen number of topics from `results/topic_analysis/k_selection.csv`,
so run `src/topic_subgroups.py` first. Approximate CPU runtimes: a few minutes for `baseline.py`,
under 30 seconds for `topic_subgroups.py`, about 5 minutes for `q5_topic_shift.py`.

## Reproducibility notes

- All scripts use `SEED = 42` for the train/validation split, KMeans, TruncatedSVD, LinearSVC,
  NumPy and PyTorch.
- The 2,000-row validation split is the same in every script (stratified, `random_state=42`).
- Topic clustering, the masking vocabulary and model selection in `q5_topic_shift.py` use only the
  8,000 training rows; the validation split is used for reporting only.
- Final models are retrained on all 10,000 labeled rows before predicting `data/test.csv`.
- Kaggle test labels are never used. Small numerical differences across machines can change
  probabilities slightly, but not the rankings in a meaningful way.

## AI use statement

We used Claude as an assistant for brainstorming, coding and writing. It suggested several ideas for
improving generalization, from which we chose leave-one-topic-out evaluation and masked style features.
It helped write the topic analysis and improvement scripts, and helped edit the report text and LaTeX
tables. We ran all experiments, checked that every script runs, checked all reported results, and changed
the design when needed. No AI tool was used to label or predict test passages, and no hidden test labels
were retrieved.
