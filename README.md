# SMS Spam Detection (NLP Practice)

A small end-to-end NLP project to refresh core text classification concepts using a classic dataset: SMS messages labeled as **spam** or **ham**. The goal is to build a strong, interpretable baseline with careful evaluation and no data leakage.

---

## Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Approach Overview](#approach-overview)
- [Key Concepts Reviewed](#key-concepts-reviewed)
- [Pipeline Details](#pipeline-details)
  - [1) Data Loading](#1-data-loading)
  - [2) Data Cleaning](#2-data-cleaning)
  - [3) Train/Test Split (Stratified)](#3-traintest-split-stratified)
  - [4) TF-IDF Vectorization](#4-tf-idf-vectorization)
  - [5) Model Training](#5-model-training)
  - [6) Evaluation](#6-evaluation)
- [Why This Works So Well](#why-this-works-so-well)
- [Common Pitfalls (What I Avoided)](#common-pitfalls-what-i-avoided)
- [Ideas for Next Iterations](#ideas-for-next-iterations)
- [How to Run](#how-to-run)

---

## Project Goal
Build a baseline SMS spam classifier that:
- Is **simple** but **strong**
- Avoids **data leakage**
- Uses appropriate **metrics** for imbalanced classification (precision/recall/F1)
- Is easy to understand and explain

---

## Dataset
This uses Kaggle’s packaging of the **UCI SMS Spam Collection** dataset.

- **Label column**: `v1` (spam/ham)
- **Text column**: `v2` (SMS message)
- Some Kaggle versions include extra columns (e.g., `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`) that are mostly empty and should be dropped.

**Why the dataset is “easy”**: spam has strong lexical patterns (“free”, “win”, “claim”, URLs, phone numbers), so classic methods like TF-IDF + linear models perform extremely well.

---

## Approach Overview
1. Load dataset
2. Drop empty columns, rename columns
3. Light text normalization:
   - lowercase
   - replace URLs with a placeholder token (`URL`)
4. Split data using a **stratified** train/test split (preserves spam/ham ratio)
5. Vectorize text with **TF-IDF** (fit on train only)
6. Train **Logistic Regression**
7. Evaluate with accuracy + full classification report

---

## Key Concepts Reviewed
- Text preprocessing decisions (what to keep vs remove)
- Feature extraction via TF-IDF
- Why “simple baselines” can be very strong in NLP
- Class imbalance and why **accuracy is not enough**
- Precision/recall tradeoffs (especially for the spam class)
- Preventing **data leakage** by fitting vectorizers only on training data
- Stratified splitting for reliable evaluation

---

## Pipeline Details

### 1) Data Loading
Dataset is loaded using `kagglehub` from:
- `uciml/sms-spam-collection-dataset`

This is convenient for Colab-style workflows.

---

### 2) Data Cleaning
#### Dropping unused columns
Some dataset versions include columns like:
- `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`

These exist due to CSV formatting/artifacts and contain mostly `NaN`. They provide no signal, so they’re removed.

#### Renaming
- `v1` → `label`
- `v2` → `text`

---

### 3) Train/Test Split (Stratified)
Spam detection is **imbalanced** (ham is the majority). A random split can accidentally change spam/ham proportions between train and test, causing unstable metrics.

Using a **stratified split** ensures:
- Train and test sets preserve the original label distribution.

This improves *evaluation reliability* (it doesn’t “boost” the model by itself).

---

### 4) TF-IDF Vectorization
**TF-IDF** converts text into a numeric feature vector where:
- **TF** (term frequency): how often a token appears in the document
- **IDF** (inverse document frequency): downweights tokens that appear in many documents

Important details:
- The vectorizer is **fit only on the training text**.
- The test text is only **transformed**, not fit.

This prevents leakage of:
- vocabulary
- IDF statistics

---

### 5) Model Training
Trained a baseline **Logistic Regression** classifier.

Why Logistic Regression works well here:
- Handles high-dimensional sparse TF-IDF vectors effectively
- Produces strong baselines for text classification
- Interpretable (you can inspect top-weighted features)

---

### 6) Evaluation
Reported:
- **Accuracy** (useful but not sufficient)
- **Classification report** including:
  - precision
  - recall
  - F1-score

For spam detection, the spam class metrics matter most:
- **Precision (spam)**: when I say “spam”, how often am I right?
- **Recall (spam)**: how much spam did I actually catch?

Why accuracy can mislead:
- If ham dominates, a model can get high accuracy while missing lots of spam.

---

## Why This Works So Well
This dataset is strongly separable using word-level features:
- spam tends to include promotional language, urgency, rewards, and links
- ham tends to look conversational and personal

TF-IDF + a linear model is often all you need.

---

## Common Pitfalls (What I Avoided)

### ✅ Data leakage from TF-IDF
Wrong approach:
- Fit TF-IDF on the entire dataset before splitting

Correct approach:
- Split first, then fit TF-IDF **only on train**

### ✅ Non-stratified splits on imbalanced data
Without stratification:
- your train/test label ratios can drift
- metrics become noisy and less trustworthy

### ✅ Over-cleaning text
For spam detection, many “noisy” elements are actually signal:
- punctuation emphasis (!!!)
- weird casing
- phone numbers
- URLs

This project keeps preprocessing minimal and lets TF-IDF learn useful patterns.

---

## Ideas for Next Iterations
If revisiting this project, good next steps (in order):

1. **Threshold tuning**
   - Logistic Regression outputs probabilities; you can lower the threshold to increase spam recall.
2. **Class weights**
   - Penalize missing spam more heavily to improve recall.
3. **Character n-grams**
   - Often excellent for spam because they capture:
     - misspellings
     - repeated punctuation
     - phone number patterns
4. **Cross-validation**
   - More stable estimate than a single train/test split.
5. **Model interpretability**
   - List top-weighted features for spam vs ham.
6. **Error analysis**
   - Inspect false positives/false negatives and identify patterns.

---

## How to Run
1. Open the notebook in Colab or run locally.
2. Install dependencies (if needed):
   - `kagglehub`
   - `scikit-learn`
3. Run cells top-to-bottom.

Outputs:
- Head of dataset
- Accuracy
- Classification report

---

## Notes to Future Me
- If performance suddenly changes, check:
  - whether the split was stratified
  - whether TF-IDF was fit on train only
  - whether preprocessing changed (URL tokenization, casing)
- Don’t chase accuracy; focus on spam precision/recall depending on the desired behavior.
