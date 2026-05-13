# Capstone-Project--MLP

A high-performance NLP classification pipeline developed for a Kaggle-style machine learning competition focused on automated comment moderation and category prediction.

The project combines advanced text preprocessing, feature engineering, imbalance handling, and ensemble-based machine learning techniques to classify user comments into multiple moderation categories using both textual and metadata features.

---

# Problem Statement

The objective of the competition was to predict the final moderation category assigned to online comments using:

- Raw text comments
- Interaction metadata
- Emoticon indicators
- Topic reference signals
- Numerical engagement features

The dataset contained approximately **198,000+ records** and exhibited significant class imbalance, making Macro F1-score optimization a critical challenge.

---

# Competition Performance

- Achieved a **Macro F1-score of 0.827**
- Ranked in the **Top 400 among 2500 participants**

---

# Key Features

## NLP Preprocessing
- Text normalization
- Lowercasing
- Noise reduction
- Feature transformations
- Word-level TF-IDF vectorization
- Character-level TF-IDF vectorization

## Feature Engineering
- Log transformations for skewed numerical features
- Combined structured + unstructured feature pipelines
- Sparse matrix optimization
- Dimensionality reduction using Truncated SVD

## Machine Learning Pipeline
Tested and evaluated multiple models including:

- Logistic Regression
- Support Vector Machines (SVM)
- LightGBM

## Model Optimization
- Hyperparameter tuning using:
  - Grid Search
  - Random Search
- Stratified K-Fold Cross Validation
- Threshold optimization for minority class recall improvement

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- NLP
- TF-IDF
- Truncated SVD
- Matplotlib

---

# Dataset Features

The dataset included:

| Feature Type | Examples |
|---|---|
| Text Features | Comment content |
| Numerical Features | Upvotes, Downvotes |
| Categorical Features | Emoticon groups |
| Sensitive Topic Indicators | Race, Religion, Gender, Disability |
| Metadata | Post IDs, timestamps |

---

# Pipeline Overview

```text
Raw Data
   ↓
EDA & Cleaning
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Cross Validation
   ↓
Threshold Optimization
   ↓
Final Submission
