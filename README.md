# SMS Spam Classifier

## Overview

This project implements an SMS Spam Classifier to distinguish between 'ham' (non-spam) and 'spam' messages using machine learning techniques. It includes preprocessing, feature engineering, model training and evaluation, hyperparameter tuning, and a Streamlit-based user interface for easy interaction.

## Table of Contents

* [Overview](#overview)
* [Dependencies](#dependencies)
* [Dataset](#dataset)
* [Steps](#steps)

  * [1. Setup and Data Loading](#1-setup-and-data-loading)
  * [2. Data Preprocessing](#2-data-preprocessing)
  * [3. Exploratory Data Analysis (EDA) and Feature Engineering](#3-exploratory-data-analysis-eda-and-feature-engineering)
  * [4. Text Preprocessing](#4-text-preprocessing)
  * [5. Feature Extraction](#5-feature-extraction)
  * [6. Data Splitting](#6-data-splitting)
  * [7. Model Training and Evaluation](#7-model-training-and-evaluation)
  * [8. Hyperparameter Tuning](#8-hyperparameter-tuning)
  * [9. Model Comparison and ROC Curve](#9-model-comparison-and-roc-curve)
  * [10. Model Saving](#10-model-saving)
* [Streamlit Interface](#streamlit-interface)

## Dependencies

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* nltk
* string
* xgboost
* joblib

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk xgboost joblib
```

Download NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Dataset

The dataset (`spam.csv`) contains SMS messages labeled as 'ham' or 'spam'. It is loaded using `pandas` with ISO-8859-1 encoding.

## Steps

### 1. Setup and Data Loading

* Import libraries and ignore warnings.
* Load dataset using `pd.read_csv()`.
* Inspect data using `df.head()`, `df.shape`, and `df.info()`.

### 2. Data Preprocessing

* Rename columns: 'v1' to 'Category', 'v2' to 'Message'.
* Label encode 'Category' using `LabelEncoder`.
* Check and drop duplicate rows.

### 3. Exploratory Data Analysis (EDA) and Feature Engineering

* Visualize class distribution with pie chart.
* Create numerical features:

  * `num_characters`
  * `num_words`
  * `num_sentences`
* View descriptive statistics grouped by 'Category'.
* Plot histograms and pair plots for EDA.

### 4. Text Preprocessing

* Import stopwords and punctuation.
* Define `transform_text()` to:

  * Lowercase
  * Tokenize
  * Remove stopwords and punctuation
  * Apply stemming
  * Return cleaned string
* Apply `transform_text()` to 'Message' column to create 'transformed\_text'.

### 5. Feature Extraction

* Use `CountVectorizer` with `max_features=3000` on 'transformed\_text'.
* Generate sparse matrix `X` and define target `y` from 'Category'.

### 6. Data Splitting

* Split data using `train_test_split` (70% train, 30% test).

### 7. Model Training and Evaluation

* Train multiple classifiers:

  * Logistic Regression
  * Linear SVC
  * Decision Tree
  * Random Forest
  * Multinomial NB
  * Gaussian NB
  * Bernoulli NB
  * XGBoost
* Evaluate using accuracy, precision, recall, F1 score.

### 8. Hyperparameter Tuning

* Define hyperparameter grids.
* Use `RandomizedSearchCV` for:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * Multinomial NB
  * Bernoulli NB
* Print best parameters and scores.

### 9. Model Comparison and ROC Curve

* Select top models (e.g., Logistic Regression, Random Forest, Multinomial NB, AdaBoost).
* Plot ROC curves and calculate AUC for comparison.

### 10. Model Saving

* Save trained models and `CountVectorizer` using `joblib.dump()`.

## Streamlit Interface

A simple and interactive user interface has been built using **Streamlit** to allow users to input SMS messages and receive classification results in real time.

---

For any enhancements or issues, feel free to fork or raise a pull request.
