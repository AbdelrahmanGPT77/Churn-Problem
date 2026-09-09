# Churn-Problem
Telecom Customer Churn Prediction &amp; Analysis System using Machine Learning (Python, Scikit-Learn, EDA, &amp; Feature Engineering).

# 📞 Telecom Customer Churn Prediction & EDA

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) project designed to predict customer churn for a telecommunications company (**Wasla**). 

This project cleans messy real-world data, uncovers hidden patterns driving customer attrition, engineers domain-specific features, and trains machine learning pipelines to accurately identify high-risk churn customers.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Highlights](#-key-features--highlights)
- [Dataset Overview](#-dataset-overview)
- [Exploratory Data Analysis (EDA) & Insights](#-exploratory-data-analysis-eda--insights)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture & Training](#-model-architecture--training)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)

---

## 🎯 Project Overview

Customer retention is critical for telecom businesses. Retaining an existing customer is significantly cheaper than acquiring a new one. This project focuses on:
1. Detecting missing/hidden values in billing records (e.g., empty string representations in numeric features).
2. Handling class imbalance between churned and non-churned customers.
3. Feature engineering based on domain knowledge (e.g., customer tenure buckets, total active services, family status).
4. Building ML pipelines using **Scikit-Learn** (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`) to train baseline classification models (Logistic Regression, Random Forest).

---

## ✨ Key Features & Highlights

- **Data Cleaning & Sanitization**: Handled hidden missing values in `TotalCharges` (converting blank spaces to numerical zero where `tenure == 0`).
- **Domain-Specific Feature Engineering**:
  - `tenure_bucket`: Categorized customer loyalty into standard ranges (`0-12 Months`, `13-24 Months`, `25-48 Months`, `48+ Months`).
  - `num_Services`: Counted active add-on services (`OnlineSecurity`, `TechSupport`, `StreamingTV`, etc.) to measure customer engagement.
  - `has_family`: Combined `Partner` and `Dependents` to determine household ties.
- **Robust Machine Learning Pipeline**:
  - Implemented automated preprocessors using Scikit-Learn `ColumnTransformer`.
  - Applied `OneHotEncoder` (with `drop='first'` & `handle_unknown='ignore'`) for categorical variables and `StandardScaler` for continuous features.
  - Stratified Train-Test Split (80/20) to maintain class balance across splits.

---

## 📊 Dataset Overview

- **Source**: `wasla_customer_records.csv`
- **Total Records**: 7,043 rows × 21 columns
- **Target Variable**: `Churn` (`Yes` / `No`)
- **Class Distribution**:
  - **No Churn**: ~73.46% (5,174 customers)
  - **Churn**: ~26.54% (1,869 customers)

---

## 🛠️ Feature Engineering Summary

| New Feature | Type | Logic / Description |
| :--- | :--- | :--- |
| `tenure_bucket` | Categorical | Groups tenure into 4 buckets (`0-12`, `13-24`, `25-48`, `48+` months) |
| `num_Services` | Numerical | Sum of subscribed add-on services (Security, Backup, Tech Support, etc.) |
| `has_family` | Categorical | Binary flag (`Yes`/`No`) if customer has a Partner or Dependents |

---

## 🚀 Model Architecture & Pipeline

The project uses modular `Pipeline` objects to prevent data leakage and streamline preprocessing:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# Baseline Models
- Logistic Regression (max_iter=1000)
- Random Forest Classifier
