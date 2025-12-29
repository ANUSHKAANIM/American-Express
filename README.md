# 📊 American Express – Customer Default Prediction

## 📌 Project Overview
Credit default prediction is a critical task in financial risk management.  
This project uses the **American Express Customer Default Prediction dataset** to build machine learning models that predict whether a customer will default on their credit card payments based on anonymized behavioral and transactional data.

The objective is to enhance **credit risk assessment** and support better lending decisions.

---

## 🧠 Problem Statement
To predict whether a customer will default (`1`) or not (`0`) using historical customer profile features and time-series behavioral data.

---

## 📂 Dataset Information
- **Dataset:** American Express Default Prediction  
- **Format:** Parquet  
- **Size:** ~5.5 million rows × 191 columns  
- **Target Variable:** `target`  
  - `1` → Default  
  - `0` → No Default  
- **Class Distribution:**  
  - 75% Non-default  
  - 25% Default (Imbalanced)

---

## 🔧 Data Preprocessing
- Converted object columns to **categorical**
- Downcasted numerical columns to reduce memory usage
- **Missing Values Handling**
  - Dropped columns with >80% missing values
  - Mean imputation for numerical features
  - Mode imputation for categorical features
- **Feature Scaling**
  - Applied **RobustScaler** to handle outliers
- Final processed dataset: **100,000 × 191**

---

## 🔍 Exploratory Data Analysis (EDA)
- Target distribution analysis
- Feature correlation analysis
- KDE plots for class separation
- Heatmaps for highly correlated features
- Feature categorization using prefixes:
  - **D:** Delinquency  
  - **S:** Spend  
  - **P:** Payment  
  - **B:** Balance  
  - **R:** Risk  

---

## 🚨 Outlier Detection & Handling
- **IQR Method**
- **Modified Z-score**
- Outliers handled using:
  - Clipping (5th–95th percentile)
  - Log transformation for skewed features

---

## 🧪 Feature Engineering
- Created behavioral and aggregated features:
  - `payment_to_spend_ratio`
  - `delinquency_frequency`
  - `spend_cv`
- Aggregated statistics: mean, std, max, last
- Trend-based features using linear regression
- Chi-square test for categorical feature significance

---

## 🤖 Models Implemented

| Model | Test Accuracy | Test ROC-AUC |
|------|--------------|-------------|
| Decision Tree | 81.46% | 0.9107 |
| Logistic Regression | 85.30% | 0.9349 |
| Random Forest | 84.17% | 0.9272 |
| LightGBM | 85.02% | 0.9372 |
| Neural Network | 87.06% | 0.9360 |
| **XGBoost** | **87.70%** | **0.9383** |

---

## 🏆 Final Model
**XGBoost** was selected as the final model due to its superior performance.

### 🔧 Hyperparameter Tuning
Performed using **GridSearchCV** with cross-validation.

Best parameters:
```python
{
  'subsample': 0.6,
  'reg_lambda': 2.0,
  'reg_alpha': 1.0,
  'n_estimators': 300,
  'max_depth': 6,
  'learning_rate': 0.05,
  'colsample_bytree': 0.6
}
