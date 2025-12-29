
# 📊 American Express – Customer Default Prediction

## 📌 Project Overview

Credit default prediction is a critical task in financial risk management. This project uses the **American Express Customer Default Prediction dataset** to build machine learning models that predict whether a customer will default on their credit card payments based on anonymized behavioral and transactional data. The objective is to enhance **credit risk assessment** and support better lending decisions.

---

## 🧠 Problem Statement

To predict whether a customer will default (`1`) or not (`0`) using historical customer profile features and time-series behavioral data.

---

## 📂 Dataset Information

* **Dataset:** American Express Default Prediction
* **Format:** Parquet
* **Size:** ~5.5 million rows × 191 columns
* **Target Variable:** `target`

  * `1` → Default
  * `0` → No Default
* **Class Distribution:**

  * 75% Non-default
  * 25% Default (Imbalanced)

---

## 🔧 Data Preprocessing

* Converted object columns to categorical
* Downcasted numerical columns to reduce memory usage
* Missing value handling:

  * Dropped columns with more than 80% missing values
  * Mean imputation for numerical features
  * Mode imputation for categorical features
* Feature scaling using **RobustScaler** to reduce outlier impact
* Final processed dataset shape: **100,000 × 191**

---

## 🔍 Exploratory Data Analysis (EDA)

* Target distribution analysis showing strong class imbalance
* Feature correlation analysis
* KDE plots for default vs non-default separation
* Heatmaps for highly correlated feature pairs
* Feature categorization using prefixes:

  * **D:** Delinquency
  * **S:** Spend
  * **P:** Payment
  * **B:** Balance
  * **R:** Risk

---

## 🚨 Outlier Detection & Handling

* Outlier detection using:

  * Interquartile Range (IQR)
  * Modified Z-score
* Outliers handled by:

  * Clipping values between 5th and 95th percentiles
  * Log transformation for skewed features

---

## 🧪 Feature Engineering

* Created new behavioral and financial features:

  * `payment_to_spend_ratio`
  * `delinquency_frequency`
  * `spend_cv`
* Aggregated statistics such as mean, standard deviation, maximum, and last observed value
* Trend-based features computed using linear regression
* Chi-square test applied to identify significant categorical features

---

## 🤖 Models Implemented

The following machine learning models were trained and evaluated:

| Model               | Test Accuracy | Test ROC-AUC |
| ------------------- | ------------- | ------------ |
| Decision Tree       | 81.46%        | 0.9107       |
| Logistic Regression | 85.30%        | 0.9349       |
| Random Forest       | 84.17%        | 0.9272       |
| LightGBM            | 85.02%        | 0.9372       |
| Neural Network      | 87.06%        | 0.9360       |
| **XGBoost**         | **87.70%**    | **0.9383**   |

---

## 🏆 Final Model Selection

**XGBoost** was selected as the final model due to its highest accuracy and ROC-AUC score among all models.

### Hyperparameter Tuning

Hyperparameter optimization was performed using **GridSearchCV** with cross-validation.
Best parameters obtained:

* subsample: 0.6
* reg_lambda: 2.0
* reg_alpha: 1.0
* n_estimators: 300
* max_depth: 6
* learning_rate: 0.05
* colsample_bytree: 0.6

Final tuned accuracy: **87.88%**

---

## 📈 Model Evaluation

* Accuracy
* ROC-AUC score
* Classification report
* ROC curve comparison between baseline and optimized models

Both baseline and optimized XGBoost models achieved an **AUC of 0.938**, indicating excellent classification performance.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost, LightGBM
* Neural Networks

---

## 👥 Team Contributions (Group 7 – DMS672)

* **Anushka Nim:** EDA, Neural Network implementation, AUC plotting, model evaluation, report writing
* **Megha Kumari:** Outlier detection and handling, feature engineering, Random Forest model
* **Rahul Ahirwar:** Missing value analysis, Logistic Regression, LightGBM, hyperparameter tuning
* **Abhishek Kumar:** XGBoost model, Chi-square test
* **Lokesh Mahawar:** Decision Tree model, feature importance visualization

---

## 📌 Key Takeaways

* Delinquency-related features are the strongest predictors of default
* Handling class imbalance and outliers is crucial
* Feature engineering significantly improves model performance
* XGBoost performs best for large-scale credit risk prediction

---

## 🙏 Acknowledgement

We sincerely thank **Dr. Faiz Hamid** for his valuable guidance and support throughout this project.


