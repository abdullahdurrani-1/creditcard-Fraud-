# creditcard-Fraud-
Here is a complete **`README.md`** for your **Credit Card Fraud Detection Project**

---

````markdown
# 💳 Credit Card Fraud Detection using Supervised Learning



## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using supervised machine learning algorithms, particularly **Logistic Regression** and **Random Forest**. The dataset is highly imbalanced, which poses a challenge and requires proper preprocessing and evaluation strategies.

---

## 📁 Dataset Description

- **Source**: Kaggle ([Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Size**: 284,807 transactions
- **Target**: `Class` (1 = Fraud, 0 = Non-Fraud)
- **Features**: 30 anonymized numerical features + `Time`, `Amount`

---

## ⚙️ Machine Learning Pipeline

### 🔹 1. **Data Preprocessing**
- Applied **StandardScaler** to `Amount` and `Time`
- Dropped scaled columns after transformation
- Checked for nulls and unique values

### 🔹 2. **Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
````

### 🔹 3. **Modeling**

* **Logistic Regression** as a baseline
* **Random Forest Classifier** (with `class_weight='balanced'`) to handle class imbalance

### 🔹 4. **Evaluation Metrics**

* Confusion Matrix
* Precision, Recall, F1 Score
* **ROC Curve & AUC**
* Feature Importance Plot

### 🔹 5. **Hyperparameter Tuning**

```python
GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=3)
```

---

## 📊 Results Summary

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------- | ------------- |
| Accuracy  | 99.4%               | ✅ Excellent   |
| Precision | 94.5%               | ✅ Excellent   |
| Recall    | 70%                 | ✅ Good        |
| F1 Score  | 80.2%               | ✅ Strong      |
| ROC-AUC   | 97.9%               | ✅ Very High   |

* **Best Parameters (RF)**: `n_estimators=200`, `max_depth=20`, `class_weight='balanced'`
* **Top Features**: V14, V17, V12, V10, V4

---

## 📈 Visualizations

* 🔍 ROC Curve
* 📊 Class Distribution
* 📉 Correlation Heatmap
* 📌 Feature Importance
* 💰 Transaction Distribution

---

## 🤖 Algorithms Used

| Algorithm           | Type                | Why Used?                               |
| ------------------- | ------------------- | --------------------------------------- |
| Logistic Regression | Linear Classifier   | Quick baseline model, interpretable     |
| Random Forest       | Ensemble Classifier | Handles imbalance, non-linear relations |

---

## 💡 Lessons Learned

* Proper preprocessing and scaling improves model reliability.
* Class imbalance requires **class weights** or **resampling**.
* ROC & F1-Score are more meaningful than accuracy for imbalanced datasets.
* Feature importance gives insight into what drives fraud detection.


---

## 📚 Future Improvements

* Use SMOTE or ADASYN for synthetic oversampling
* Try other models: XGBoost, LightGBM
* Deploy with Flask or Streamlit for real-time predictions

---

## 🙌 Acknowledgements

* Scikit-Learn, Matplotlib, Seaborn

---

## 🧠 Author

**Abdullah Durrani**
*Student of BS Statistics | Future AI Expert | Poetic Soul on a Mission to Oxford*
🌍 Pakistan
🚀 Learning ML, DS, Agentic AI — devlepor



> **– Abdullah Durrani**
```
