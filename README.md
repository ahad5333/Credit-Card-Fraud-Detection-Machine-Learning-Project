# Credit Card Fraud Detection – Machine Learning Project

## Project Overview

This project builds a **machine learning system to detect fraudulent credit card transactions**. Fraud detection is a classic **imbalanced classification problem** because fraudulent transactions are extremely rare compared to normal transactions.

The goal of this project is to:

* preprocess transaction data
* handle severe class imbalance
* train multiple machine learning models
* compare model performance using ROC-AUC
* visualize model performance
* understand which features are most important for detecting fraud

This project demonstrates a **typical end-to-end machine learning workflow** used in real data science and ML projects.

---

# Dataset

Dataset used: **creditcard.csv**

The dataset contains **real anonymized credit card transactions**.

### Dataset Characteristics

* Total transactions: ~284,807
* Fraudulent transactions: 492
* Normal transactions: ~284,315

This means fraud cases represent **only about 0.17% of the data**, which makes this a **highly imbalanced dataset**.

### Features

The dataset contains:

| Feature  | Description                                                        |
| -------- | ------------------------------------------------------------------ |
| Time     | Seconds elapsed between this transaction and the first transaction |
| V1 – V28 | Anonymized features obtained using PCA                             |
| Amount   | Transaction amount                                                 |
| Class    | Target variable (0 = normal, 1 = fraud)                            |

Most features are transformed using **Principal Component Analysis (PCA)** to protect user privacy.

---

# Project Workflow

The project follows these major steps:

1. Load dataset
2. Split features and target
3. Train-test split
4. Feature scaling
5. Handle class imbalance with SMOTE
6. Train multiple ML models
7. Compare models using ROC-AUC
8. Plot ROC curves
9. Analyze feature importance

Each step is explained in detail below.

---

# Step 1 – Load Dataset

```python
df = pd.read_csv("creditcard.csv")
```

Purpose:

* Load the dataset into a pandas DataFrame
* Allows data manipulation and analysis

Why pandas is used:

* Efficient data manipulation
* Easy column operations
* Integration with machine learning libraries

---

# Step 2 – Separate Features and Target

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

Explanation:

* `X` contains **input features**
* `y` contains the **target variable**

Target variable:

```
0 → Normal transaction
1 → Fraud transaction
```

Separating features and labels is necessary before training machine learning models.

---

# Step 3 – Train Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Purpose:

Split the dataset into:

| Dataset      | Purpose                    |
| ------------ | -------------------------- |
| Training set | Used to train the model    |
| Testing set  | Used to evaluate the model |

80% → Training
20% → Testing

Why this is important:

Without a test set, we cannot measure **how well the model generalizes to new data**.

`random_state=42` ensures the split is **reproducible**.

---

# Step 4 – Feature Scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Why scaling is needed:

Some machine learning algorithms perform better when features have similar ranges.

Example:

| Feature  | Value Range |
| -------- | ----------- |
| Amount   | 0 – 25,000  |
| V1 – V28 | -5 – 5      |

Without scaling, large-range features dominate the model.

StandardScaler transforms features to:

```
mean = 0
standard deviation = 1
```

Important rule:

```
fit_transform → training data
transform → test data
```

This avoids **data leakage**.

---

# Step 5 – Handle Class Imbalance (SMOTE)

Problem:

```
Normal transactions: 227,451
Fraud transactions: 394
```

This imbalance can cause models to **always predict normal transactions**.

Solution: **SMOTE (Synthetic Minority Oversampling Technique)**

```
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
```

What SMOTE does:

* Generates **synthetic fraud samples**
* Balances the dataset

After SMOTE:

```
227,451 normal
227,451 fraud
```

Now the model can learn fraud patterns properly.

---

# Step 6 – Define Machine Learning Models

Three models are used:

### Logistic Regression

Simple linear classification model.

Advantages:

* Fast
* Interpretable
* Good baseline model

---

### Random Forest

Ensemble model that builds **many decision trees**.

Advantages:

* Handles nonlinear patterns
* Robust to noise
* Good performance on tabular data

---

### Gradient Boosting

Boosting model that builds trees sequentially.

Advantages:

* Very powerful
* Often performs well on structured datasets

---

# Step 7 – Model Training and Evaluation

Each model is trained using:

```
model.fit(X_train_smote, y_train_smote)
```

Then predictions are made:

```
model.predict_proba(X_test_scaled)
```

We evaluate models using **ROC-AUC**.

---

# Why ROC-AUC is Used

Accuracy is misleading for imbalanced datasets.

Example:

```
If model predicts all transactions as normal:

Accuracy = 99.8%
But model detects zero frauds.
```

ROC-AUC measures the model's ability to **separate fraud from normal transactions**.

Higher AUC means better performance.

---

# Step 8 – ROC Curve Visualization

The ROC curve plots:

```
False Positive Rate (X-axis)
True Positive Rate (Y-axis)
```

This helps visualize how well each model distinguishes fraud transactions.

The closer the curve is to the **top-left corner**, the better the model.

---

# Step 9 – Feature Importance

Random Forest provides **feature importance scores**.

```
rf_model.feature_importances_
```

Purpose:

Identify which features contribute most to fraud detection.

Top features are visualized using a bar chart.

This step helps understand **which transaction characteristics influence predictions**.

---

# Expected Output

Example results:

```
Logistic Regression AUC Score: 0.9785
Random Forest AUC Score: 0.9991
Gradient Boosting AUC Score: 0.9862
```

Typical interpretation:

* Random Forest performs best
* Logistic Regression provides a strong baseline
* Gradient Boosting performs competitively

---

# Project Learnings

This project demonstrates important machine learning concepts:

* handling imbalanced datasets
* feature scaling
* train/test splitting
* model comparison
* ROC-AUC evaluation
* model interpretability

These concepts are widely used in **real-world machine learning systems**.

---

# Future Improvements

Possible improvements:

* hyperparameter tuning (GridSearchCV)
* cross-validation
* anomaly detection models
* deep learning approaches
* real-time fraud detection pipeline

---

# How to Run the Project

Install dependencies:

```
pip install pandas matplotlib scikit-learn imbalanced-learn
```

Run the script:

```
python app.py
```

---

# Conclusion

This project builds a complete **fraud detection machine learning pipeline**, covering:

* data preprocessing
* imbalance handling
* model training
* evaluation
* visualization
* model interpretation

It provides a strong foundation for understanding **practical machine learning workflows** used in financial fraud detection systems.
