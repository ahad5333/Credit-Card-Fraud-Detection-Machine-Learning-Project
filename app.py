import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE


# STEP 1 — Load Dataset
df = pd.read_csv("creditcard.csv")

print("Dataset Preview:")
print(df.head())


# STEP 2 — Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]


# STEP 3 — Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# STEP 4 — Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# STEP 5 — Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# STEP 6 — Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=20, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
}


# Store trained models and probabilities
trained_models = {}
probabilities = {}


# STEP 7 — Train Models and Compare AUC
print("\nModel AUC Comparison:\n")

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train_smote, y_train_smote)

    # Save trained model
    trained_models[name] = model

    # Predict probabilities
    probs = model.predict_proba(X_test_scaled)[:, 1]

    probabilities[name] = probs

    # Calculate AUC
    auc = roc_auc_score(y_test, probs)

    print(f"{name} AUC Score: {auc:.4f}")


# STEP 8 — Plot ROC Curves
plt.figure()

for name, probs in probabilities.items():

    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], '--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model Comparison ROC Curve")

plt.legend()

plt.show()


# STEP 9 — Feature Importance (Random Forest)

rf_model = trained_models["Random Forest"]

importances = rf_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:\n")
print(importance_df.head(10))


# Plot Feature Importance
plt.figure()

top_features = importance_df.head(10)

plt.barh(top_features["Feature"], top_features["Importance"])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features (Random Forest)")

plt.gca().invert_yaxis()

plt.show()