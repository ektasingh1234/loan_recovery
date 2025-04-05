import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE

start_time = time.time()

# Load & Preprocess Data
df = pd.read_parquet("cleaned_loan_data.parquet")
drop_cols = ["LoanID", "HasCoSigner", "MaritalStatus", "LoanPurpose"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

X = df.drop(columns=["Default"])
y = df["Default"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for Minority Class Balancing
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=209,
    max_depth=8,
    learning_rate=0.0544,
    scale_pos_weight=2.604,
    subsample=0.819,
    colsample_bytree=0.903,
    tree_method="hist",
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Train LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=250,
    max_depth=9,
    learning_rate=0.05,
    boosting_type="gbdt",
    class_weight="balanced",
    random_state=42
)
lgb_model.fit(X_train, y_train)

# Train RandomForest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate Each Model
models = {"XGBoost": xgb_model, "LightGBM": lgb_model, "RandomForest": rf_model}
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n📊 {name} Model Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

# Stacking Classifier
meta_nn = MLPClassifier(hidden_layer_sizes=(50, 25), activation="relu", solver="adam", random_state=42)
stacked_model = StackingClassifier(
    estimators=[("xgb", xgb_model), ("lgb", lgb_model), ("rf", rf_model)],
    final_estimator=meta_nn,
    n_jobs=-1
)
stacked_model.fit(X_train, y_train)

# Save the Stacked Model
joblib.dump(stacked_model, "final_optimized_model.pkl")
print("✅ Final Model saved as 'final_optimized_model.pkl' 🎯")

print(f"🚀 Total Execution Time: {round(time.time() - start_time, 2)} seconds")
