import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os
import time


start_time = time.time()


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("📥 Loading Model & Data...")


model = joblib.load("loan_default_fast_model.pkl")


df = pd.read_parquet("cleaned_loan_data.parquet")

X = df.drop(columns=["Default"])


if hasattr(model, "feature_names_in_"):
    X = X[model.feature_names_in_]


X_sample = X.sample(500, random_state=42)

print(f"✅ Sampled {X_sample.shape[0]} rows for SHAP analysis.")


print("🔍 Running SHAP Analysis...")
explainer = shap.Explainer(model.predict, X_sample)
shap_values = explainer(X_sample)


plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=True)
plt.title("SHAP Feature Importance")
plt.show()


exec_time = round(time.time() - start_time, 2)
print(f"🚀 SHAP Analysis Completed in {exec_time} seconds! 🔥")
