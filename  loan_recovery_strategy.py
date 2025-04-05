import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt


start_time = time.time()

print("📥 Loading Borrower Segments...")

try:
    df = pd.read_parquet("segmented_borrowers.parquet")
except FileNotFoundError:
    print("❌ ERROR: File not found! Check if 'segmented_borrowers.parquet' exists.")
    exit()


if "RiskCategory" not in df.columns:
    print("❌ ERROR: 'RiskCategory' column missing! Check segmentation step.")
    exit()


def suggest_recovery_strategy(risk_category, loan_amount):
    if risk_category == "High Recovery":
        return f"Offer EMI restructuring: ₹{round(loan_amount / 12, 2)} per month"
    elif risk_category == "Medium Risk":
        discount = loan_amount * 0.3
        return f"Offer one-time settlement at ₹{round(loan_amount - discount, 2)}"
    else:
        return "Initiate legal action notice"


df["RecoveryStrategy"] = df.apply(lambda row: suggest_recovery_strategy(row["RiskCategory"], row["LoanAmount"]), axis=1)


if "Medium Risk" not in df["RiskCategory"].unique():
    print("❌ No Medium-Risk Borrowers Found! Check Data Processing.")
else:

    plt.figure(figsize=(8, 5))
    sns.histplot(df[df["RiskCategory"] == "Medium Risk"]["LoanAmount"] * 0.3, bins=20, color="blue", kde=True)
    plt.xlabel("Discount Amount (₹)", fontsize=12)
    plt.ylabel("Number of Borrowers", fontsize=12)
    plt.title("Distribution of Discounts for Medium-Risk Borrowers", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


df.to_parquet("loan_recovery_plan.parquet", index=False)
print("✅ Discount Optimization Completed & Saved as 'loan_recovery_plan.parquet'!")


exec_time = round(time.time() - start_time, 2)
print(f"🚀 AI-Powered Discount Optimization Completed in {exec_time} seconds! 🔥")

