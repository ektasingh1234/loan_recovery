import pandas as pd
import numpy as np
import time
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

start_time = time.time()

print("📥 Loading dataset...")
df = pd.read_parquet("cleaned_loan_data.parquet")  # Load preprocessed dataset

# 🎯 Select features for clustering
selected_features = ["LoanAmount", "CreditScore"]
df_cluster = df[selected_features].copy()

# 🧼 Standardize features
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)

# 🤖 Run KMeans clustering
print("🔍 Running K-Means Clustering...")
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=1000, random_state=42, n_init="auto")
df["Cluster"] = kmeans.fit_predict(df_cluster_scaled)

# 🔍 Analyze cluster centers to map risk categories
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=selected_features)
print("\n📊 Cluster Centers (Original Scale):")
print(centers_df)

# 🧠 Map clusters to risk categories based on logic:
# Higher CreditScore + Lower LoanAmount → High Recovery
# Lower CreditScore + Higher LoanAmount → Low Recovery
# Middle values → Medium Risk

# Sort clusters: High Recovery → Medium → Low Recovery
centers_df["Cluster"] = centers_df.index
sorted_clusters = centers_df.sort_values(by=["CreditScore", "LoanAmount"], ascending=[False, True]).reset_index(drop=True)
risk_labels = ["High Recovery", "Medium Risk", "Low Recovery"]
cluster_mapping = dict(zip(sorted_clusters["Cluster"], risk_labels))

df["RiskCategory"] = df["Cluster"].map(cluster_mapping)

# 📝 Output category distribution
print("\n✅ Updated Risk Categories:", df["RiskCategory"].unique())
print(df["RiskCategory"].value_counts())

# 💾 Save the segmented dataset
df.drop(columns=["Cluster"], inplace=True)
df.to_parquet("segmented_borrowers.parquet", index=False)
print("✅ Borrower Segmentation Completed & Saved!")

# 📊 Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["LoanAmount"], y=df["CreditScore"], hue=df["RiskCategory"], palette="coolwarm", alpha=0.6, edgecolor="black")
plt.xlabel("Loan Amount (₹)", fontsize=12)
plt.ylabel("Credit Score", fontsize=12)
plt.title("Borrower Segmentation (Optimized)", fontsize=14)
plt.legend(title="Risk Category", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# ⏱️ Execution Time
exec_time = round(time.time() - start_time, 2)
print(f"🚀 Borrower Segmentation Completed in {exec_time} seconds! 🔥")
