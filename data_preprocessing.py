import pandas as pd
df = pd.read_csv("loan_data.csv")

if "LoanID" in df.columns:
    df.drop(columns=["LoanID"], inplace=True)
print("✅ LoanID column dropped (if exists)")

df.fillna(df.median(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("✅ Missing values filled successfully!")


for col in df.select_dtypes(include=["object"]).columns:
    df[col] = pd.factorize(df[col])[0]
print("✅ Categorical columns converted to numerical format!")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "NumCreditLines", "InterestRate", "DTIRatio"]

df[num_cols] = scaler.fit_transform(df[num_cols])
print("✅ Numerical features normalized!")


df.to_csv("cleaned_loan_data.csv", index=False)
print("✅ Preprocessed dataset saved as 'cleaned_loan_data.csv'")
