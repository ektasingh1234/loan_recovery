import pandas as pd
df = pd.read_csv("cleaned_loan_data.csv")
df.to_parquet("cleaned_loan_data.parquet", index=False)
print("✅ CSV converted to Parquet Successfully!")
