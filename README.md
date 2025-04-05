# 🏦 Loan Recovery ML Project

This project predicts the likelihood of loan recovery using machine learning models. It helps financial institutions identify high-risk customers and improve recovery strategies.

## 📌 Objective
To classify loans as:
- Fully Recovered
- Partially Recovered
- Defaulted

## 📁 Dataset
The dataset includes:
- Loan amount  
- Repayment behavior  
- Credit history  
- Customer demographics  

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Random Forest
- Matplotlib, Seaborn

## 🧠 Model Training Results

| Model         | Accuracy | Recall  | Precision | F1 Score |
|---------------|----------|---------|-----------|----------|
| XGBoost       | 76.99%   | 50.85%  | 25.45%    | 33.92%   |
| LightGBM      | **83.87%** | 35.32%  | **32.24%** | 33.71%   |
| Random Forest | 74.89%   | **53.13%** | 23.88%    | 32.95%   |

📌 **Final model saved as:** `final_optimized_model.pkl`

## 📊 LightGBM Classification Report

