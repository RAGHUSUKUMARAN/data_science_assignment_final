# EDA1: Cardiotocographic Dataset
# Author: Maddy
# Directory: D:\DATA SCIENCE\ASSIGNMENTS\5 EDA1\EDA1\files

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- 1) Path setup ----------
base_path = r"D:\DATA SCIENCE\ASSIGNMENTS\5 EDA1\EDA1\files"

# Try to find your data file in that folder
candidates = [
    "Cardiotocographic.csv",
    "Cardiotocographic_cleaned.csv",
    "cardiotocographic.csv"
]

for filename in candidates:
    full_path = os.path.join(base_path, filename)
    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        print(f"Loaded dataset: {filename}")
        break
else:
    raise FileNotFoundError("No CSV file found in the folder. Please check the filename.")

# ---------- 2) Basic overview ----------
print("\n--- Dataset Overview ---")
print(f"Shape: {df.shape}")
print("\nColumns:", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())

# ---------- 3) Cleaning ----------
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing numeric columns with median
for col in numeric_cols:
    if df[col].isna().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"Filled missing values in {col} with median.")

# ---------- 4) Summary statistics ----------
print("\n--- Descriptive Statistics ---")
print(df.describe().T)

# ---------- 5) Distribution plots ----------
sns.set(style="whitegrid")
plt.figure(figsize=(15, 10))
df[numeric_cols].hist(bins=25, figsize=(15, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# ---------- 6) Correlation analysis ----------
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Heatmap", fontsize=14)
plt.show()

# ---------- 7) Outlier detection using IQR ----------
def iqr_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return ((data[column] < lower) | (data[column] > upper)).sum()

print("\n--- Outlier Counts (per numeric column) ---")
for col in numeric_cols:
    outliers = iqr_outliers(df, col)
    print(f"{col}: {outliers} outliers")

# ---------- 8) PCA for dimensionality inspection ----------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

pca = PCA(n_components=2)
pcs = pca.fit_transform(scaled_data)
print("\nExplained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))
plt.scatter(pcs[:,0], pcs[:,1], alpha=0.7, c='blue')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Data Distribution")
plt.show()

# ---------- 9) Save cleaned dataset ----------
cleaned_path = os.path.join(base_path, "Cardiotocographic_cleaned_final.csv")
df.to_csv(cleaned_path, index=False)
print(f"\nCleaned dataset saved to: {cleaned_path}")

print("\n--- EDA1 COMPLETED SUCCESSFULLY ---")
