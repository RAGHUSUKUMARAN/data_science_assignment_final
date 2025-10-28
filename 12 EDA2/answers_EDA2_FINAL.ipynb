# EDA2 — Data Preprocessing & Feature Engineering

# =======================
# 1. Introduction
# =======================

# This notebook performs exploratory data analysis and preprocessing on the UCI Adult dataset.
# The tasks include handling missing values, scaling, encoding categorical features,
# feature engineering, outlier detection, and feature selection. This work aligns with
# the EDA2 assignment objectives.

# =======================
# 2. Load Dataset
# =======================

DATA_PATH = r"D:\DATA-SCIENCE\ASSIGNMENTS\12 EDA2\adult_with_headers.csv"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    display(df.head())
except FileNotFoundError:
    print(f"File not found at {DATA_PATH}. Please check path.")

# =======================
# 3. Data Exploration and Cleaning
# =======================

# Check for missing values and descriptive stats
display(df.info())
print("\nMissing Values:\n", df.isna().sum())
display(df.describe(include='all').T)

# Fill missing categorical values with mode and numerical with median
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=[np.number]).columns

df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

print("✅ Missing values handled.")

# =======================
# 4. Feature Scaling
# =======================

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_std = StandardScaler()
scaler_mm = MinMaxScaler()

scaled_std = pd.DataFrame(scaler_std.fit_transform(df[num_cols]), columns=num_cols)
scaled_mm = pd.DataFrame(scaler_mm.fit_transform(df[num_cols]), columns=num_cols)

print("Standard Scaler sample:")
display(scaled_std.head())
print("MinMax Scaler sample:")
display(scaled_mm.head())

# =======================
# 5. Encoding Techniques
# =======================

from sklearn.preprocessing import LabelEncoder

# Separate small and large categorical variables
onehot_cols = [c for c in cat_cols if df[c].nunique() <= 5]
label_cols = [c for c in cat_cols if df[c].nunique() > 5]

print(f"One-hot encoding {len(onehot_cols)} columns, Label encoding {len(label_cols)} columns.")

# Apply encodings
df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

for c in label_cols:
    le = LabelEncoder()
    df_encoded[c] = le.fit_transform(df_encoded[c].astype(str))

print("✅ Encoding complete.")

# =======================
# 6. Feature Engineering
# =======================

if set(['capital_gain','capital_loss']).issubset(df_encoded.columns):
    df_encoded['capital_diff'] = df_encoded['capital_gain'] - df_encoded['capital_loss']
if 'hours_per_week' in df_encoded.columns and 'age' in df_encoded.columns:
    df_encoded['hours_per_age'] = df_encoded['hours_per_week'] / df_encoded['age']
if 'capital_gain' in df_encoded.columns:
    df_encoded['log_capital_gain'] = np.log1p(df_encoded['capital_gain'])

print("✅ Feature engineering complete.")
display(df_encoded[['capital_diff','hours_per_age','log_capital_gain']].head())

# =======================
# 7. Outlier Detection
# =======================

from sklearn.ensemble import IsolationForest

num_df = df_encoded.select_dtypes(include=[np.number])
iso = IsolationForest(contamination=0.01, random_state=42)

outlier_labels = iso.fit_predict(num_df)
df_encoded['outlier_iforest'] = (outlier_labels == -1)
print(f"Outliers detected: {df_encoded['outlier_iforest'].sum()}")

# =======================
# 8. Feature Selection via PPS
# =======================

try:
    import ppscore as pps
    pps_matrix = pps.matrix(df_encoded)
    print("Top PPS scores:")
    display(pps_matrix.sort_values('ppscore', ascending=False).head(10))
except Exception as e:
    print("⚠ PPS could not be computed:", e)

# =======================
# 9. Save Processed Dataset
# =======================

OUT = r"D:\DATA-SCIENCE\ASSIGNMENTS\12 EDA2\adult_processed_for_modeling.csv"
df_encoded.to_csv(OUT, index=False)
print(f"✅ Saved processed dataset to: {OUT}")

# =======================
# 10. Conclusion
# =======================

# Summary of key insights
# - The Adult dataset contains 32k rows with mixed numeric and categorical data.
# - Missing values were minimal and successfully imputed.
# - Standard and MinMax scaling were applied for normalization.
# - Encoding handled both small and large categorical variables effectively.
# - Feature engineering introduced three useful derived features.
# - Outlier detection identified rare anomalies for further study.
# - The processed dataset is ready for modeling or ML pipeline integration.
