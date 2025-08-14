import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vanig\Downloads\archive (1)\Housing.csv")
print(df)

df.head(10)
df.info()

from sklearn.preprocessing import MinMaxScaler

# Select numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Apply MinMaxScaler
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df_normalized[numeric_cols].head())

from sklearn.preprocessing import RobustScaler
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = RobustScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(df_scaled[numeric_cols].head())

from sklearn.preprocessing import StandardScaler
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df_standardized = df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(df_standardized[numeric_cols].head())