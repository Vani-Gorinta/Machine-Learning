import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vanig\Downloads\archive (1)\Housing.csv")
print(df)

df.head(10)

df.info()

print(df.columns.tolist())

df.drop(columns=['airconditioning', 'hotwaterheating', 'prefarea', 'basement'], inplace=True, errors='ignore')
df['area'] = df['area'].fillna(df['area'].median())
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
print(df.isnull().sum())

duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")
df.drop_duplicates(inplace=True)
print(f"Data shape after removing duplicates: {df.shape}")

print("Null values before dropping:\n", df.isnull().sum())
df.dropna(inplace=True)
print(f"Shape after dropping nulls: {df.shape}")
print("Null values after dropping:\n", df.isnull().sum())

Q1 = df['area'].quantile(0.25)
Q3 = df['area'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['area'] >= lower_bound) & (df['area'] <= upper_bound)]
print(f"Data shape after removing outliers: {df.shape}")