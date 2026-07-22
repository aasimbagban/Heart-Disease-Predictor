# ─────────────────────────────────────────
# data_preprocessing.py
# Handles loading and cleaning the dataset
# ─────────────────────────────────────────

import pandas as pd
import numpy as np


def load_data(filepath):
    # Load the CSV file into a dataframe
    df = pd.read_csv(filepath)
    print("Data loaded successfully!")
    print("Shape:", df.shape)
    return df


def inspect_data(df):
    # Print basic info about the dataset
    print("─── Shape ───")
    print(df.shape)

    print("\n─── First 5 Rows ───")
    print(df.head())

    print("\n─── Data Types ───")
    print(df.dtypes)

    print("\n─── Missing Values ───")
    print(df.isnull().sum())

    print("\n─── Basic Statistics ───")
    print(df.describe())

    print("\n─── Target Distribution ───")
    print(df['target'].value_counts())


def remove_duplicates(df):
    # Remove any duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows")
    return df


def handle_missing_values(df):
    # Fill missing values with median of each column
    df = df.fillna(df.median())
    print("Missing values handled!")
    print("Missing values remaining:", df.isnull().sum().sum())
    return df


def remove_outliers(df):
    # Clip extreme statistical anomalies using 3.0 * IQR to preserve true clinical risk metrics
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    for col in num_cols:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR

        # Clip extreme anomalies
        df[col] = df[col].clip(lower, upper)

    print("Outliers handled!")
    return df


def clean_data(df):
    # Run all cleaning steps in order
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_outliers(df)
    print("\nData cleaning complete!")
    return df
