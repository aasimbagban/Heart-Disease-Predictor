# ─────────────────────────────────────────
# feature_engineering.py
# Handles feature scaling and preparation
# ─────────────────────────────────────────

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def separate_features_target(df, target_col='target'):
    # Separate input features and output target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    return X, y


def scale_features(X):
    # Scale all features to same range using StandardScaler
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print("Features scaled successfully!")
    return X_scaled, scaler


def scale_train_test(X_train, X_test):
    # Fit scaler ONLY on training data to prevent Data Leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    print("Train/Test features scaled without data leakage!")
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y        # keeps same class ratio in both splits
    )

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
    return X_train, X_test, y_train, y_test

