# ─────────────────────────────────────────
# model_training.py
# Handles training all 6 ML models
# ─────────────────────────────────────────

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def train_logistic_regression(X_train, y_train):
    # Train Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    print("Logistic Regression trained!")
    return model


def train_knn(X_train, y_train):
    # Train KNN model
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    print("KNN trained!")
    return model


def train_decision_tree(X_train, y_train):
    # Train Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Decision Tree trained!")
    return model


def train_random_forest(X_train, y_train):
    # Train Random Forest model
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    print("Random Forest trained!")
    return model


def train_naive_bayes(X_train, y_train):
    # Train Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Naive Bayes trained!")
    return model


def train_svm(X_train, y_train):
    # Train SVM model
    model = SVC(random_state=42, probability=True)
    model.fit(X_train, y_train)
    print("SVM trained!")
    return model


def get_accuracy(model, X_test, y_test):
    # Get accuracy score for a model
    pred = model.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    return round(acc, 4)


def get_roc_auc(model, X_test, y_test):
    # Get ROC-AUC score for a model
    prob    = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, prob)
    return round(roc_auc, 4)


def save_model(model, filepath):
    # Save the trained model to a file
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    # Load a saved model from file
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def save_scaler(scaler, filepath):
    # Save the fitted scaler to a file
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")


def load_scaler(filepath):
    # Load a saved scaler from file
    scaler = joblib.load(filepath)
    print(f"Scaler loaded from {filepath}")
    return scaler

