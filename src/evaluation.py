# ─────────────────────────────────────────
# evaluation.py
# Handles model evaluation and charts
# ─────────────────────────────────────────

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay


def get_all_metrics(model, X_test, y_test, model_name):
    # Get all evaluation metrics for one model
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Model':     model_name,
        'Accuracy':  round(accuracy_score(y_test, pred),  4),
        'Precision': round(precision_score(y_test, pred), 4),
        'Recall':    round(recall_score(y_test, pred),    4),
        'F1-Score':  round(f1_score(y_test, pred),        4),
        'ROC-AUC':   round(roc_auc_score(y_test, prob),   4),
    }

    return metrics


def print_classification_report(model, X_test, y_test, model_name):
    # Print detailed classification report
    pred = model.predict(X_test)
    print(f"\n─── Classification Report — {model_name} ───")
    print(classification_report(y_test, pred,
          target_names=['No Disease', 'Disease']))


import os

def plot_confusion_matrix(model, X_test, y_test, model_name):
    # Plot confusion matrix for a model
    os.makedirs("../reports/figures", exist_ok=True)
    cm   = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['No Disease', 'Disease'])
    plt.figure(figsize=(6, 5))
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix — {model_name}")
    plt.savefig(f"../reports/figures/cm_{model_name.lower().replace(' ', '_')}.png")
    plt.show()
    print(f"Confusion matrix saved!")


def plot_roc_curve(model, X_test, y_test, model_name):
    # Plot ROC curve for a model
    os.makedirs("../reports/figures", exist_ok=True)
    plt.figure(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=model_name)
    plt.title(f"ROC Curve — {model_name}")
    plt.savefig(f"../reports/figures/roc_{model_name.lower().replace(' ', '_')}.png")
    plt.show()
    print(f"ROC curve saved!")


def plot_model_comparison(results_df):
    # Plot accuracy and ROC-AUC comparison bar charts
    os.makedirs("../reports/figures", exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Accuracy', y='Model', data=results_df, palette='Blues_d')
    plt.title("Model Accuracy Comparison")
    plt.xlim(0.7, 1.0)
    plt.savefig("../reports/figures/07_model_comparison_accuracy.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(x='ROC-AUC', y='Model', data=results_df, palette='Greens_d')
    plt.title("Model ROC-AUC Comparison")
    plt.xlim(0.7, 1.0)
    plt.savefig("../reports/figures/08_model_comparison_rocauc.png")
    plt.show()

    print("Comparison charts saved!")

