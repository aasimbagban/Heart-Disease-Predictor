# 🫀 Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask)
![Pandas](https://img.shields.io/badge/Pandas-2.1-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

> A complete end-to-end Machine Learning project that predicts whether a patient
> has heart disease using 13 clinical parameters. Built with Python, Scikit-learn,
> and deployed as a web app using Flask.

---

## 📌 Problem Statement

Cardiovascular disease is the **number one cause of death globally**.
Early prediction of heart disease risk using routine clinical data
can enable timely medical intervention and save lives.

This project trains and compares **6 classification algorithms** on the
UCI Cleveland Heart Disease dataset to find the best model for prediction.

---

## 🎯 Project Highlights

- ✅ Trained and compared **6 ML models** side by side
- ✅ Best model: **Tuned Random Forest** with **90.16% test accuracy** and **0.9481 ROC-AUC**
- ✅ Full **EDA** with saved figures and correlation heatmaps
- ✅ **Train-only Feature Scaling** using `StandardScaler` to eliminate Data Leakage
- ✅ **Hyperparameter tuning** via `GridSearchCV`
- ✅ **Flask Clinical Web App** (`CardioScan AI Pro v2.5`) with real-time risk probability calculation & factor breakdown

---

## 📊 Model Results

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| **Random Forest (Tuned)** | **90.16%**| **0.9481** |
| Logistic Regression | 86.89%   | 0.9513  |
| SVM                 | 85.25%   | 0.9437  |
| KNN                 | 83.61%   | 0.9453  |
| Naive Bayes         | 83.61%   | 0.9170  |
| Decision Tree       | 75.41%   | 0.7543  |

> 🏆 **Best Model: Tuned Random Forest Classifier** — 90.16% Accuracy & 0.9481 ROC-AUC

---

## 📁 Project Structure

```
heart-disease-predictor/
│
├── data/
│   ├── raw/
│   │   └── heart.csv                  ← Original UCI dataset
│   └── processed/
│       └── heart_cleaned.csv          ← Preprocessed dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← EDA and charts
│   ├── 02_feature_engineering.ipynb   ← Scaling and feature prep
│   └── 03_model_building.ipynb        ← All 6 models + tuning + saving
│
├── src/
│   ├── __init__.py                    ← Makes src a Python package
│   ├── data_preprocessing.py          ← Data cleaning functions
│   ├── feature_engineering.py         ← Train-test scaling functions
│   ├── model_training.py              ← Model training & saving functions
│   └── evaluation.py                  ← Metrics and chart functions
│
├── models/
│   ├── best_model.pkl                 ← Saved Random Forest model
│   └── scaler.pkl                     ← Saved StandardScaler
│
├── app/
│   ├── app.py                         ← Flask backend with validation & probability scoring
│   └── templates/
│       └── index.html                 ← Clinical CardioScan AI Pro UI
│
├── reports/
│   └── figures/                       ← Generated visualization charts
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🗂️ Dataset

- **Source:** UCI Machine Learning Repository — Cleveland Heart Disease Dataset
- **Rows:** 303 patients
- **Columns:** 13 features + 1 target
- **Target:** 0 = No Heart Disease, 1 = Heart Disease

| Feature   | Description                          |
|-----------|--------------------------------------|
| age       | Age of patient                       |
| sex       | Sex (1 = Male, 0 = Female)           |
| cp        | Chest pain type (0–3)                |
| trestbps  | Resting blood pressure               |
| chol      | Serum cholesterol                    |
| fbs       | Fasting blood sugar > 120mg (0/1)    |
| restecg   | Resting ECG results (0–2)            |
| thalach   | Maximum heart rate achieved          |
| exang     | Exercise induced angina (0/1)        |
| oldpeak   | ST depression induced by exercise    |
| slope     | Slope of peak exercise ST segment    |
| ca        | Number of major vessels (0–3)        |
| thal      | Thalassemia type (1/2/3)             |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/aasimbagban/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Install all libraries
```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order
```
01_data_exploration.ipynb      ← EDA and charts
02_feature_engineering.ipynb   ← Scaling
03_model_building.ipynb        ← Train models and save best_model.pkl
```

### 4. Start the Flask app
```bash
cd app
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:3000
```

---

## 🧪 Test the App

Use these sample values to test a **high risk** prediction:

| Field    | Value |
|----------|-------|
| Age      | 52    |
| Sex      | 1     |
| CP       | 0     |
| Trestbps | 125   |
| Chol     | 212   |
| FBS      | 0     |
| Restecg  | 1     |
| Thalach  | 168   |
| Exang    | 0     |
| Oldpeak  | 1.0   |
| Slope    | 2     |
| CA       | 2     |
| Thal     | 3     |

---

## 🛠️ Libraries Used

```
flask        → web application framework
joblib       → saving and loading the trained model
numpy        → numerical calculations
pandas       → data loading and manipulation
scikit-learn → all 6 ML models and evaluation metrics
matplotlib   → charts and plots
seaborn      → statistical visualizations
jupyter      → running notebooks
```

---

## 📈 Key Learnings

- **KNN outperformed all models** including Random Forest on this dataset
- **Recall matters more than accuracy** in medical diagnosis — missing a disease is dangerous
- **ROC-AUC is a better metric** than accuracy for binary medical classification
- **StandardScaler is critical for KNN** — KNN uses distance so unscaled features give wrong results
- **Cross-validation gives honest results** — single train/test split can be misleading

---

## 🚀 Future Improvements

- [ ] Test XGBoost and LightGBM models
- [ ] Add SHAP values for model explainability
- [ ] Show prediction probability percentage in the web app

- [ ] Add patient history tracking

---

## 👨‍💻 Author

**Aasim bagban**
- GitHub:https://github.com/aasimbagban
- LinkedIn: https://www.linkedin.com/in/aasimbagban/

---

## 📄 License

This project is licensed under the MIT License.

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
