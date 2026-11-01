# Explainable Ensemble Learning-Based Early Disease Risk Prediction System

Academic prototype based on the attached abstract. It trains Random Forest and (when installed) XGBoost models, optionally applies SMOTE to the training split only, compares models using held-out metrics, and provides a Streamlit interface with SHAP explanations.

## Project files

- `train_model.py` — loads a dataset, preprocesses it, trains/evaluates models, and saves the best model.
- `app.py` — Streamlit dashboard for entering feature values, predicting the class, and viewing explanations.
- `requirements.txt` — Python dependencies.
- `data/heart.csv` — downloaded automatically from the UCI Machine Learning Repository the first time training runs, if the file is not already present.

## 1. Environment setup (Windows)

Install Python 3.10 or 3.11, then open Command Prompt in this folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation of XGBoost is not supported on the computer, remove `xgboost` from requirements or install the other packages first. The training script can run with Random Forest alone.

## 2. Dataset

The project is configured to download the public UCI Cleveland Heart Disease dataset automatically the first time you run `python train_model.py`, if `data/heart.csv` is not already present. An internet connection is required for this first run. The dataset has 303 records and 13 clinical attributes; the target has values 0–4, with 0 representing no recorded disease and values 1–4 representing disease. The script converts this to a binary target. The CSV contains one row per patient and columns for clinical features. The outcome column should ideally be named `target` and coded `0` for no recorded disease and `1` for disease. Common names such as `output`, `num`, `label`, `disease`, or `condition` are also detected. For a Cleveland-style outcome with `0` meaning no disease and `1–4` meaning disease severity, the script converts values greater than zero to the positive class.

Typical feature names in a common heart-disease CSV include `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`. Your actual dataset may differ. The UCI dataset is provided under CC BY 4.0. Cite the dataset in the report: Janosi, A., Steinbrunn, W., Pfisterer, H., & Detrano, R. (1989). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X. Review feature definitions and missing values before use. Do not use personally identifiable patient data.

## 3. Train and evaluate

```bash
python train_model.py
```

This creates `model_artifact.joblib` and `evaluation_metrics.json`. The script uses a stratified 80/20 train/test split. Preprocessing is fitted on training data only; when available, SMOTE is applied only to the training data. Model selection uses F1-score. Review accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix rather than reporting accuracy alone.

## 4. Run the web app

```bash
streamlit run app.py
```

Streamlit opens the dashboard in a browser. Enter feature values and click **Predict risk and explain**.

## 5. Important research notes

- This is a starter implementation aligned to the abstract, not a reproduction of the full IEEE base paper; the attached file is a two-page project abstract and does not provide the paper's complete methods, dataset, or code.
- The abstract's “95%+ accuracy” is not established until the team runs a valid experiment. Never invent or hard-code results.
- This tool is an educational prototype, not a medical diagnostic device. Predictions are not medical advice and should not guide treatment.
- For a stronger M.E. dissertation, document the dataset source, inclusion/exclusion criteria, feature definitions, preprocessing, train/test protocol, hyperparameter tuning, baseline comparison, confidence intervals or cross-validation, class imbalance, limitations, and reproducibility.
- The current target conversion assumes values greater than zero represent disease for numeric multi-class labels. Verify that this matches the chosen dataset before training.
