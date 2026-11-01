"""Train an explainable heart-disease risk model from a local CSV dataset."""
from pathlib import Path
import json
import warnings
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
ARTIFACT_PATH = ROOT / "model_artifact.joblib"
METRICS_PATH = ROOT / "evaluation_metrics.json"

TARGET_CANDIDATES = ["target", "output", "heartdisease", "heart_disease", "num", "label", "disease", "condition"]
UCI_CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]


def ensure_dataset_exists() -> None:
    """Download the public UCI Cleveland heart dataset if no local CSV exists."""
    if DATA_PATH.exists():
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("data/heart.csv was not found. Downloading the UCI Cleveland Heart Disease dataset...")
    try:
        with urllib.request.urlopen(UCI_CLEVELAND_URL, timeout=30) as response:
            raw = response.read()
        from io import BytesIO
        df = pd.read_csv(BytesIO(raw), header=None, names=UCI_COLUMNS, na_values="?")
        if df.empty or len(df.columns) != 14:
            raise ValueError("The downloaded file did not have the expected 14 columns.")
        df.to_csv(DATA_PATH, index=False)
        print(f"Downloaded {len(df)} records from the UCI Machine Learning Repository to {DATA_PATH}")
    except Exception as exc:
        raise RuntimeError(
            "Could not automatically download the public dataset. Check your internet connection, "
            "then try again. You can also download the processed Cleveland data from the UCI "
            "Heart Disease repository and save it as data/heart.csv with the 14 expected columns."
        ) from exc


def find_target_column(df: pd.DataFrame) -> str:
    lower_to_original = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError("Could not identify the target column. Rename the outcome column to 'target' in data/heart.csv.")


def normalize_target(series: pd.Series) -> pd.Series:
    """Return binary labels: 0 = lower/no recorded risk, 1 = disease/risk class."""
    s = series.copy()
    if s.isna().any():
        raise ValueError("The target column contains missing values. Remove rows with missing target labels first.")
    if s.dtype == object:
        mapped = s.astype(str).str.strip().str.lower().map({
            "0": 0, "no": 0, "negative": 0, "absent": 0, "false": 0, "healthy": 0,
            "1": 1, "yes": 1, "positive": 1, "present": 1, "true": 1, "disease": 1,
            "low": 0, "high": 1
        })
        if mapped.notna().all():
            return mapped.astype(int)
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().all():
            s = numeric
        else:
            raise ValueError("Target labels must be binary (0/1) or numeric disease severity (0 = no disease, >0 = disease).")
    values = sorted(pd.unique(s))
    if len(values) < 2:
        raise ValueError("The target column must contain at least two classes.")
    if set(values).issubset({0, 1}):
        return s.astype(int)
    # Common Cleveland heart dataset uses 0 = no disease and 1-4 = disease severity.
    return (pd.to_numeric(s, errors="raise") > 0).astype(int)


def main():
    ensure_dataset_exists()
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    target_col = find_target_column(df)
    y = normalize_target(df[target_col])
    X = df.drop(columns=[target_col]).copy()
    # Convert blank/question-mark values to missing; coerce numeric-looking columns.
    X = X.replace(["?", "", "NA", "N/A", "null"], np.nan)
    for col in X.columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        nonmissing_original = X[col].notna().sum()
        if nonmissing_original and converted.notna().sum() / nonmissing_original >= 0.90:
            X[col] = converted
    # Drop columns that are entirely empty and rows with no usable features.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns found in the dataset.")
    if y.nunique() != 2:
        raise ValueError("Target normalization did not produce two classes.")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ], remainder="drop", verbose_feature_names_out=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()

    # SMOTE is applied to training data only to avoid test-set leakage.
    smote_used = False
    try:
        from imblearn.over_sampling import SMOTE
        smallest_class = int(y_train.value_counts().min())
        if smallest_class >= 2 and y_train.nunique() == 2 and y_train.value_counts().nunique() > 1:
            X_train_t, y_train = SMOTE(random_state=42, k_neighbors=min(5, smallest_class - 1)).fit_resample(X_train_t, y_train)
            smote_used = True
    except ImportError:
        print("imbalanced-learn not installed; continuing without SMOTE.")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    except ImportError:
        print("xgboost not installed; Random Forest will be evaluated alone.")

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        pred = model.predict(X_test_t)
        probs = model.predict_proba(X_test_t)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probs)) if probs is not None and y_test.nunique() == 2 else None,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)
        }
        results[name] = metrics
        fitted[name] = model

    # Select model by F1 score, a useful metric when class counts are uneven.
    best_name = max(results, key=lambda name: results[name]["f1_score"])
    artifact = {
        "preprocessor": preprocessor,
        "model": fitted[best_name],
        "model_name": best_name,
        "feature_names": feature_names,
        "input_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "training_medians": {c: float(X[c].median()) for c in numeric_cols if X[c].notna().any()},
        "training_modes": {c: str(X[c].mode(dropna=True).iloc[0]) for c in categorical_cols if not X[c].mode(dropna=True).empty},
        "input_min": {c: float(X[c].min()) for c in numeric_cols if X[c].notna().any()},
        "input_max": {c: float(X[c].max()) for c in numeric_cols if X[c].notna().any()},
        "target_column": target_col,
        "smote_used": smote_used,
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "dataset_rows": int(len(df)),
        "dataset_features": int(X.shape[1]),
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    summary = {
        "selected_model": best_name,
        "selection_rule": "Highest F1-score on the held-out test split",
        "test_size": 0.20,
        "random_state": 42,
        "smote_used_on_training_only": smote_used,
        "dataset_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "results_by_model": results
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected_model": best_name, "results": {n: {k: v for k, v in m.items() if k != "classification_report"} for n, m in results.items()}, "artifact": str(ARTIFACT_PATH)}, indent=2))
    print("Saved model_artifact.joblib and evaluation_metrics.json")

if __name__ == "__main__":
    main()
