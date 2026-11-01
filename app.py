"""Streamlit interface for the explainable disease-risk prediction project."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "model_artifact.joblib"
METRICS_PATH = ROOT / "evaluation_metrics.json"

st.set_page_config(page_title="CardioLens", page_icon="❤️", layout="wide")

logo_path = ROOT / "cardiolens_logo.png"
if logo_path.exists():
    st.image(str(logo_path), use_container_width=True)


st.caption("Machine-learning research prototype using ensemble learning and SHAP explanations")
st.warning("For academic demonstration only. This tool is not a medical diagnosis and must not be used to make treatment decisions.")

if not ARTIFACT_PATH.exists():
    st.error("Trained model not found. Follow the README: add data/heart.csv, run train_model.py, then restart the app.")
    st.stop()

artifact = joblib.load(ARTIFACT_PATH)
preprocessor = artifact["preprocessor"]
model = artifact["model"]
feature_names = artifact["feature_names"]

with st.sidebar:
    st.header("Project information")
    st.write(f"**Selected model:** {artifact['model_name']}")
    st.write(f"**Dataset rows:** {artifact['dataset_rows']}")
    st.write(f"**Input features:** {artifact['dataset_features']}")
    st.write(f"**SMOTE used:** {'Yes' if artifact['smote_used'] else 'No'}")
    st.caption("Predictions are only as reliable as the dataset and evaluation process.")

st.subheader("Enter clinical parameters")
st.write("Enter values from a suitable dataset or a hypothetical example. Input ranges are based on the training dataset.")
inputs = {}
numeric_cols = artifact["numeric_columns"]
categorical_cols = artifact["categorical_columns"]
cols = st.columns(3)
for i, col in enumerate(artifact["input_columns"]):
    with cols[i % 3]:
        if col in numeric_cols:
            default = float(artifact["training_medians"].get(col, 0.0))
            min_value = float(artifact["input_min"].get(col, default - 100))
            max_value = float(artifact["input_max"].get(col, default + 100))
            if min_value == max_value:
                max_value = min_value + 1.0
            inputs[col] = st.number_input(
                col.replace("_", " ").title(), min_value=min_value, max_value=max_value,
                value=min(max(default, min_value), max_value), help="Training-dataset range; values outside this range are not accepted."
            )
        else:
            default = artifact["training_modes"].get(col, "")
            inputs[col] = st.text_input(col.replace("_", " ").title(), value=default,
                                        help="Enter a category exactly as represented in the dataset.")

if st.button("Predict risk and explain", type="primary", use_container_width=True):
    row = pd.DataFrame([inputs], columns=artifact["input_columns"])
    for col in numeric_cols:
        row[col] = pd.to_numeric(row[col], errors="coerce")
    try:
        transformed = preprocessor.transform(row)
        prediction = int(model.predict(transformed)[0])
        probability = float(model.predict_proba(transformed)[0, 1]) if hasattr(model, "predict_proba") else None
        left, right = st.columns(2)
        with left:
            st.subheader("Prediction")
            if prediction == 1:
                st.error("Higher predicted disease-risk class")
            else:
                st.success("Lower predicted disease-risk class")
            if probability is not None:
                st.metric("Model-estimated positive-class probability", f"{probability * 100:.1f}%")
                st.caption("This is a model score, not a calibrated personal medical probability.")
        with right:
            st.subheader("Input summary")
            st.dataframe(row.T.rename(columns={0: "Entered value"}), use_container_width=True)

        st.subheader("Why did the model predict this?")
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed)
            # SHAP output differs by library/model version; normalize to one row of feature contributions.
            if isinstance(shap_values, list):
                values = np.asarray(shap_values[-1])[0]
            else:
                values = np.asarray(shap_values)
                if values.ndim == 3:
                    values = values[0, :, -1]
                elif values.ndim == 2:
                    values = values[0]
            values = np.ravel(values)
            if len(values) != len(feature_names):
                raise ValueError("SHAP feature dimensions did not match the model feature names.")
            explanation = pd.DataFrame({"Feature": feature_names, "SHAP contribution": values})
            explanation["Absolute contribution"] = explanation["SHAP contribution"].abs()
            explanation = explanation.sort_values("Absolute contribution", ascending=False).head(12)
            fig, ax = plt.subplots(figsize=(9, max(4, len(explanation) * 0.38)))
            ordered = explanation.sort_values("SHAP contribution")
            ax.barh(ordered["Feature"], ordered["SHAP contribution"])
            ax.axvline(0, linewidth=0.8)
            ax.set_xlabel("SHAP contribution (direction depends on model output)")
            ax.set_title("Features influencing this prediction")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(explanation[["Feature", "SHAP contribution"]], hide_index=True, use_container_width=True)
            st.caption("Positive/negative contributions indicate movement in the model output; they do not prove medical causation.")
        except Exception as e:
            st.info("SHAP explanation could not be displayed for this model/version. Feature importance is shown instead.")
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(12)
                st.bar_chart(imp.set_index("Feature"))
            else:
                st.caption(f"Explanation detail: {e}")
    except Exception as e:
        st.error(f"Could not create a prediction from these inputs: {e}")

st.divider()
st.subheader("Model evaluation")
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    st.write(f"Evaluation used a stratified 80/20 train/test split. Selected model: **{metrics['selected_model']}**.")
    rows = []
    for name, vals in metrics["results_by_model"].items():
        rows.append({"Model": name, "Accuracy": vals["accuracy"], "Precision": vals["precision"], "Recall": vals["recall"], "F1-score": vals["f1_score"], "ROC-AUC": vals["roc_auc"]})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"), use_container_width=True)
    st.caption("These metrics come from the selected dataset and held-out split; they must be reported honestly in the project.")
else:
    st.info("Run train_model.py to create evaluation metrics.")
