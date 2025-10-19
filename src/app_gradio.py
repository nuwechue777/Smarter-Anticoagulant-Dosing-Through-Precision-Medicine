# app_gradio.py
# ---------------------------------------------------------
# Genexa Warfarin Dosing — Gradio Prototype
# Loads the MLflow-registered champion model and serves
# guardrailed dose recommendations using the engineered
# feature pipeline from training.
#
# ⚠️ Prototype for internal evaluation only. Not for clinical use.
# ---------------------------------------------------------

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gradio as gr
import mlflow
import mlflow.sklearn
import category_encoders as ce  # mirrors training encoder

# =========================
# Configuration
# =========================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "Smarter_Antocoagulant_Dosing")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")  # or use "Production" stage

# Dose guardrails
DOSE_MIN, DOSE_MAX = 1.0, 10.0
DOSE_STEP = 0.5

# Fallback training columns (schema safety net if we can't read from the model)
# (Will be overridden by trained_columns.json or model.feature_names_in_ if available.)
FALLBACK_TRAINED_COLUMNS: List[str] = [
    "Age", "BMI", "Weight_kg", "Height_cm",
    "Sex_F", "Sex_M",
    "Ethnicity_African American", "Ethnicity_Asian", "Ethnicity_Caucasian", "Ethnicity_Hispanic", "Ethnicity_Other",
    "Alcohol_Intake_Light", "Alcohol_Intake_Moderate", "Alcohol_Intake_Heavy", "Alcohol_Intake_nan",
    "Smoking_Status_Former Smoker", "Smoking_Status_Non-smoker", "Smoking_Status_Smoker",
    "Diet_VitK_Intake_High", "Diet_VitK_Intake_Medium", "Diet_VitK_Intake_Low",
    "CYP2C9_*1/*1", "CYP2C9_*1/*2", "CYP2C9_*1/*3", "CYP2C9_*2/*2", "CYP2C9_*2/*3", "CYP2C9_*3/*3",
    "VKORC1_A/A", "VKORC1_A/G", "VKORC1_G/G",
    "CYP4F2_C/C", "CYP4F2_C/T", "CYP4F2_T/T",
    "On_Amiodarone",
    "Comorbidity_Burden",
    # Some training artifacts showed this exact column name; we include both spellings and rely on reindexing.
    "Polypharamcy_index",
    "Polypharamcy_Index",
    "Polypharmacy_Index",
]

# Canonical category order (stabilizes one-hot column order)
CAT_SPECS = {
    "Sex": ["F", "M"],
    "Ethnicity": ["African American", "Asian", "Caucasian", "Hispanic", "Other"],
    "Alcohol_Intake": ["None", "Light", "Moderate", "Heavy"],
    "Diet_VitK_Intake": ["High", "Medium", "Low"],
    "Smoking_Status": ["Former Smoker", "Non-smoker", "Smoker"],
    "CYP2C9": ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"],
    "VKORC1": ["A/A", "A/G", "G/G"],
    "CYP4F2": ["C/C", "C/T", "T/T"],
}

# Categorical columns encoded in training
CATEGORICAL_COLS = [
    "Sex", "Ethnicity", "Alcohol_Intake", "Diet_VitK_Intake", "Smoking_Status",
    "CYP2C9", "VKORC1", "CYP4F2",
]

GENE_DEFAULTS = {"CYP2C9": "*1/*1", "VKORC1": "G/G", "CYP4F2": "C/C"}
COMORBS = ["Hypertension", "Diabetes", "Chronic_Kidney_Disease", "Heart_Failure"]
MEDS = ["Amiodarone", "Antibiotics", "Aspirin", "Statins"]


# =========================
# Helpers
# =========================
def clip_and_round(dose: float, step: float = DOSE_STEP) -> float:
    dose = float(np.clip(dose, DOSE_MIN, DOSE_MAX))
    return round(step * round(dose / step), 2)


def ensure_genes(payload: Dict) -> Dict:
    fixed = payload.copy()
    for g, dflt in GENE_DEFAULTS.items():
        if fixed.get(g) in [None, "", "NA", "Unknown"]:
            fixed[g] = dflt
    return fixed


def engineer_single(example: Dict) -> pd.DataFrame:
    """
    Build the same engineered features used in training.
    """
    ex = ensure_genes(example)

    row = {
        "Age": ex.get("Age"),
        "Weight_kg": ex.get("Weight_kg"),
        "Height_cm": ex.get("Height_cm"),
        "Sex": ex.get("Sex"),
        "Ethnicity": ex.get("Ethnicity"),
        "Alcohol_Intake": ex.get("Alcohol_Intake"),
        "Diet_VitK_Intake": ex.get("Diet_VitK_Intake"),
        "Smoking_Status": ex.get("Smoking_Status"),
        "CYP2C9": ex.get("CYP2C9"),
        "VKORC1": ex.get("VKORC1"),
        "CYP4F2": ex.get("CYP4F2"),
        "On_Amiodarone": int(ex.get("Amiodarone", 0) or ex.get("On_Amiodarone", 0)),
    }
    for c in COMORBS:
        row[c] = int(ex.get(c, 0))
    for m in MEDS:
        row[m] = int(ex.get(m, 0))

    df = pd.DataFrame([row])

    # Fix categorical dtype with explicit categories
    for col, cats in CAT_SPECS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=cats, ordered=False)

    # Engineered features (mirror training)
    df["BMI"] = df["Weight_kg"] / ((df["Height_cm"] / 100.0) ** 2)
    df["Comorbidity_Burden"] = df[COMORBS].sum(axis=1)
    # Two spellings to ensure alignment with whichever the trained schema contains
    df["Polypharmacy_Index"] = df[MEDS].sum(axis=1)
    df["Polypharamcy_Index"] = df["Polypharmacy_Index"]
    df["Polypharamcy_index"] = df["Polypharmacy_Index"]

    engineered_cols = [
        "Age", "Weight_kg", "Height_cm", "Sex", "Ethnicity",
        "Alcohol_Intake", "Diet_VitK_Intake", "Smoking_Status",
        "CYP2C9", "VKORC1", "CYP4F2", "On_Amiodarone",
        "BMI", "Comorbidity_Burden",
        # include all to be safe; final alignment happens in to_model_matrix()
        "Polypharmacy_Index", "Polypharamcy_Index", "Polypharamcy_index",
    ]
    return df[engineered_cols]


def load_trained_columns(model) -> List[str]:
    """
    Load the exact training schema:
    1) trained_columns.json (if saved during training)
    2) model.feature_names_in_ (sklearn >= 1.0)
    3) fallback hard-coded list
    """
    if os.path.exists("trained_columns.json"):
        try:
            with open("trained_columns.json", "r") as f:
                cols = json.load(f)
            if isinstance(cols, list) and len(cols) > 0:
                return cols
        except Exception:
            pass

    try:
        cols = list(model.feature_names_in_)  # available for most sklearn estimators
        if len(cols) > 0:
            return cols
    except Exception:
        pass

    return FALLBACK_TRAINED_COLUMNS


def to_model_matrix(df_one: pd.DataFrame, trained_columns: List[str]) -> pd.DataFrame:
    """
    Mirror the training encoder (category_encoders.OneHotEncoder with use_cat_names=True).
    We fit on the single row (safe for OHE), then align to the exact training schema.
    """
    enc = ce.OneHotEncoder(
        cols=CATEGORICAL_COLS,
        use_cat_names=True,
        handle_unknown="ignore",   # unseen levels -> ignored (zeroed by reindex)
        handle_missing="value"     # produces *_nan columns if NaN is present
    )
    X = enc.fit_transform(df_one)  # DataFrame with named columns
    # Align to model schema; add any missing columns as zeros, drop extras
    X = X.reindex(columns=trained_columns, fill_value=0)
    return X


# =========================
# Load model & schema
# =========================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
champion_model = mlflow.sklearn.load_model(MODEL_URI)
TRAINED_COLUMNS = load_trained_columns(champion_model)


# =========================
# Gradio predict function
# =========================
def predict_fn(
    Age, Weight_kg, Height_cm, Sex, Ethnicity,
    Alcohol_Intake, Diet_VitK_Intake, Smoking_Status,
    CYP2C9, VKORC1, CYP4F2,
    Hypertension, Diabetes, Chronic_Kidney_Disease, Heart_Failure,
    Amiodarone, Antibiotics, Aspirin, Statins
) -> Tuple[str, str, pd.DataFrame, pd.DataFrame]:
    payload: Dict = {
        "Age": Age, "Weight_kg": Weight_kg, "Height_cm": Height_cm,
        "Sex": Sex, "Ethnicity": Ethnicity,
        "Alcohol_Intake": Alcohol_Intake, "Diet_VitK_Intake": Diet_VitK_Intake, "Smoking_Status": Smoking_Status,
        "CYP2C9": CYP2C9, "VKORC1": VKORC1, "CYP4F2": CYP4F2,
        "Hypertension": int(Hypertension), "Diabetes": int(Diabetes),
        "Chronic_Kidney_Disease": int(Chronic_Kidney_Disease), "Heart_Failure": int(Heart_Failure),
        "Amiodarone": int(Amiodarone), "Antibiotics": int(Antibiotics),
        "Aspirin": int(Aspirin), "Statins": int(Statins),
    }

    df_one = engineer_single(payload)
    X_one = to_model_matrix(df_one, TRAINED_COLUMNS)

    raw_pred = float(champion_model.predict(X_one)[0])
    dose = clip_and_round(raw_pred, step=DOSE_STEP)

    # Return: guardrailed dose, raw prediction, engineered row, model matrix row
    return f"{dose} mg/day", f"{raw_pred:.3f} mg/day", df_one, X_one


# =========================
# Build Gradio UI
# =========================
with gr.Blocks(title="Genexa Warfarin Dosing (Prototype)") as demo:
    gr.Markdown("## Genexa Warfarin Dosing — Prototype")
    gr.Markdown(
        "Interactive prototype using MLflow-registered champion model. "
        "**Not for clinical use.**"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Demographics")
            Age = gr.Slider(18, 100, value=67, step=1, label="Age")
            Weight_kg = gr.Slider(30, 200, value=75, step=0.5, label="Weight (kg)")
            Height_cm = gr.Slider(120, 220, value=170, step=0.5, label="Height (cm)")
            Sex = gr.Radio(["M", "F"], value="M", label="Sex")
            Ethnicity = gr.Dropdown(
                ["African American", "Asian", "Caucasian", "Hispanic", "Other"],
                value="Caucasian", label="Ethnicity"
            )

            gr.Markdown("### Lifestyle")
            Alcohol_Intake = gr.Dropdown(
                ["None", "Light", "Moderate", "Heavy"],
                value="Light", label="Alcohol Intake"
            )
            Diet_VitK_Intake = gr.Dropdown(
                ["High", "Medium", "Low"],
                value="Medium", label="Vitamin K Intake"
            )
            Smoking_Status = gr.Dropdown(
                ["Former Smoker", "Non-smoker", "Smoker"],
                value="Non-smoker", label="Smoking Status"
            )

        with gr.Column():
            gr.Markdown("### Genomics")
            CYP2C9 = gr.Dropdown(
                ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"],
                value="*1/*2", label="CYP2C9"
            )
            VKORC1 = gr.Dropdown(["A/A", "A/G", "G/G"], value="A/G", label="VKORC1")
            CYP4F2 = gr.Dropdown(["C/C", "C/T", "T/T"], value="C/T", label="CYP4F2")

            gr.Markdown("### Comorbidities")
            Hypertension = gr.Checkbox(True, label="Hypertension")
            Diabetes = gr.Checkbox(False, label="Diabetes")
            Chronic_Kidney_Disease = gr.Checkbox(False, label="Chronic Kidney Disease")
            Heart_Failure = gr.Checkbox(False, label="Heart Failure")

            gr.Markdown("### Medications")
            Amiodarone = gr.Checkbox(False, label="Amiodarone")
            Antibiotics = gr.Checkbox(False, label="Antibiotics")
            Aspirin = gr.Checkbox(True, label="Aspirin")
            Statins = gr.Checkbox(True, label="Statins")

    predict_btn = gr.Button("Predict Dose", variant="primary")

    with gr.Row():
        with gr.Column():
            dose_out = gr.Textbox(label="Recommended Dose (guardrailed)", interactive=False)
            raw_out = gr.Textbox(label="Raw Model Prediction (pre-guardrails)", interactive=False)
        with gr.Column():
            with gr.Accordion("Debug: Engineered Feature Row", open=False):
                eng_out = gr.Dataframe(wrap=True)
            with gr.Accordion("Debug: Model Matrix Row (aligned to training columns)", open=False):
                X_out = gr.Dataframe(wrap=True)

    gr.Markdown(
        f"**Model:** `{MODEL_NAME}` — version `{MODEL_VERSION}`  \n"
        f"**Tracking URI:** `{MLFLOW_TRACKING_URI}`  \n"
        f"Guardrails: dose clipped to {DOSE_MIN}–{DOSE_MAX} mg, rounded to {DOSE_STEP} mg steps."
    )

    predict_btn.click(
        fn=predict_fn,
        inputs=[
            Age, Weight_kg, Height_cm, Sex, Ethnicity,
            Alcohol_Intake, Diet_VitK_Intake, Smoking_Status,
            CYP2C9, VKORC1, CYP4F2,
            Hypertension, Diabetes, Chronic_Kidney_Disease, Heart_Failure,
            Amiodarone, Antibiotics, Aspirin, Statins
        ],
        outputs=[dose_out, raw_out, eng_out, X_out]
    )

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)
    # demo.launch()
