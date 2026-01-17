import os
import joblib
import numpy as np

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XGB_MODEL_PATH = os.path.join(BASE_DIR, "../models/xgboost_sepsis_model.pkl")
LLM_MODEL_PATH = os.path.join(BASE_DIR, "../models/llama-2-7b-chat.gguf")

xgb_model = joblib.load(XGB_MODEL_PATH)

llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=0.2,
    max_tokens=256,
    n_ctx=2048,
    verbose=False
)

PROMPT_TEMPLATE = """
You are a clinical decision explanation system.

The predictive model has already classified this patient as:
{risk_label} (probability {risk_prob:.2f}).

Your task is ONLY to explain why the model arrived at this decision.

Rules:
- Do NOT reassess or contradict the model.
- Do NOT introduce new diagnoses.
- If values are within acceptable clinical limits, state they do not significantly elevate sepsis risk.
- Mention only relevant features.

Patient data:
Temperature: {temp} °C
Heart Rate: {hr} bpm
Blood Pressure: {sbp}/{dbp} mmHg
White Blood Cell Count: {wbc}
Lactate: {lactate} mmol/L

Provide a concise, clinician-facing justification.
"""

prompt = PromptTemplate(
    input_variables=[
        "risk_label", "risk_prob",
        "temp", "hr", "sbp", "dbp", "wbc", "lactate"
    ],
    template=PROMPT_TEMPLATE
)

chain = LLMChain(llm=llm, prompt=prompt)

def predict_and_explain(patient_input: dict):

    features = np.array([[
        patient_input["Temp"],
        patient_input["HR"],
        patient_input["BP_Systolic"],
        patient_input["BP_Diastolic"],
        patient_input["WBC_Count"],
        patient_input["Lactate_mmol_L"]
    ]])

    prob = xgb_model.predict_proba(features)[0][1]
    risk_label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"

    explanation = chain.invoke({
        "risk_label": risk_label,
        "risk_prob": prob,
        "temp": patient_input["Temp"],
        "hr": patient_input["HR"],
        "sbp": patient_input["BP_Systolic"],
        "dbp": patient_input["BP_Diastolic"],
        "wbc": patient_input["WBC_Count"],
        "lactate": patient_input["Lactate_mmol_L"]
    })["text"]

    return {
        "risk_probability": float(prob),
        "risk_label": risk_label,
        "clinical_justification": explanation.strip()
    }
