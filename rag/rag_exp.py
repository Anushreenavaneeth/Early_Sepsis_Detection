import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import joblib
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XGB_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgboost_sepsis_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "model_features.pkl")

xgb_model = joblib.load(XGB_MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)

clinical_docs = [
    "Lactate levels above 2 mmol/L indicate tissue hypoperfusion and increased sepsis risk.",
    "Systolic blood pressure below 100 mmHg reflects circulatory instability.",
    "Heart rate above 100 bpm indicates systemic inflammatory response.",
    "Temperature above 38°C or below 36°C is clinically abnormal.",
    "White blood cell count outside normal range suggests inflammatory stress.",
    "Concurrent abnormalities across multiple physiological systems increase sepsis risk."
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

documents = text_splitter.create_documents(clinical_docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(documents, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

#llm pipeline

llm_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=120,
    temperature=0.2,
    device=0 if torch.cuda.is_available() else -1
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

#rag chain

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
)


def predict_and_explain(patient_input: dict):

    df_input = pd.DataFrame([patient_input])

    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model_features]

    sepsis_prob = xgb_model.predict_proba(df_input)[0][1]
    risk_label = "HIGH RISK" if sepsis_prob >= 0.5 else "LOW RISK"

    prompt = f"""
    The predictive model classified this patient as {risk_label}
    with a probability of {sepsis_prob:.2f}.

    Abnormal clinical findings:
    - Temperature: {patient_input['Temp']} °C
    - Heart Rate: {patient_input['HR']} bpm
    - Blood Pressure: {patient_input['BP_Systolic']}/{patient_input['BP_Diastolic']} mmHg
    - White Blood Cell Count: {patient_input['WBC_Count']}
    - Lactate: {patient_input['Lactate_mmol_L']} mmol/L

    Provide a concise clinical justification explaining why the model reached this decision.
    Focus only on abnormal findings. Do not explain sepsis.
    """

    rag_response = rag_chain.invoke({"query": prompt}) 

    explanation = rag_response["result"].strip()

    if not explanation:
        explanation = "No clinically significant abnormal physiological findings contributing to sepsis risk were identified."

    print("DEBUG RAG OUTPUT:", explanation)

    return {
        "Sepsis_Risk": risk_label,
        "Risk_Probability": round(sepsis_prob, 3),
        "Clinical_Explanation": explanation
    }