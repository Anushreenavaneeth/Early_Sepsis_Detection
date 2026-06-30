from flask import Flask, render_template, request
from datetime import datetime

from rag.rag_exp import predict_and_explain

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    #read input
    pid = request.form["pid"]

    patient_input = {
        "Temp": float(request.form["Temp"]),
        "HR": float(request.form["HR"]),
        "BP_Systolic": float(request.form["BP_Systolic"]),
        "BP_Diastolic": float(request.form["BP_Diastolic"]),
        "WBC_Count": float(request.form["WBC_Count"]),
        "Lactate_mmol_L": float(request.form["Lactate_mmol_L"])
    }

    
    result = predict_and_explain(patient_input)
    print("DEBUG RAG OUTPUT:", result["Clinical_Explanation"])
    
    frontend_result = {
        "pid": pid,
        "risk": int(result["Risk_Probability"] * 100),
        "label": result["Sepsis_Risk"],
        "explanation": result["Clinical_Explanation"],
        "time": datetime.now().strftime("%d %b %Y %H:%M"),
        "vitals": patient_input
    }

    return render_template("index.html", result=frontend_result)

if __name__ == "__main__":
    app.run(debug=True)
