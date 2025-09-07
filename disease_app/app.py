from flask import Flask, render_template, request, send_file, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from werkzeug.utils import secure_filename
import os

# ---------------- Helpers ----------------
def safe_float(val):
    """Convert form input to float safely, else 0.0"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

# ---------------- App Config ----------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- Load Model ----------------
model_path = "diabetes_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if os.path.getsize(model_path) == 0:
    raise ValueError(f"Model file is empty: {model_path}")

with open(model_path, "rb") as f:
    try:
        model = pickle.load(f)
    except EOFError:
        raise EOFError(f"Error loading model: {model_path} is corrupted or incomplete")

# Feature order same as training
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ---------------- Routes ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [safe_float(request.form.get(k)) for k in FEATURE_NAMES]
        data = np.array(values).reshape(1, -1)

        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0][prediction]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template(
            "report.html",
            result=result,
            probability=round(proba * 100, 2),
            pairs=list(zip(FEATURE_NAMES, values)),
            features=FEATURE_NAMES,
            values=values,
            bmi=values[5],
            glucose=values[1]
        )
    except Exception as e:
        return render_template("index.html", error=f"Error: {e}", request_form=request.form)

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        result_from_form = request.form.get("result")
        probability_from_form = request.form.get("probability")

        values = [safe_float(request.form.get(k)) for k in FEATURE_NAMES]

        if result_from_form is None or probability_from_form is None:
            data = np.array(values).reshape(1, -1)
            prediction = model.predict(data)[0]
            proba = model.predict_proba(data)[0][prediction]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"
            probability = round(proba * 100, 2)
        else:
            result = result_from_form
            probability = safe_float(probability_from_form)

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        elems = []

        elems.append(Paragraph("Diabetes Detection Report", styles["Heading1"]))
        elems.append(Spacer(1, 8))
        elems.append(Paragraph(f"<b>Prediction Result:</b> {result}", styles["Normal"]))
        elems.append(Paragraph(f"<b>Probability:</b> {probability}%", styles["Normal"]))
        elems.append(Spacer(1, 12))

        table_data = [["Feature", "Value"]]
        for f, v in zip(FEATURE_NAMES, values):
            table_data.append([f, str(v)])

        table = Table(table_data, colWidths=[220, 220])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ]))
        elems.append(table)

        doc.build(elems)
        buf.seek(0)

        return send_file(buf,
                         as_attachment=True,
                         download_name="diabetes_report.pdf",
                         mimetype="application/pdf")
    except Exception as e:
        return f"Error generating PDF: {e}"

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
