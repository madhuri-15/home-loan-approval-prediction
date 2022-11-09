#coding: utf-8

# Import libraries
import pickle

import pandas as pd
import numpy as np

from flask import Flask
from flask import request, jsonify
from flask import render_template, url_for

# To load model and preprocessor.
model_file = "model.bin"
with open(model_file, 'rb') as f_in:
    preprocessor, model = pickle.load(f_in)


# Function to predict single input record
def prediction(customer, preprocessor, model):
    
    X = preprocessor.transform(customer)
    y_preds = model.predict_proba(X)[0, 1]
    
    status = y_preds > 0.5
    return status, round(y_preds*100)


# Create flask app
app = Flask("Loan Approval Status", template_folder='templates')


# Route to home page
@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

# Route to predictions
@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        
        customer = {
                    "loan_id": request.form['loan_id'],
                    "gender": request.form['gender'],
                    "married": request.form['married'],
                    "dependents": request.form['dependents'],
                    "education": request.form['education'],
                    "self_employed": request.form['self_employed'],
                    "applicant_income": np.log1p(float(request.form['applicant_income'])),
                    "coapplicant_income": np.log1p(float(request.form['coapplicant_income'])),
                    "loan_amount": np.log1p(float(request.form['loan_amount'])),
                    "loan_amount_term": int(request.form['loan_amount_term']),
                    "credit_history": int(request.form['credit_history']),
                    "property_area": request.form['property_area']
                    }

        customer = pd.DataFrame.from_records([customer])
        status, y_preds = prediction(customer, preprocessor, model)

        result = {
            "prediction_probability": float(y_preds),
            "loan_status": bool(status)
            }

        if result['loan_status']:
            return render_template('index.html',status="True", probability=result["prediction_probability"])
        else:
            return render_template('index.html', status="False", probability=(100-result["prediction_probability"]))

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
