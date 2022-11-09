#coding: utf-8

# Importing libraries
import pickle
import pandas as pd

from flask import Flask
from flask import request, jsonify


# To load model and preprocessor.
model_file = "model.bin"
with open(model_file, 'rb') as f_in:
    preprocessor, model = pickle.load(f_in)


# Function to predict single input record
def prediction(customer, preprocessor, model):
    
    X = preprocessor.transform(customer)
    y_preds = model.predict_proba(X)[0, 1]
    
    status = y_preds > 0.5
    return status, y_preds

# Create flask app
app = Flask("Loan Approval Status")

@app.route("/predict", methods=['POST'])
def predict():

    customer = request.get_json()
    customer = pd.DataFrame.from_records([customer])

    status, y_preds = prediction(customer, preprocessor, model)

    result = {
        "prediction_probability": float(y_preds),
        "loan_status": bool(status)
        }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="9696")
