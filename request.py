# coding: "utf-8"
"""Python script to test the local flask application"""

# Importing libraries
import requests
import json


customer = {
    "loan_id": "LP001032",
    "gender": "male",
    "married": "no",
    "dependents": "0",
    "education": "graduate",
    "self_employed": "no",
    "applicant_income": 4950,
    "coapplicant_income": 0.0,
    "loan_amount": 125.0,
    "loan_amount_term": 360.0,
    "credit_history": 1.0,
    "property_area": "urban"
    }

url = "http://localhost:9696//predict"
response = requests.post(url, json=customer)
result = response.json()

if result['loan_status']:
    print("Prediction Probabililty:: ", result['prediction_probability'])
    print("Loan Status:: Approved")
else:
    print("Prediction Probabililty:: ", result['prediction_probability'])
    print("Loan Status:: Not Approved")
