#coding: utf-8

# Importing libraries
import requests
import json

# Input record
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
    print("Congratulations!! Your loan has been approved.")
    print("Prediction probability:: ", result['prediction_probability'])

else:
    print("We are Sorry!! Your loan is not approved at this time.")
    print("Prediction probability:: ", result['prediction_probability'])


