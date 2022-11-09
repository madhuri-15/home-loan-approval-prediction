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
    print("Congratulations!!")
    print("""You successfully complete the loan process.
          You are eligible for this Home Loan and your loan
          can be approved with chance of %.3f %.""" result['prediction_probability'])
else:
    print("We are Sorry!!")
    print("""Your are not eligible for this Home Loan at this time.
             Since, there are %.3f % of chances that your
             loan will not approved.""" result['prediction_probability'])
