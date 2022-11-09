# coding: utf-8

"""Python script to train and save a final model in a pickle file."""

# Importing libraries

import warnings                       # To ignore the warnings
warnings.filterwarnings('ignore')

# Data preprocessing
import pandas as pd
import numpy as np 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Cross validation
from sklearn.model_selection import train_test_split

# Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# To save model
import pickle

# Read data
datafile = "data.csv"
data = pd.read_csv(datafile)


"""Data preprocessing"""
# Renaming columns
column_list = data.columns.tolist() 
new_cols = {
    'ApplicantIncome':'applicant_income',
    'CoapplicantIncome':'coapplicant_income',
    'LoanAmount':'loan_amount'
    }

data.rename(columns=new_cols, inplace=True)
data.rename(columns={col:col.lower() for col in column_list},
            inplace=True
            )

# Log transformation
for col in ['applicant_income', 'coapplicant_income', 'loan_amount']:
    data[col] = np.log1p(data[col])

""" Data pipeline """

# numeric features
numeric_features = ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_amount_term']
numeric_transformer = Pipeline(
    steps = [
        ("knn_imputer", KNNImputer(n_neighbors=2)), 
        ("scaler", StandardScaler())
    ]
)

# categorical features
categorical_features = ['gender', 'married', 'dependents', 'credit_history', 'education', 'property_area']
categorical_transformer = Pipeline(
    steps = [
        ("simple_imputer", SimpleImputer(strategy='most_frequent')),
        ("oh_encoder", OneHotEncoder(handle_unknown='ignore'))
    ]
)

# preprocessors
preprocessor = ColumnTransformer(
    transformers = [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Convert response variable from 'Y', 'N' to 1 and 0 respectively.
data['loan_status'] = (data['loan_status']=='Y').astype('int').reset_index(drop=True)

# Split the data into training and test dataset in ratio 70:20:10 ratio
df_full, df_test = train_test_split(data, test_size=0.1, shuffle=True, stratify=data['loan_status'], random_state=42)
df_train, df_val = train_test_split(df_full, test_size=0.22, shuffle=True, stratify=df_full['loan_status'], random_state=42)

# reset index
df_train.reset_index(drop=True)
df_val.reset_index(drop=True)
df_test.reset_index(drop=True)

# Split the data into X and y
X_train, y_train = df_train[numeric_features + categorical_features], df_train['loan_status']
X_val, y_val = df_val[numeric_features + categorical_features], df_val['loan_status']

x_train = preprocessor.fit_transform(X_train)
x_val = preprocessor.transform(X_val)

# Model development
clf = GradientBoostingClassifier(n_estimators=250,
                                 learning_rate=0.03, 
                                 max_depth=3, 
                                 min_samples_split=200, 
                                 min_samples_leaf=50,
                                 max_features=11,
                                 warm_start=True,
                                 random_state=42
                                 )
# Train the model
clf.fit(x_train, y_train)

# Predictions
y_preds = clf.predict(x_val)

# Model evaluation
score = f1_score(y_val, y_preds, average='macro')
print("Validation F1-score::%f" %score)

# Model evaluation on test data
X_test, y_test = df_test[numeric_features + categorical_features], df_test['loan_status']
x_test = pd.DataFrame(preprocessor.transform(X_test))

# predictions
y_preds = clf.predict(x_test)
score = f1_score(y_test, y_preds, average='macro')
print("Test F1-Score::%f" % score)

  
# Save model in pickle file
output_file = "model.bin"
with open(output_file, 'wb') as f_out:
     pickle.dump((preprocessor, clf), f_out)

print("Model saved as %s file" %output_file)


