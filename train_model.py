# Example: How to train and save your model

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your training data
# Replace 'train.csv' with your actual training data file
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')  # Update with your file name

# Preprocessing
# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()

# Create a copy for encoding
df_encoded = df.copy()

# Encode features
df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])  # Male=1, Female=0
df_encoded['Married'] = le.fit_transform(df_encoded['Married'])  # Yes=1, No=0
df_encoded['Education'] = le.fit_transform(df_encoded['Education'])  # Graduate=1, Not Graduate=0
df_encoded['Self_Employed'] = le.fit_transform(df_encoded['Self_Employed'])  # Yes=1, No=0
df_encoded['Property_Area'] = le.fit_transform(df_encoded['Property_Area'])  # Rural=0, Semiurban=1, Urban=2

# Handle Dependents
df_encoded['Dependents'] = df_encoded['Dependents'].replace('3+', 3).astype(int)

# Encode target variable (if needed)
df_encoded['Loan_Status'] = le.fit_transform(df_encoded['Loan_Status'])  # N=0, Y=1

# Prepare features and target
feature_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                   'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                   'Loan_Amount_Term', 'Credit_History', 'Property_Area']

X = df_encoded[feature_columns]
y = df_encoded['Loan_Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Score: {train_score:.4f}")
print(f"Testing Score: {test_score:.4f}")

# Save the model
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'loan_model.pkl'")
print("\nFeature order for prediction:")
for i, col in enumerate(feature_columns):
    print(f"{i+1}. {col}")
