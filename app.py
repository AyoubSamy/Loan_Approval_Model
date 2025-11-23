from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model (you'll need to provide the model file)
MODEL_PATH = 'loan_model.pkl'

def load_model():
    """Load the trained linear regression model"""
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            print(f"Warning: Model file '{MODEL_PATH}' not found. Using placeholder.")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/')
def index():
    """Render the loan application form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle loan prediction request"""
    try:
        # Get form data
        gender = request.form.get('Gender')
        married = request.form.get('Married')
        dependents = request.form.get('Dependents')
        education = request.form.get('Education')
        self_employed = request.form.get('Self_Employed')
        applicant_income = float(request.form.get('ApplicantIncome'))
        coapplicant_income = float(request.form.get('CoapplicantIncome'))
        loan_amount = float(request.form.get('LoanAmount'))
        loan_amount_term = float(request.form.get('Loan_Amount_Term'))
        credit_history = float(request.form.get('Credit_History'))
        property_area = request.form.get('Property_Area')
        
        # Prepare features for model (you'll need to adjust this based on your model's preprocessing)
        features = prepare_features(
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        )
        
        # Make prediction
        if model is not None:
            prediction = model.predict([features])[0]
            # Assuming binary classification: 1 = Approved, 0 = Rejected
            result = "Approved" if prediction >= 0.5 else "Rejected"
            confidence = prediction if prediction >= 0.5 else 1 - prediction
        else:
            # Placeholder prediction when model is not available
            result = "Approved" if credit_history == 1 and applicant_income > 3000 else "Rejected"
            confidence = 0.75
        
        return render_template('result.html', 
                             result=result, 
                             confidence=f"{confidence*100:.2f}%",
                             applicant_data={
                                 'Gender': gender,
                                 'Married': married,
                                 'Dependents': dependents,
                                 'Education': education,
                                 'Self_Employed': self_employed,
                                 'Applicant_Income': applicant_income,
                                 'Coapplicant_Income': coapplicant_income,
                                 'Loan_Amount': loan_amount,
                                 'Loan_Amount_Term': loan_amount_term,
                                 'Credit_History': credit_history,
                                 'Property_Area': property_area
                             })
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)"""
    try:
        data = request.get_json()
        
        # Prepare features
        features = prepare_features(
            data['Gender'], data['Married'], data['Dependents'],
            data['Education'], data['Self_Employed'],
            float(data['ApplicantIncome']), float(data['CoapplicantIncome']),
            float(data['LoanAmount']), float(data['Loan_Amount_Term']),
            float(data['Credit_History']), data['Property_Area']
        )
        
        # Make prediction
        if model is not None:
            prediction = model.predict([features])[0]
            result = "Approved" if prediction >= 0.5 else "Rejected"
            confidence = prediction if prediction >= 0.5 else 1 - prediction
        else:
            result = "Approved" if data['Credit_History'] == 1 and data['ApplicantIncome'] > 3000 else "Rejected"
            confidence = 0.75
        
        return jsonify({
            'result': result,
            'confidence': float(confidence),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

def prepare_features(gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount,
                    loan_amount_term, credit_history, property_area):
    """
    Prepare features for model prediction
    Adjust this function based on your model's expected input format
    """
    # Encode categorical variables (adjust based on your model's encoding)
    gender_encoded = 1 if gender == 'Male' else 0
    married_encoded = 1 if married == 'Yes' else 0
    
    # Dependents encoding
    if dependents == '0':
        dependents_encoded = 0
    elif dependents == '1':
        dependents_encoded = 1
    elif dependents == '2':
        dependents_encoded = 2
    else:  # '3+'
        dependents_encoded = 3
    
    education_encoded = 1 if education == 'Graduate' else 0
    self_employed_encoded = 1 if self_employed == 'Yes' else 0
    
    # Property area encoding
    if property_area == 'Urban':
        property_area_encoded = 2
    elif property_area == 'Semiurban':
        property_area_encoded = 1
    else:  # Rural
        property_area_encoded = 0
    
    # Combine all features
    features = [
        gender_encoded,
        married_encoded,
        dependents_encoded,
        education_encoded,
        self_employed_encoded,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area_encoded
    ]
    
    return features

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
