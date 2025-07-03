from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        Age = float(request.form['Age'])
        Gender = request.form['Gender']
        Hypertension = int(request.form['Hypertension'])
        Heart_Disease = int(request.form['Heart_Disease'])
        Residence_Type = request.form['Residence_Type']
        Avg_Glucose_Level = float(request.form['Avg_Glucose_Level'])
        BMI = float(request.form['BMI'])
        Smoking_Status = request.form['Smoking_Status']
        Physical_Activity = request.form['Physical_Activity']
        Alcohol_Consumption = request.form['Alcohol_Consumption']
        Chronic_Stress = int(request.form['Chronic_Stress'])
        Sleep_Hours = float(request.form['Sleep_Hours'])
        Family_History_of_Stroke = int(request.form['Family_History_of_Stroke'])
        Education_Level = request.form['Education_Level']
        Income_Level = request.form['Income_Level']
        Stroke_Risk_Score = float(request.form['Stroke_Risk_Score'])
        
        # Create a DataFrame with the correct column names
        columns = ['Age', 'Gender', 'Hypertension', 'Heart_Disease', 'Residence_Type', 'Avg_Glucose_Level', 'BMI', 'Smoking_Status', 'Physical_Activity', 'Alcohol_Consumption', 'Chronic_Stress', 'Sleep_Hours', 'Family_History_of_Stroke', 'Education_Level', 'Income_Level', 'Stroke_Risk_Score']
        features = pd.DataFrame([[Age, Gender, Hypertension, Heart_Disease, Residence_Type, Avg_Glucose_Level, BMI, Smoking_Status, Physical_Activity, Alcohol_Consumption, Chronic_Stress, Sleep_Hours, Family_History_of_Stroke, Education_Level, Income_Level, Stroke_Risk_Score]], columns=columns)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Render result
        return render_template('index.html', prediction_text='Stroke Prediction: {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)