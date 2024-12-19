import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define numerical columns used in training
numerical_cols = [
    'absence_days', 'weekly_self_study_hours', 'math_score',
    'history_score', 'physics_score', 'chemistry_score',
    'biology_score', 'english_score', 'geography_score'
]

# Streamlit app
st.title("Student Abandon Prediction")

st.header("Enter Student Data")
# User inputs for numerical columns
numerical_inputs = {}
for col in numerical_cols:
    if col in ['absence_days', 'weekly_self_study_hours']:
        numerical_inputs[col] = st.number_input(
            f"Enter {col.replace('_', ' ').capitalize()}",
            value=0,
            step=1,
            format="%d"
        )
    else:
        numerical_inputs[col] = st.number_input(
            f"Enter {col.replace('_', ' ').capitalize()}",
            value=0.0
        )

# User inputs for categorical columns
gender = st.radio("Gender", ['Female', 'Male'])
part_time_job = st.radio("Has Part-Time Job?", ['No', 'Yes'])
extracurricular_activities = st.radio("In Extracurricular Activities?", ['No', 'Yes'])
career_aspiration = st.radio("Career Aspiration", ['Other', 'Professional', 'Academic', 'Entrepreneurial'])

# Map user input to label encoding
categorical_inputs = {
    'gender': 1 if gender == 'Male' else 0,
    'part_time_job': 1 if part_time_job == 'Yes' else 0,
    'extracurricular_activities': 1 if extracurricular_activities == 'Yes' else 0,
    'career_aspiration': {
        'Other': 0,
        'Professional': 1,
        'Academic': 2,
        'Entrepreneurial': 3
    }[career_aspiration]
}

# Prepare input data
numerical_data = pd.DataFrame([numerical_inputs], columns=numerical_cols)
categorical_data = pd.DataFrame([categorical_inputs], columns=categorical_inputs.keys())

# Combine numerical and categorical data
input_data = pd.concat([numerical_data, categorical_data], axis=1)

# Standardize the numerical data
scaler = StandardScaler()
input_data[numerical_cols] = scaler.fit_transform(input_data[numerical_cols])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    # Display result
    if prediction[0] == 1:
        st.error(f"The model predicts that the student is likely to abandon the course. (Confidence: {prediction_proba[0]:.2f})")
    else:
        st.success(f"The model predicts that the student is unlikely to abandon the course. (Confidence: {1 - prediction_proba[0]:.2f})")
