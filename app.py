import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf

# Load the trained models and preprocessor
dt_model = joblib.load('decision_tree_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
ann_model = tf.keras.models.load_model('ann_model.h5')
preprocessor = joblib.load('preprocessor.pkl')

# Define a function for prediction
def predict_diabetes(model, input_data):
    # Create DataFrame with the same structure as the training data
    input_df = pd.DataFrame([input_data], columns=[
        'gender', 'age', 'hypertension', 'heart_disease',
        'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
    ])
    # Preprocess the input data
    processed_data = preprocessor.transform(input_df)
    
    # Predict using the selected model
    if model == 'Decision Tree':
        prediction = dt_model.predict(processed_data)
    elif model == 'Random Forest':
        prediction = rf_model.predict(processed_data)
    elif model == 'ANN':
        prediction = (ann_model.predict(processed_data) > 0.5).astype("int32")
    
    return 'Diabetic' if prediction == 1 else 'Not Diabetic'

# Streamlit app layout



# Add an image (logo or banner)
st.image('_f7396f64-c815-445f-a3e9-ca4c73fb7a9e.jpg', use_column_width=True)  # This will adjust the width to fit the screen

st.title("Diabetes Prediction App")

st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Select a Model", ('Decision Tree', 'Random Forest', 'ANN'))

st.header("Enter your health information:")

# Input fields for features
gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
age = st.slider('Age', 0, 120, 30)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
smoking_history = st.selectbox('Smoking History', ['No Info', 'never', 'former', 'current', 'not current', 'ever'])
bmi = st.slider('BMI', 10.0, 60.0, 25.0)
hba1c_level = st.slider('HbA1c Level', 2.0, 20.0, 5.0)
blood_glucose_level = st.slider('Blood Glucose Level', 50, 300, 100)

# Mapping inputs to the correct format
gender_map = {'Female': 'Female', 'Male': 'Male', 'Other': 'Other'}
hypertension_map = {'No': 0, 'Yes': 1}
heart_disease_map = {'No': 0, 'Yes': 1}
smoking_map = {
    'No Info': 'No Info',
    'never': 'never',
    'former': 'former',
    'current': 'current',
    'not current': 'not current',
    'ever': 'ever'
}

# Convert inputs to match model training data format
input_data = {
    'gender': gender_map[gender],
    'age': age,
    'hypertension': hypertension_map[hypertension],
    'heart_disease': heart_disease_map[heart_disease],
    'smoking_history': smoking_map[smoking_history],
    'bmi': bmi,
    'HbA1c_level': hba1c_level,
    'blood_glucose_level': blood_glucose_level
}

# Prediction button
if st.button("Predict Diabetes"):
    prediction = predict_diabetes(model_choice, input_data)
    st.write(f"The model predicts: **{prediction}**")


import matplotlib.pyplot as plt

# Sample data
sizes = [70, 30]  # Replace with your actual data
labels = ['Non-Diabetic', 'Diabetic']
colors = ['#4CAF50', '#F44336']  # Green for Non-Diabetic, Red for Diabetic
explode = (0.1, 0)  # Explode the 1st slice (Non-Diabetic)

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
       colors=colors, explode=explode, shadow=True)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')  

# Add a title
plt.title('Diabetes Prediction Distribution')

# Display the plot in Streamlit
st.pyplot(fig)


