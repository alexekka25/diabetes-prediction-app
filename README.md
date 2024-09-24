# 🩺 Diabetes Prediction Web App 💻

Welcome to the **Diabetes Prediction** web app! This app uses machine learning models to predict the likelihood of diabetes based on user input for several health parameters. It's built using **Streamlit**, **TensorFlow**, and **scikit-learn**.

## 🎯 Key Features
- **Model Selection**: Choose between Decision Tree, Random Forest, and Artificial Neural Network (ANN) models.
- **Interactive Input Form**: Enter health data such as age, BMI, and blood glucose levels to get real-time predictions.
- **Real-time Prediction**: Instantly see the model's prediction on whether you are at risk of diabetes.
  
![App Screenshot](Screenshotfirst.png)

## 🚀 How to Run This App Locally

### 1. Clone this repository

git clone https://github.com/alexekka25/diabetes-prediction-app.git

cd diabetes-prediction-app 


### 2. Install dependencies
Make sure you have Python 3.7+ installed. Then, install the required packages:


### 3. Run the app
To start the Streamlit app, run the following command: 

streamlit run app.py


### 4. Open in your browser
The app will automatically open in your default browser. 

# 🔍 How It Works

Input Health Information: Provide details like age, BMI, blood glucose levels, and smoking history in the interactive form.

Model Selection: Select the machine learning model you want to use for prediction (Decision Tree, Random Forest, or ANN).

Prediction: Click on the "Predict Diabetes" button to see whether you're predicted to have diabetes based on the model's output.

### Input Fields
**`Gender:`** Select 'Male', 'Female', or 'Other'.

**`Age:`** Enter your age.

**`Hypertension:`** Specify whether you have hypertension (Yes/No).

**`Heart_Disease:`** Specify whether you have heart disease (Yes/No).

**`Smoking_History:`** Choose your smoking history (Never, Former, Current, etc.).
**`BMI`**: Enter your Body Mass Index (BMI).

**`HbA1c_Level:`** Provide your HbA1c level, a common diabetes metric.

**`Blood_Glucose_Level:`** Input your current blood glucose level.

# 🛠️ Model Details
 `DecisionTree:` A simple, interpretable model.

`RandomForest:` A more robust ensemble method that handles variance better.

`ANN(Artificial_Neural_Network):` A deep learning model that learns from data patterns.

# 📸 Screenshots

![App Screenshot](Screenshotsecond.png)

### 📊 Model Performance
After training, the model performances were as follows

| Model           | Accuracy | F1 Score | 
|-----------------|----------|----------|
| Decision Tree   | 0.9529   | 0.7254   | 
| Random Forest   | 0.9700   | 0.7969   | 
| ANN (Neural Net)| 0.9721   | 0.8068   | 


### 📦 Files in the Repository

`app.py:` The main file for running the Streamlit web app.

`model_training.py:` The script for training and saving machine learning models.

`requirements.txt:` Lists all necessary dependencies to install.

`README.md:` This file, explaining the project.

`models/:` Folder containing the pre-trained models.

### ⚙️ Technologies Used

Streamlit: Interactive app framework.

scikit-learn: Machine learning models.

TensorFlow: Used for building the ANN model.

Pandas/Numpy: Data manipulation and preprocessing.

Joblib: Model saving/loading.

### 👨‍💻 Author
Your Alex

GitHub: alexekka25



