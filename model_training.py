# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load the dataset
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Split dataset into features (X) and target (y)
X = data.drop(columns=['diabetes'])  # 'diabetes' is the target column
y = data['diabetes']

# One-hot encode the categorical variables ('gender' and 'smoking_history')
categorical_features = ['gender', 'smoking_history']
numeric_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Using ColumnTransformer for encoding categorical data and scaling numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Apply transformation to the dataset
X = preprocessor.fit_transform(X)

# Save the preprocessor for use in Streamlit
joblib.dump(preprocessor, 'preprocessor.pkl')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)

print(f'Decision Tree - Accuracy: {dt_accuracy:.4f}, F1 Score: {dt_f1:.4f}')

# 2. Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f'Random Forest - Accuracy: {rf_accuracy:.4f}, F1 Score: {rf_f1:.4f}')

# 3. Artificial Neural Network (ANN) with TensorFlow
ann_model = Sequential()
ann_model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
ann_model.add(Dense(16, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluate ANN model
ann_loss, ann_accuracy = ann_model.evaluate(X_test, y_test, verbose=0)
y_pred_ann = (ann_model.predict(X_test) > 0.5).astype("int32")
ann_f1 = f1_score(y_test, y_pred_ann)

print(f'ANN - Accuracy: {ann_accuracy:.4f}, F1 Score: {ann_f1:.4f}')

# Save the models
joblib.dump(dt_model, 'decision_tree_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
ann_model.save('ann_model.h5')

# Print summary
print("\nModel training completed. Summary:")
print(f"Decision Tree: Accuracy={dt_accuracy:.4f}, F1 Score={dt_f1:.4f}")
print(f"Random Forest: Accuracy={rf_accuracy:.4f}, F1 Score={rf_f1:.4f}")
print(f"ANN: Accuracy={ann_accuracy:.4f}, F1 Score={ann_f1:.4f}")

# Saving performance metrics for comparison later
performance_metrics = {
    'Decision Tree': {'Accuracy': dt_accuracy, 'F1 Score': dt_f1},
    'Random Forest': {'Accuracy': rf_accuracy, 'F1 Score': rf_f1},
    'ANN': {'Accuracy': ann_accuracy, 'F1 Score': ann_f1}
}

joblib.dump(performance_metrics, 'model_performance.pkl')
