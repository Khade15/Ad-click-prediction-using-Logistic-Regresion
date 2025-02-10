import streamlit as st
import pickle
import numpy as np

# Load the logistic regression model
MODEL_FILE = "logistic_regression_model.pkl"

try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    model_loaded = True
except (FileNotFoundError, pickle.PickleError) as e:
    model_loaded = False
    error_message = str(e)

# Streamlit UI
st.title("Ad Click Prediction using Logistic Regression")
st.write("Enter values to predict whether a user will click on the ad.")

if model_loaded:
    # Number of input features (update based on your dataset)
    num_features = model.coef_.shape[1]  
    input_values = []

    for i in range(num_features):
        value = st.number_input(f"Feature {i+1}", value=0.0)
        input_values.append(value)

    if st.button("Predict"):
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)
        st.write("Prediction:", "User will click the ad" if prediction[0] == 1 else "User will NOT click the ad")
else:
    st.error(f"Error loading model: {error_message}")
    st.write("Ensure 'logistic_regression_model.pkl' exists in the same directory.")
