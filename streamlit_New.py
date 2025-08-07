# Import necessary libraries
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
import os
import pickle  # Only use pickle, not joblib since only pickle was used to save

# Title and description
st.title("Attorney Prediction App")
st.write("This app predicts the type of attorney based on user input.")

# Check if the model file exists
model_path = "logistic.pkl"
if not os.path.isfile(model_path):
    st.error(f"Model file '{model_path}' not found. Please upload or check deployment.")
    st.stop()

# Load the model
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# UI: Example input fields (customize based on your model)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Predict"):
    try:
        prediction = model.predict([[feature1, feature2, feature3]])
        st.success(f"Predicted Attorney Type: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
