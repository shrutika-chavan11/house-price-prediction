import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("House Price Predictor")

st.write("Enter house details below:")

area = st.number_input("GrLivArea", 500, 10000, 1500)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
quality = st.number_input("Overall Quality", 1, 10, 5)

if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, quality]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.0f}")