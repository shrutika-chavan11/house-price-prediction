import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

print("\n--- House Price Predictor ---")

# Input
area = float(input("GrLivArea (sq ft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Full Bathrooms: "))
quality = int(input("Overall Quality (1-10): "))

# Prepare input in SAME order as training
import pandas as pd

input_data = pd.DataFrame([{
    "GrLivArea": area,
    "BedroomAbvGr": bedrooms,
    "FullBath": bathrooms,
    "OverallQual": quality
}])

# Predict
prediction = model.predict(input_data)

print("\nPredicted House Price: ", round(prediction[0], 2), "rs")