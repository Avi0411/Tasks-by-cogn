import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Load model files
scaler = joblib.load(BASE_DIR / "Scaler.pkl")
model = joblib.load(BASE_DIR / "mlmodel.pkl")

st.set_page_config(layout="wide")

st.title("Restaurant Rating Prediction App")

st.caption("This app helps you predict a restaurant rating.")

st.divider()

averagecost = st.number_input("Please enter the estimated average cost for two", min_value=50, max_value=999999, value = 1000, step = 200 )

tablebooking = st.selectbox("Restaurant has table booking?",["Yes", "No"])

onlinedelivery = st.selectbox("Restaurant has online booking?", ["Yes", "No"])

pricerange = st.selectbox("What is the price range (1 Cheapest, 4 Most Expensive)",[1,2,3,4])

predictbutton = st.button("Predict the review")

st.divider()

model = joblib.load(BASE_DIR / "mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0

deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost,bookingstatus,deliverystatus,pricerange]]

my_X_values = np.array(values)

X = scaler.transform(my_X_values)


if predictbutton:
    st.snow()

    prediction = model.predict(X)

    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")

    
