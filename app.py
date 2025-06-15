import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define label map
label_map = {
    0: "Non-Diabetic",
    1: "Prediabetic",
    2: "Diabetic"
}

st.set_page_config(page_title="Diabetes Prediction System", layout="centered")
st.title("Diabetes Prediction System")
st.markdown("Provide the following health and lifestyle details to predict your diabetes status.")

# Feature inputs (21)
def get_user_input():
    col1, col2, col3 = st.columns(3)

    with col1:
        HighBP = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        HighChol = st.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        CholCheck = st.selectbox("Cholesterol Check in last 5 years", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        BMI = st.number_input("Body Mass Index", 10, 100, 25)
        Smoker = st.selectbox("Smoked at least 100 cigarettes in lifetime", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Stroke = st.selectbox("Ever had a stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        HeartDiseaseorAttack = st.selectbox("Ever had Heart Disease or Attack", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col2:
        PhysActivity = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Fruits = st.selectbox("Consume Fruits Daily", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Veggies = st.selectbox("Consume Vegetables Daily", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        AnyHealthcare = st.selectbox("Have Healthcare Access", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        NoDocbcCost = st.selectbox("Couldn't see Doctor due to cost in last 1 year", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    with col3:
        MentHlth = st.slider("Days of poor Mental Health (Days)", 0, 30, 5)
        PhysHlth = st.slider("Days of poor Physical Health (Days)", 0, 30, 5)
        DiffWalk = st.selectbox("Difficulty in Walking/Climbing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Sex = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        Age = st.selectbox("Age Category (9=18-24 to 13=80+)", list(range(1, 14)), index=8)
        Education = st.selectbox("Education (1 to 6)", list(range(1, 7)), index=3)
        Income = st.selectbox("Income Level (1 to 8)", list(range(1, 9)), index=4)

    input_data = pd.DataFrame([[
        HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
        PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
        NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
    ]], columns=[
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
        "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
        "Education", "Income"
    ])

    return input_data

input_df = get_user_input()

if st.button("🔍 Predict Diabetes Risk"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    label = label_map[prediction]

    st.subheader("🧮 Prediction Result")
    st.write(f"**The person is likely: `{label}`**")

    if prediction == 1:
        st.markdown(f"💡 *Prediabetic detected. Risk of becoming diabetic: `{probabilities[2]*100:.2f}%`*")

    # Probability breakdown
    #st.markdown("---")
    #st.markdown("### 🔢 Prediction Probabilities")
    #st.markdown(f"""
    #- Non-Diabetic: **{probabilities[0]*100:.2f}%**  
    #- Prediabetic: **{probabilities[1]*100:.2f}%**  
    #- Diabetic: **{probabilities[2]*100:.2f}%**
    #""")

    #st.markdown("### 🧪 Raw Debug Output")
    #st.code(f"Raw probability array: {np.round(probabilities.reshape(1, -1), 5)}", language="python")
    
