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
        HighBP = st.selectbox("High Blood Pressure", [0: No, 1: Yes])
        HighChol = st.selectbox("High Cholesterol", [0: No, 1: Yes])
        CholCheck = st.selectbox("Cholesterol Check in last 5 years", [0: No, 1: Yes])
        BMI = st.number_input("Body Mass Index", 10, 100, 25)
        Smoker = st.selectbox("Smoked at least 100 cigarettes in lifetime", [0: No, 1: Yes])
        Stroke = st.selectbox("Ever had a stroke", [0: No, 1: Yes])
        HeartDiseaseorAttack = st.selectbox("Ever had Heart Disease or Attack", [0: No, 1: Yes])

    with col2:
        PhysActivity = st.selectbox("Physical Activity", [0: No, 1: Yes])
        Fruits = st.selectbox("Consume Fruits Daily", [0: No, 1: Yes])
        Veggies = st.selectbox("Consume Vegetables Daily", [0, 1])
        HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0, 1])
        AnyHealthcare = st.selectbox("Have Healthcare Access", [0, 1])
        NoDocbcCost = st.selectbox("Couldn't see Doctor due to cost in last 1 year", [0, 1])
        GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    with col3:
        MentHlth = st.slider("Days of poor Mental Health (Days)", 0, 30, 5)
        PhysHlth = st.slider("Days of poor Physical Health (Days)", 0, 30, 5)
        DiffWalk = st.selectbox("Difficulty in Walking/Climbing", [0, 1])
        Sex = st.selectbox("Gender", [0: Female, 1: Male])
        Age = st.selectbox("Age Category (9=18-24 to 13=80+)", 1, 13, 9)
        Education = st.selectbox("Education (1 to 6)", 1, 6, 4)
        Income = st.selectbox("Income Level (1 to 8)", 1, 8, 5)

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

    st.subheader("🧾 Prediction Result")
    st.write(f"**The person is likely: `{label}`**")

    if prediction == 1:
        st.markdown(f"🟡 *Prediabetic detected. Risk of becoming diabetic: `{probabilities[2]*100:.2f}%`*")

    # Probability breakdown
    st.markdown("---")
    st.markdown("### 🔬 Prediction Probabilities")
    st.markdown(f"""
    - Non-Diabetic: **{probabilities[0]*100:.2f}%**  
    - Prediabetic: **{probabilities[1]*100:.2f}%**  
    - Diabetic: **{probabilities[2]*100:.2f}%**
    """)
    
    st.markdown("### 🛠️ Raw Debug Output")
    st.code(f"Raw probability array: {np.round(probabilities.reshape(1, -1), 5)}", language="python")

