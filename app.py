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
        PhysActivity = st.selectbox("Physical activity in past 30 days", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Fruits = st.selectbox("Consume Fruits Daily", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Veggies = st.selectbox("Consume Vegetables Daily", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        AnyHealthcare = st.selectbox("Have Health Coverage", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        NoDocbcCost = st.selectbox("Couldn't see Doctor due to cost in last 1 year", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

    with col3:
        MentHlth = st.slider("Poor Mental Health Days (last 30)", 0, 30, 5)
        PhysHlth = st.slider("Poor Physical Health Days (last 30)", 0, 30, 5)
        DiffWalk = st.selectbox("Difficulty Walking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        Sex = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

        # Age mapping
        age_categories = {
            1: "18 to 24", 2: "25 to 29", 3: "30 to 34", 4: "35 to 39",
            5: "40 to 44", 6: "45 to 49", 7: "50 to 54", 8: "55 to 59",
            9: "60 to 64", 10: "65 to 69", 11: "70 to 74", 12: "75 to 79", 13: "80 or older"
        }
        Age = st.selectbox(
            "Select Age Category (1 to 13)", 
            options=list(age_categories.keys()), 
            format_func=lambda x: f"{x}: {age_categories[x]}",
            index=8
        )

        # Education mapping
        education_categories = {
            1: "None or Only Kindergarten",
            2: "Grades 1 through 8",
            3: "Grades 9 through 11",
            4: "Grade 12",
            5: "College 1 to 3 years",
            6: "College 4+ years (Graduate)"
        }
        Education = st.selectbox(
            "Select Education Level (1 to 6)",
            options=list(education_categories.keys()),
            format_func=lambda x: f"{x}: {education_categories[x]}",
            index=3
        )

        # Income mapping
        income_categories = {
            1: "< $10K",
            2: "$10K – <$15K",
            3: "$15K – <$20K",
            4: "$20K – <$25K",
            5: "$25K – <$35K",
            6: "$35K – <$50K",
            7: "$50K – <$75K",
            8: "$75K or more"
        }
        Income = st.selectbox(
            "Select Income Level (1 to 8)",
            options=list(income_categories.keys()),
            format_func=lambda x: f"{x}: {income_categories[x]}",
            index=4
        )

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
    
