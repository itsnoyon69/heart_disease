import streamlit as st
import numpy as np
import pickle

# ==========================================
# LOAD MODEL
# ==========================================

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("❤️ Heart Disease Prediction System")

st.write(
    "Fill up the patient medical information below."
)

st.info(
    "Reference values are provided like a medical report."
)

# ==========================================
# PATIENT INFORMATION
# ==========================================

st.header("🩺 Patient Information")

# AGE
age = st.slider(
    "Age",
    min_value=1,
    max_value=100,
    value=25,
    help="Normal Adult Age Range: 18 - 65"
)

# SEX
sex = st.selectbox(
    "Sex",
    ["Female", "Male"],
    help="Select Patient Gender"
)

# CHEST PAIN
cp = st.selectbox(
    "Chest Pain Type (cp)",
    [0, 1, 2, 3],
    help="""
0 = Typical Angina
1 = Atypical Angina
2 = Non-anginal Pain
3 = Asymptomatic
"""
)

# BLOOD PRESSURE
trestbps = st.slider(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=250,
    value=120,
    help="Normal Range: 90 - 120 mm Hg"
)

# CHOLESTEROL
chol = st.slider(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200,
    help="""
Normal: Less than 200
Borderline: 200 - 239
High: 240+
"""
)

# FASTING BLOOD SUGAR
fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1],
    help="""
0 = False
1 = True
"""
)

# ECG
restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2],
    help="""
0 = Normal
1 = ST-T Wave Abnormality
2 = Left Ventricular Hypertrophy
"""
)

# MAX HEART RATE
thalach = st.slider(
    "Maximum Heart Rate",
    min_value=60,
    max_value=250,
    value=150,
    help="Normal Adult Heart Rate: 60 - 100 bpm"
)

# EXERCISE ANGINA
exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1],
    help="""
0 = No
1 = Yes
"""
)

# OLDPEAK
oldpeak = st.slider(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    help="ST Depression Measurement"
)

# SLOPE
slope = st.selectbox(
    "Slope",
    [0, 1, 2],
    help="""
0 = Upsloping
1 = Flat
2 = Downsloping
"""
)

# CA
ca = st.selectbox(
    "Number of Major Vessels (ca)",
    [0, 1, 2, 3, 4],
    help="0 - 4 Major Vessels"
)

# THAL
thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3],
    help="""
0 = Unknown
1 = Normal
2 = Fixed Defect
3 = Reversible Defect
"""
)

# ==========================================
# CONVERT SEX
# ==========================================

if sex == "Male":
    sex = 1
else:
    sex = 0

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Predict Heart Disease"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Scaling
    input_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    # ==========================================
    # RESULT
    # ==========================================

    st.header("📋 Medical Prediction Report")

    st.write(
        f"### Heart Disease Probability: "
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK: Heart Disease Detected"
        )

        st.warning(
            "Recommendation: Please consult a cardiologist immediately."
        )

    else:

        st.success(
            "✅ LOW RISK: No Heart Disease Detected"
        )

        st.info(
            "Recommendation: Maintain a healthy lifestyle."
        )