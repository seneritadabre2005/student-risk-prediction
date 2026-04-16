import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("student_model.pkl")

# Load data
data = pd.read_csv("data/student_data.csv")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Dashboard", "Prediction"])

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("📊 Student Analytics Dashboard")

    st.subheader("Dataset")
    st.dataframe(data)

    st.subheader("Risk Distribution")
    st.bar_chart(data["Risk"].value_counts())

    st.subheader("Attendance vs Marks")
    st.scatter_chart(data[["Attendance", "Internal_Marks"]])

    # Real-time simulation
    if st.button("Add New Student"):
        new_data = {
            "Attendance": np.random.randint(30, 100),
            "Internal_Marks": np.random.randint(20, 100),
            "Assignment_Score": np.random.randint(20, 100),
            "Study_Hours": np.random.randint(1, 10),
            "Sleep_Hours": np.random.randint(4, 10),
            "Risk": np.random.choice([0, 1])
        }

        data = pd.concat([data, pd.DataFrame([new_data])], ignore_index=True)
        data.to_csv("../data/student_data.csv", index=False)

        st.success("New student added!")

# ---------------- PREDICTION ----------------
else:
    st.title("🎯 Student Risk Prediction")

    attendance = st.slider("Attendance", 0, 100)
    marks = st.slider("Internal Marks", 0, 100)
    assignment = st.slider("Assignment Score", 0, 100)
    study = st.slider("Study Hours", 0, 10)
    sleep = st.slider("Sleep Hours", 0, 10)

    if st.button("Predict"):
        input_data = np.array([[attendance, marks, assignment, study, sleep]])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Student is AT RISK")
        else:
            st.success("✅ Student is NOT AT RISK")

        prob = model.predict_proba(input_data)
        st.write("Confidence:", prob)