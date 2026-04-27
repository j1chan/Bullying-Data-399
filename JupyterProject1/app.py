import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("model.pkl")
X_columns = joblib.load("model_columns.pkl")

st.title("Students Being Bullied on School Property Predictor")

st.write("Enter student information:")


sex = st.selectbox("Sex", ["Male", "Female"])

age = st.selectbox("Age", [
    "11 years old or younger",
    "12 years old",
    "13 years old",
    "14 years old",
    "15 years old",
    "16 years old",
    "17 years old",
    "18 years old or older"
])

close_friends = st.selectbox("Close Friends", ["0", "1", "2", "3 or more"])

loneliness = st.selectbox("Loneliness Level", [
    "Never",
    "Rarely",
    "Sometimes",
    "Most of the time",
    "Always"
])

kindness = st.selectbox("Other students are kind and helpful", [
    "Never",
    "Rarely",
    "Sometimes",
    "Most of the time",
    "Always"
])


x_df = pd.DataFrame([np.zeros(len(X_columns))], columns=X_columns)


def set_feature(prefix, value):
    col_name = f"{prefix}_{value}"
    if col_name in x_df.columns:
        x_df[col_name] = 1


set_feature("Age", age)
set_feature("Sex", sex)
set_feature("Close Friends", close_friends)
loneliness_map = {
    "Never": "1",
    "Rarely": "2",
    "Sometimes": "3",
    "Most of the time": "4",
    "Always": "5"
}

loneliness_val = loneliness_map[loneliness]

set_feature("Felt Lonely", loneliness_val)
set_feature("Other Students Kind and Helpful", kindness)


if st.button("Predict Bullying Risk"):

    prob = model.predict_proba(x_df)[0, 1]

    st.subheader("Result")

    st.write(f"**Probability of bullying: {prob:.2f}**")

    if prob >= 0.5:
        st.error("High risk of bullying")
    else:
        st.success("Low risk of bullying")