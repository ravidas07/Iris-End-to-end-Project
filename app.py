# Import required Packages
import streamlit as st
import pandas as pd
import joblib

# Load the model object
model = joblib.load("notebook/iris_model.joblib")

# Fuction to predict
def predict_label(model, sep_len, sep_wid, pet_len, pet_wid):
    xnew = [
        {
            "sepal_length": sep_len,
            "sepal_width": sep_wid,
            "petal_length": pet_len,
            "petal_width": pet_wid
        }
    ]
    df_xnew = pd.DataFrame(xnew)
    pred = model.predict(df_xnew)
    prob = model.predict_proba(df_xnew)

    res_prob = {}
    # Get probability as Dictionary
    for c, p in zip(model.classes_,prob.flatten()):
        res_prob[c] = p.round(4)

    return pred[0] , res_prob

# Build the streamlit app

st.set_page_config(page_title="Iris Project")
# Add title
st.title("Iris Prediction Project")
st.subheader("by Ravi") 

# Create number input for predic
sep_len = st.number_input("Sepal Length : " , min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width : ", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length : " , min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width : " , min_value=0.00, step=0.01)

# Create Button for prediction
button = st.button("Predict" , type = "primary")

# After Button pressed
if button:
    pred, prob = predict_label(model, sep_len, sep_wid, pet_len, pet_wid)
    st.subheader(f"Predicted Species : {pred}")
    for c, p in prob.items():
        st.subheader(f"{c} : {p}")
        st.progress(p)