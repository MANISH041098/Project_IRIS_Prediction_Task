import streamlit as st
from pickle import load
import numpy as np


scaler = load(open('models\standard_scaler.pkl','rb'))
lr_model = load(open('models\lr_model.pkl','rb'))

sl = st.text_input("Sepal Length",placeholder="Enter value in cm")
sw = st.text_input("Sepal Width",placeholder="Enter value in cm")
pl = st.text_input("Petal Length",placeholder="Enter value in cm")
pw = st.text_input("Petal Width",placeholder="Enter value in cm")

btn_ = st.button("Predict")

if btn_ == True:
    if sl and sw and pl and pw:
        query = np.array([float(sl),float(sw),float(pl),float(pw)]).reshape(1,-1)
        query_trans = scaler.transform(query)
        pred = lr_model.predict(query_trans)
        st.success(pred)
    else:
        st.error('Enter the values properly ')


    