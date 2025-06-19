import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

model = load_model('ANN.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    gender_lbe = pickle.load(file)

with open('ohe_geography.pkl', 'rb') as file:
    geo_ohe = pickle.load(file)

with open('standard_scaler.pkl', 'rb') as file:
    std_scaler = pickle.load(file)  



# Web app code

st.title('Churn likeliness...')

gender = st.selectbox('Gender', gender_lbe.classes_)
age = st.slider('Age', 18, 92)
credit_score = st.number_input('Credit Score')
balance = st.number_input('Balance')
geography = st.selectbox('Geography', geo_ohe.categories_[0])
tenure = st.slider('Tenure', 0, 10)
estimated_salary = st.number_input('Estimated Salary')
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
num_products = st.slider('Number of Products', 1, 4)
is_active = st.selectbox('Is Active Member', [0, 1])

user_input = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender_lbe.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [estimated_salary]
})

input_data = pd.DataFrame(user_input)

geo_data = geo_ohe.transform([input_data['Geography']]).toarray()
geo_data = pd.DataFrame(data=geo_data, columns=geo_ohe.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.drop(['Geography'], axis=1), geo_data], axis=1)

scaled_input = std_scaler.transform(input_data)

prediction = model.predict(scaled_input)
prediction_result = prediction[0][0]


st.write(f'Churn Probability: {prediction_result:.2f}')