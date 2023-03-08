import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import keras
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from keras.utils import pad_sequences

## Loading the ann model
model = keras.models.load_model('heart_ann_model')

### Opening a file with Pickle (Logistic Regression model was saved as Pickle (binary) format)
#with open(r'C:\Users\User\Pablo\Data Science Bootcamp\Phase 2\Stream 2 - Specific Algorithms\Week 10.2 - Supervised machine learning  - classification 1\REG & CLASS\logistic_regression_model.pkl', 'rb') as file:
 #         model = pickle.load(file)

## load the copy of the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

## set page configuration
st.set_page_config(page_title= 'Death Classifier', layout='wide')

## add page title and content
st.title('Heart Disease Classification using Artificial Neural Network')
st.write('''
 Heart disease Prediction App.

 This app predicts if a patient has a heart disease and therefore would potentially die.
''')

#st.sidebar.header('User Input Features')

## add image
image = Image.open('cdd20-7sUlk1-PLZ0-unsplash.jpg')
st.image(image, width=800)


## get user imput
#email_text = st.text_input('Email Text:')
st.sidebar.header('User Input Features')
def user_input_features():
    age = st.sidebar.number_input("Enter your age:")
    anaemia = st.sidebar.selectbox("Enter anaemia",(0,1))
    creatinine_phosphokinase = st.sidebar.number_input("Enter creatinine_phosphokinase:")
    diabetes = st.sidebar.selectbox("Enter diabetes",(0,1))
    ejection_fraction = st.sidebar.number_input("Enter ejection_fraction:")
    high_blood_pressure = st.sidebar.selectbox("Enter high_blood_pressure",(0,1))
    platelets = st.sidebar.number_input("Enter platelets:")
    serum_creatinine = st.sidebar.number_input("Enter serum_creatinine:")
    serum_sodium = st.sidebar.number_input("Enter serum_sodium:")
    sex = st.sidebar.selectbox("Enter sex",(0,1))
    smoking = st.sidebar.selectbox("Enter smoking",(0,1))
    time = st.sidebar.number_input("Enter time:")

    data = {'age':age,
            'anaemia':anaemia,
            'creatinine_phosphokinase':creatinine_phosphokinase,
            'diabetes':diabetes,
            'ejection_fraction':ejection_fraction,
            'high_blood_pressure':high_blood_pressure,
            'platelets':platelets,
            'serum_creatinine':serum_creatinine,
            'serum_sodium':serum_sodium,
            'sex':sex,
            'smoking':smoking,
            'time':time
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.write(input_df)
## Combines user input with entire dataset
data = data.drop(columns=['DEATH_EVENT'])
df = pd.concat([input_df,data],axis=0)
# ## Standarising 
col_names = list(df.columns)
s_scaler = StandardScaler()
X_norm= s_scaler.fit_transform(df)
X_norm = pd.DataFrame(X_norm, columns=col_names) 

df2 = X_norm[:1]
#df2 = df2.flatten()

#st.write(df2)

# Make prediction
prediction = model.predict(df2)

st.subheader('Prediction')
st.write(prediction)

if prediction > 0.5:
    st.write('You will be likely to have heart disease and, as a result, die')
else:
    st.write('You will not be likely to have a heart failure')

#print(classification_report(y_test, prediction))
