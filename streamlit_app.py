import streamlit as st
import pandas as pd
import pickle
from model import DecisionTreeRegressorScratch, RandomForestRegressorScratch

# Load the model from the pickle file
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Student Performance Prediction App')

st.info('ML Group project of team number 35')

with st.expander('Data'):
    st.write("**Raw Data**")
    df = pd.read_csv('./data/student-mat.csv', delimiter=';')
    df
    st.write('**X**')
    X_raw = df.drop('G3', axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.G3
    y_raw

with st.expander('Data Visualization'):
    st.scatter_chart(data=df, x='G1', y='G2', color='G3')

with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('school', ('GP', 'MS'))
    sex = st.selectbox('Gender', ('M', 'F'))
    # min, max, avg
    age = st.slider('Age (years)', 15, 22, 18)
    address = st.selectbox('address', ('U', 'R'))
    famsize = st.selectbox('Family Size', ('LE3', 'GT3'))
    famrel = st.slider('Quality of family Relationship', 1, 5, 3)
    Pstatus = st.selectbox('Parents Status', ('T', 'A'))
    Medu = st.slider('Mothers Education', 1, 4, 3)
    Fedu = st.slider('Fathers Education', 1, 4, 3)
    Mjob = st.selectbox('Mothers Job', ('at_home', 'health', 'other', 'services', 'teacher'))
    Fjob = st.selectbox('Fathers Job', ('teacher', 'other', 'services', 'health', 'at_home'))
    reason = st.selectbox('Reason for choosing school', ('course', 'other', 'home', 'reputation'))
    guardian = st.selectbox('Guardian', ('mother', 'father', 'other'))
    traveltime = st.slider('Travel Time', 1, 4, 2)
    studytime = st.slider('Study Time', 1, 4, 2)
    failures = st.slider('Failures', 1, 4, 2)
    schoolsup = st.selectbox('School Support', ('yes', 'no'))
    famsup = st.selectbox('Family Support', ('yes', 'no'))
    activities = st.selectbox('Extracurricular activities', ('yes', 'no'))
    paid = st.selectbox('Extra paid class', ('yes', 'no'))
    internet = st.selectbox('Internet Access', ('yes', 'no'))
    nursery = st.selectbox('Attended Nursery', ('yes', 'no'))
    higher = st.selectbox('Higher Education', ('yes', 'no'))
    romantic = st.selectbox('Romantic Relationship', ('yes', 'no'))
    freetime = st.slider('Free Time', 1, 5, 3)
    goout = st.slider('Go out with friends', 1, 5, 3)
    Walc = st.slider('Weekend Alcohol Consumption', 1, 5, 2)
    Dalc = st.slider('Workday Alcohol Consumption', 1, 5, 2)
    health = st.slider('Current health status', 1, 5, 3)
    absence = st.slider('Absences', 0, 93, 10)
    G1 = st.slider('Grade in term 1', 0, 20, 15)
    G2 = st.slider('Grade in term 2', 0, 20, 15)

    # Create a DataFrame for the input features
    data = {
        'school': island,
        'sex': sex,
        'age': age,
        'address': address,
        'famsize': famsize,
        'famrel': famrel,
        'Pstatus': Pstatus,
        'Medu': Medu,
        'Fedu': Fedu,
        'Mjob': Mjob,
        'Fjob': Fjob,
        'reason': reason,
        'guardian': guardian,
        'traveltime': traveltime,
        'studytime': studytime,
        'failures': failures,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'activities': activities,
        'paid': paid,
        'internet': internet,
        'nursery': nursery,
        'higher': higher,
        'romantic': romantic,
        'freetime': freetime,
        'goout': goout,
        'Walc': Walc,
        'Dalc': Dalc,
        'health': health,
        'absences': absence,
        'G1': G1,
        'G2': G2,
    }

    input_df = pd.DataFrame(data, index=[0])

with st.expander('Input Features'):
    st.write('**Input Student**')
    input_df

# Data preparation

encoding_mappings = {
    'school': {'GP': 1, 'MS': 2},
    'sex': {'M': 1, 'F': 2},
    'address': {'U': 2, 'R': 1},
    'famsize': {'LE3': 1, 'GT3': 2},
    'Pstatus': {'A': 1, 'T': 2},
    'Mjob': {'at_home': 1, 'health': 3, 'other': 2, 'services': 4, 'teacher': 5},
    'Fjob': {'at_home': 1, 'health': 3, 'other': 2, 'services': 4, 'teacher': 5},
    'reason': {'course': 3, 'other': 1, 'home': 2, 'reputation': 4},
    'guardian': {'mother': 2, 'father': 3, 'other': 1},
    'schoolsup': {'yes': 2, 'no': 1},
    'famsup': {'yes': 2, 'no': 1},
    'activities': {'yes': 2, 'no': 1},
    'paid': {'yes': 2, 'no': 1},
    'internet': {'yes': 2, 'no': 1},
    'nursery': {'yes': 2, 'no': 1},
    'higher': {'yes': 2, 'no': 1},
    'romantic': {'yes': 2, 'no': 1},
}

# Encode the input data
encoded_input = input_df.replace(encoding_mappings)

# Encode the raw dataset (X_raw)
encoded_X_raw = X_raw.replace(encoding_mappings)

# Align columns of encoded_input with encoded_X_raw
encoded_input = encoded_input[encoded_X_raw.columns]

with st.expander('Data preparation'):
    st.write('**Encoded Input Data**')
    st.write(encoded_input)  # Show only the first row (user input)
    st.write('**Encoded Raw Data**')
    st.write(encoded_X_raw)  # Show only the first row (user input)

# Convert the DataFrame to an array
input_array = encoded_input.values

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_array)
    st.write(f"The predicted grade (G3) for the student is: {prediction[0]:.2f}")
