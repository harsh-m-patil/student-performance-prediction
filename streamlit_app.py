import streamlit as st
import pandas as pd

st.title('Student Performance Prediction App')

st.info('ML Group project of team number 35')

with st.expander('Data'):
    st.write("**Raw Data**")
    df = pd.read_csv('./data/student-mat.csv',delimiter=';')
    df
    st.write('**X**')
    X_raw = df.drop('G3',axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.G3
    y_raw

with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('school',('GP','MS'))
    sex = st.selectbox('Gender',('M','F'))
    # min,max,avg
    age = st.slider('Age (years)',15,22,18)
    address = st.selectbox('address',('U','R'))
    famsize = st.selectbox('Family Size',('LE3','GT3'))
    Pstatus = st.selectbox('Parents Status',('T','A'))
    Medu = st.slider('Mothers Education',1,4,3)
    Fedu = st.slider('Fathers Education',1,4,3)
    Mjob = st.selectbox('Mothers Job',('at_home','health','other','services','teacher'))
    Fjob = st.selectbox('Fathers Job',('teacher', 'other', 'services', 'health','at_home'))
    reason = st.selectbox('Reason for choosing school',('course','other','home','reputation'))
    guardian = st.selectbox('Guardian',('mother','father','other'))
    traveltime = st.slider('Travel Time',1,4,2)
    studytime = st.slider('Study Time',1,4,2)
    failures = st.slider('Failures',1,4,2)
    schoolsup = st.selectbox('School Support',('yes','no'))
    famsup = st.selectbox('Family Support',('yes','no'))
    activities = st.selectbox('Extracurricular activities',('yes','no'))
    paidclass = st.selectbox('Extra paid class',('yes','no'))
    internet = st.selectbox('Internet Access',('yes','no'))
    nursery = st.selectbox('Attended Nursery',('yes','no'))
    higher = st.selectbox('Higher Education',('yes','no'))
    romantic = st.selectbox('Romantic Relationship',('yes','no'))
    freetime = st.slider('Free Time',1,5,3)
    goout = st.slider('Go out with friends',1,5,3)
    Walc = st.slider('Weekend Alcohol Consumption',1,5,2)
    Dalc = st.slider('Workday Alcohol Consumption',1,5,2)
    health = st.slider('Current health status',1,5,3)
    absence = st.slider('Absences',0,93,10)
    G1 = st.slider('Grade in term 1',0,20,15)
    G2 = st.slider('Grade in term 2',0,20,15)

    # Create a DataFrame for the input features
# Create a DataFrame for the input features
    data = {
        'school': island,
        'sex': sex,
        'age': age,
        'address': address,
        'famsize': famsize,
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
        'paidclass': paidclass,
        'internet': internet,
        'nursery': nursery,
        'higher': higher,
        'romantic': romantic,
        'freetime': freetime,
        'goout': goout,
        'Walc': Walc,
        'Dalc': Dalc,
        'health': health,
        'absence': absence,
        'G1': G1,
        'G2': G2,
    }
    
    input_df = pd.DataFrame(data,index=[0])
    input_students = pd.concat([input_df,X_raw],axis=0)

with st.expander('Input Features'):
    st.write('**Input Student**')
    input_df
    

