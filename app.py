import streamlit as st
from joblib import load
import numpy as np

# Load the models
employment_model = load('models/employment_model.joblib')
field_continuation_model = load('models/field_continuation_model.joblib')
salary_model = load('models/salary_model.joblib')

# Mapping dictionaries
gender_mapping = {'Lelaki': 1, 'Perempuan': 2}
ethnicity_mapping = {
    'Bukan Warganegara': 23, 'Melayu': 100, 'Minangkabau': 106, 'Cina': 200, 'India': 300,
    'Malayali': 301, 'Melayu Sri Lanka': 602, 'Bajau': 801, 'Kadazan': 803, 'Brunei': 904,
    'Melanau': 913, 'Iban': 1004, 'Bidayuh': 1005, 'Orang Asli Semenanjung': 1200
}
marital_status_mapping = {
    'Bujang': 1, 'Berkahwin': 2, 'Bercerai': 3, 'Berpisah': 4, 'Duda': 5, 'Balu/Janda': 6
}
income_status_mapping = {'B40': 1, 'M40/T20': 2}
location_preference_mapping = {'Yes': 1, 'No': 2, 'None': -1}
study_field_mapping = {
    'PENTADBIRAN PERNIAGAAN/PERDAGANGAN': 104,
    'KEJURUTERAAN ELEKTRIK/ELEKTRONIK/TELEKOMUNIKASI': 303,
    'KEJURUTERAAN MEKANIKAL/MEKATRONIK': 304,
    'LAIN-LAIN KEJURUTERAAN': 306,
    'SAINS KOMPUTER': 401
}
internship_mapping = {'Yes': 1, 'No': 2}
education_sponsor_mapping = {
    'Jabatan Perkhidmatan Awam (JPA)': 1, 'PTPTN': 2, 'Majlis Amanah Rakyat (MARA)': 3,
    'Kerajaan Negeri/Yayasan Negeri (Malaysia)': 4, 'Persendirian (Ibu bapa/Keluarga/Sendiri)': 5,
    'Lain-lain': 6, 'Kementerian Pendidikan Tinggi Malaysia': 8,
    'Tenaga Nasional Berhad (TNB)': 11, 'Majlis Agama Islam Negeri': 41,
    'Yayasan Tunku Abdul Rahman': 1102, 'Permodalan Nasional Berhad (PNB)': 1208
}
level_of_study_mapping = {
    'Diploma': 1, 'Ph.D': 3, 'Ijazah Pertama/Sarjana Muda': 4, 'Sarjana': 5, 'DEng': 17
}
activity_level_mapping = {
    'Sangat tidak aktif': 1, 'Tidak aktif': 2, 'Sederhana aktif': 3,
    'Aktif': 4, 'Sangat aktif': 5, 'Tidak Berkenaan': 9
}

# Mapping for salary categories with labels
salary_category_mapping = {
    13: 'RM1000 dan ke bawah (13)',
    3: 'RM1001 - RM1500 (3)',
    4: 'RM1501 - RM2000 (4)',
    5: 'RM2001 - RM2500 (5)',
    6: 'RM2501 - RM3000 (6)',
    9: 'RM3001 – RM4000 (9)',
    10: 'RM4001 – RM5000 (10)',
    11: 'RM5001 – RM10000 (11)',
    12: 'Lebih daripada RM10000 (12)'
}

# Function to make predictions
def predict_employability(features):
    return employment_model.predict([features])[0]


def predict_field_continuation(features):
    return field_continuation_model.predict([features])[0]


def predict_salary(features):
    return salary_model.predict([features])[0]


# Streamlit App
st.title('Career Prediction App')

# Input fields for employability prediction
st.header('Input Data')
e_umur = st.number_input('Age', min_value=0, max_value=100)
e_cgpa = st.number_input('CGPA', min_value=0.0, max_value=4.0, step=0.01)
e_jantina = st.selectbox('Gender', list(gender_mapping.keys()))
e_keturunan = st.selectbox('Ethnicity', list(ethnicity_mapping.keys()))
e_status_kahwin = st.selectbox('Marital Status', list(marital_status_mapping.keys()))
location_preference = st.selectbox('Prefer Work Near Home', list(location_preference_mapping.keys()))
e_sub_bidang = st.selectbox('Study field', list(study_field_mapping.keys()))
e_17 = st.selectbox('Internship', list(internship_mapping.keys()))
e_penaja = st.selectbox('Education Sponsor', list(education_sponsor_mapping.keys()))
e_pendapatan = st.selectbox('Household Income Status', list(income_status_mapping.keys()))
e_peringkat = st.selectbox('Level of Study', list(level_of_study_mapping.keys()))
e_15_a_i = st.selectbox('Co-curricular Activity Level (Persatuan)', list(activity_level_mapping.keys()))
e_15_a_ii = st.selectbox('Co-curricular Activity Level (Kelab)', list(activity_level_mapping.keys()))
e_15_a_iii = st.selectbox('Co-curricular Activity Level (Sukan)', list(activity_level_mapping.keys()))

# Encode the input values
employment_features = [
    e_umur,
    e_cgpa,
    gender_mapping[e_jantina],
    ethnicity_mapping[e_keturunan],
    marital_status_mapping[e_status_kahwin],
    location_preference_mapping[location_preference],
    study_field_mapping[e_sub_bidang],
    internship_mapping[e_17],
    education_sponsor_mapping[e_penaja],
    income_status_mapping[e_pendapatan],
    level_of_study_mapping[e_peringkat],
    activity_level_mapping[e_15_a_i],
    activity_level_mapping[e_15_a_ii],
    activity_level_mapping[e_15_a_iii]
]

# # Debugging: Print input features to inspect
# st.write("Encoded Input Features:", employment_features)

if st.button('Predict Employability'):
    st.header('Employability Prediction')
    employability_status = predict_employability(employment_features)
    st.write(f'Employability Status: {"Employed" if employability_status == 1 else "Unemployed"}')

    if employability_status == 1:  # If employed
        st.header('Field Continuation Prediction')
        field_cont_prob = predict_field_continuation(employment_features)
        st.write(f'Field Continuation Prediction: {"Yes" if field_cont_prob == 1 else "No"}')

        st.header('Salary Prediction')
        salary_label = predict_salary(employment_features)
        predicted_salary_category = salary_category_mapping.get(salary_label, 'Unknown')
        st.write(f'Predicted Salary Category: {predicted_salary_category}')
