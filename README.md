# Employability Prediction System

## Project Overview
This project is a full machine learning-based Employability Prediction System developed using Python and Streamlit.

It consists of:
- Model training pipeline using SKPG UTeM 2019 dataset
- Machine learning models for prediction tasks
- Interactive web application for real-time prediction

The dataset used is not included in this repository due to institutional confidentiality.

---

## System Features

The system provides three main predictions:

### 1. Employability Prediction
Predicts whether a graduate is:
- Employed
- Unemployed

### 2. Field Continuation Prediction
Predicts whether a graduate continues working in the same field of study.

### 3. Salary Prediction
Predicts expected salary category based on graduate background.

---

## Machine Learning Models Used

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM) for baseline comparison
- GridSearchCV for Random Forest hyperparameter tuning

---

## Web Application (Streamlit)

The system includes an interactive UI built using Streamlit where users can:

- Input personal and academic details
- Get employability prediction instantly
- View field continuation prediction
- View predicted salary category

---

## Features Used for Prediction

- Age
- CGPA
- Gender
- Ethnicity
- Marital status
- Location preference
- Field of study
- Internship status
- Education sponsor
- Income category
- Level of study
- Co-curricular activity levels

---

## Model Pipeline

1. Data cleaning and preprocessing
2. Feature selection
3. Encoding categorical variables
4. Model training
5. Model evaluation
6. Model saving using Joblib
7. Deployment using Streamlit

---

## Dataset Information

The model was trained using SKPG UTeM 2019 dataset.

Note:
This dataset is not included in this repository due to confidentiality and institutional restrictions.

---

## Key Highlights

- End-to-end ML pipeline
- Multi-model comparison
- Real-world educational dataset
- Interactive prediction system
- Full deployment using Streamlit
