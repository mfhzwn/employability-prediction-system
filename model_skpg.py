import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

file_path = 'SKPG1_UteM_2019_selected1_NEW.xlsx'
df = pd.read_excel(file_path)

df.head()

"""# Preprocessing

### select columns
"""

# Update 'location_preference' based on 'e_41_e_negeri'
df.loc[df['e_41_e_negeri'].isna(), 'location_preference'] = np.nan

# Columns to change to int64
columns_to_int = ['e_41_e_negeri', 'location_preference', 'e_50_b', 'e_44']

# Fill NaN values with a placeholder and convert to int64
df[columns_to_int] = df[columns_to_int].fillna(-1).astype('int64')

# Verify the changes
print(df.dtypes)

df.head(15)

"""# Modeling

## Employability Prediction
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

employment_features = ['e_umur', 'e_cgpa', 'e_jantina', 'e_keturunan', 'e_status_kahwin', 'location_preference', 'e_sub_bidang', 'e_17', 'e_penaja', 'e_pendapatan', 'e_peringkat', 'e_15_a_i', 'e_15_a_ii', 'e_15_a_iii']

X_employment = df[employment_features]
y_employment = df['e_status'].astype(int)

# Convert NumPy array to DataFrame
df_employment = pd.DataFrame(X_employment, columns=employment_features)

# Display the DataFrame
df_employment.head()

import matplotlib.pyplot as plt

# Show the distribution of classes in the 'e_status' column
class_distribution1 = df['e_status'].value_counts()
print(class_distribution1)

# Plot the distribution
class_distribution1.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in e_status')
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_employment, y_employment, test_size=0.2, random_state=42)

# Check the distribution of each class in the training set
print(f"Number of data points in training set: {y_train.shape[0]}")
train_class_distribution = y_train.value_counts()
print("Training set class distribution:")
print(train_class_distribution)
print("-" * 60)
# Check the distribution of each class in the test set
print(f"Number of data points in test set: {y_test.shape[0]}")
test_class_distribution = y_test.value_counts()
print("Test set class distribution:")
print(test_class_distribution)

"""### Logistic Regression"""

# Train the model with regularization
employment_model = LogisticRegression(C=0.0008, max_iter=1000)
employment_model.fit(X_train, y_train)

# Evaluate the model
y_pred = employment_model.predict(X_test)
print(f"Employment Prediction")
print("Classification Report:")
print(classification_report(y_test, y_pred))

"""Cross-Validation"""

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(employment_model, X_employment, y_employment, cv=5, scoring='accuracy')

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Get the coefficients
coefficients = employment_model.coef_[0]
odds_ratios = np.exp(coefficients)
feature_importance_df = pd.DataFrame({'Feature': employment_features, 'Coefficient': coefficients, 'Odds Ratio': odds_ratios})
feature_importance_df = feature_importance_df.sort_values(by='Odds Ratio', ascending=False)

print(feature_importance_df)

"""## Random Forest"""

from sklearn.ensemble import RandomForestClassifier
# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
print(f"Employment Prediction with Random Forest")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Get feature importance
importances = rf.feature_importances_
feature_names = ['e_umur', 'e_cgpa', 'e_jantina', 'e_keturunan', 'e_status_kahwin',
                 'location_preference', 'e_sub_bidang', 'e_17', 'e_penaja', 'e_pendapatan',
                 'e_peringkat', 'e_15_a_i', 'e_15_a_ii', 'e_15_a_iii']

# Create a DataFrame to store feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance values
print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()
plt.show()

"""### Save model"""

from joblib import dump

# Save the models
dump(employment_model, 'models/employment_model.joblib')

"""### Field-Continuation Prediction"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Filter the DataFrame for rows where 'e_status' == 1 and select the field continuation features
filtered_df = df[df['e_status'] == 1][employment_features]

# Convert the filtered DataFrame to a numpy array
X_field_continuation = filtered_df.values

# Extract the target variable for field continuation
y_field_continuation = df[df['e_status'] == 1]['e_50_b'].astype(int)

# Convert NumPy array to DataFrame
df_field_continuation = pd.DataFrame(X_field_continuation, columns=employment_features)

# Display the DataFrame
df_field_continuation.head()

# Show the distribution of classes in the 'e_50_b' column
class_distribution2 = y_field_continuation.value_counts()
print(class_distribution2)

# Plot the distribution
class_distribution2.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in e_50_b')
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_field_continuation, y_field_continuation, test_size=0.2, random_state=42)

# Check the distribution of each class in the training set
print(f"Number of data points in training set: {y_train.shape[0]}")
train_class_distribution = y_train.value_counts()
print("Training set class distribution:")
print(train_class_distribution)
print("-" * 60)
# Check the distribution of each class in the test set
print(f"Number of data points in test set: {y_test.shape[0]}")
test_class_distribution = y_test.value_counts()
print("Test set class distribution:")
print(test_class_distribution)

# Define the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
    'Support Vector Classifier': SVC()
}

# Iterate over each classifier
for clf_name, clf in classifiers.items():
    print(f"Training {clf_name}...")

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Print classification report
    print(f"{clf_name} Evaluation:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 60)

"""### Hyperparamter tuning for Best Performing model

### Random Forest
"""

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV
grid_search_rf.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters:", grid_search_rf.best_params_)
print("Best Accuracy Score:", grid_search_rf.best_score_)

# Initialize SVM with best parameters
best_rf_model = RandomForestClassifier(random_state=42, max_depth=10, min_samples_leaf=3, min_samples_split=10, n_estimators=200)

# Train the model
best_rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Evaluation:")
print("Classification Report:")
print(classification_report(y_test, y_pred))

dump(best_rf_model, 'models/field_continuation_model.joblib')

"""### Salary Prediction"""

X_salary = df[df['e_status'] == 1][['e_umur', 'e_cgpa', 'e_jantina', 'e_keturunan', 'e_status_kahwin', 'location_preference', 'e_sub_bidang', 'e_17', 'e_penaja', 'e_pendapatan', 'e_peringkat', 'e_15_a_i', 'e_15_a_ii', 'e_15_a_iii']]
y_salary = df[df['e_status'] == 1]['e_44']

X_salary.head()

# Show the distribution of classes in the 'e_44' column
class_distribution3 = y_salary.value_counts()
print(class_distribution3)

# Plot the distribution
class_distribution3.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in e_44')
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_salary, y_salary, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score
from joblib import dump

# Initialize variables to track the highest accuracy and the corresponding model
best_accuracy = 0
best_model = None
best_model_name = None

# Define the classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
    'Support Vector Classifier': SVC()
}

# Iterate over each classifier
for clf_name, clf in classifiers.items():
    print(f"Training {clf_name}...")

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print classification report
    print(f"{clf_name} Evaluation:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy}")
    print("-" * 60)

    # Check if current model has higher accuracy than the best one found so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf
        best_model_name = clf_name

# Save the best model using joblib
if best_model:
    dump(best_model, 'models/salary_model.joblib')
    print(f"Best model ({best_model_name}) saved with accuracy: {best_accuracy}")
else:
    print("No model met the criteria to be saved.")
