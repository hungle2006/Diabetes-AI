import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, confusion_matrix
import uuid

# Load data
data = pd.read_csv('/content/drive/MyDrive/kaggle_data/Data_other/diabetes_prediction_dataset.csv')

# Key features
key_features = [
    'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'hypertension', 'heart_disease'
]

# Feature engineering
# Encode gender
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
key_features.append('gender')

# Encode smoking_history
smoking_mapping = {
    'never': 0, 'No Info': 1, 'former': 2, 'current': 3,
    'not current': 4, 'ever': 5
}
data['smoking_history'] = data['smoking_history'].map(smoking_mapping)
key_features.append('smoking_history')

# Define severity labels
NORMAL = 0
PREDIABETES = 1
DIABETES = 2
SEVERE_DIABETES = 3

# Labeling Functions
def hba1c_lf(row):
    if pd.isna(row['HbA1c_level']):
        return -1
    if row['HbA1c_level'] >= 7.0:
        return SEVERE_DIABETES
    elif row['HbA1c_level'] >= 6.5:
        return DIABETES
    elif row['HbA1c_level'] >= 5.7:
        return PREDIABETES
    else:
        return NORMAL

def glucose_lf(row):
    if pd.isna(row['blood_glucose_level']):
        return -1
    if row['blood_glucose_level'] >= 200:
        return SEVERE_DIABETES
    elif row['blood_glucose_level'] >= 140:
        return DIABETES
    elif row['blood_glucose_level'] >= 100:
        return PREDIABETES
    else:
        return NORMAL

def bmi_lf(row):
    if pd.isna(row['bmi']):
        return -1
    if row['bmi'] >= 35:
        return SEVERE_DIABETES
    elif row['bmi'] >= 30:
        return DIABETES
    elif row['bmi'] >= 25:
        return PREDIABETES
    else:
        return NORMAL

# Create labels using weighted voting
def create_labels(data):
    labels = []
    weights = {
        'hba1c': 0.5,
        'glucose': 0.3,
        'bmi': 0.2
    }
    for _, row in data.iterrows():
        lf_scores = [
            hba1c_lf(row),
            glucose_lf(row),
            bmi_lf(row)
        ]
        valid_scores = [s for s in lf_scores if s != -1]
        if not valid_scores:
            labels.append(np.nan)
            continue
        weighted_score = (
            (lf_scores[0] * weights['hba1c'] if lf_scores[0] != -1 else 0) +
            (lf_scores[1] * weights['glucose'] if lf_scores[1] != -1 else 0) +
            (lf_scores[2] * weights['bmi'] if lf_scores[2] != -1 else 0)
        ) / sum([weights[k] for k, s in zip(weights.keys(), lf_scores) if s != -1])
        labels.append(round(weighted_score))
    return labels

# Add labels to data
data['severity_label'] = create_labels(data)

# Remove samples without labels
data = data.dropna(subset=['severity_label'])

# Process data
X = data[key_features]
y = data['severity_label']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Tune XGBoost with GridSearchCV
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=4, random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Absolute Error: {mae}")
print("Confusion Matrix:")
print(conf_matrix)

# Save model
best_model.save_model('diabetes_prediction_dataset_severity_model.json')

# Find columns with NaN
cols_with_nan = data.columns[data.isna().any()].tolist()

# Print list of columns with NaN
print("Columns with NaN:", cols_with_nan)

# Handle only numeric columns
for col in cols_with_nan:
    if data[col].dtype in ['float64', 'int64']:
        data[col].fillna(data[col].mean(), inplace=True)
print(data.info())
print(data.head())
output_path = '/content/drive/MyDrive/diabetes_prediction_dataset_filled.csv'
data.to_csv(output_path, index=False)
