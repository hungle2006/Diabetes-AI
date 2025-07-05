import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, confusion_matrix
import uuid

# Load data
data = pd.read_csv('/content/drive/MyDrive/kaggle_data/Data_other/diabetes.csv')

# Key features
key_features = [
    'chol', 'stab.glu', 'hdl', 'ratio', 'glyhb', 'age', 'height', 'weight',
    'bp.1s', 'bp.1d', 'bp.2s', 'bp.2d', 'waist', 'hip', 'time.ppn'
]

# Feature engineering
# Calculate BMI: weight (lbs) / (height (in))^2 * 703
data['bmi'] = (data['weight'] / (data['height'] ** 2)) * 703
key_features.append('bmi')

# Convert gender to numeric
data['gender'] = data['gender'].map({'male': 1, 'female': 0})
key_features.append('gender')

# Convert location to numeric
data['location'] = data['location'].map({'Buckingham': 0, 'Louisa': 1})
key_features.append('location')

# Convert frame to numeric
data['frame'] = data['frame'].map({'small': 0, 'medium': 1, 'large': 2})
key_features.append('frame')

# Define labels
NORMAL = 0
PREDIABETES = 1
DIABETES = 2
SEVERE_DIABETES = 3

# Labeling Functions
def glyhb_lf(row):
    if pd.isna(row['glyhb']):
        return -1
    if row['glyhb'] >= 7.0:
        return SEVERE_DIABETES
    elif row['glyhb'] >= 5.7:
        return DIABETES
    elif row['glyhb'] >= 5.0:
        return PREDIABETES
    else:
        return NORMAL

def stab_glu_lf(row):
    if pd.isna(row['stab.glu']):
        return -1
    if row['stab.glu'] >= 200:
        return SEVERE_DIABETES
    elif row['stab.glu'] >= 140:
        return DIABETES
    elif row['stab.glu'] >= 100:
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

def chol_lf(row):
    if pd.isna(row['chol']):
        return -1
    if row['chol'] >= 240:
        return DIABETES
    elif row['chol'] >= 200:
        return PREDIABETES
    else:
        return NORMAL

def ratio_lf(row):
    if pd.isna(row['ratio']):
        return -1
    if row['ratio'] >= 6.0:
        return DIABETES
    elif row['ratio'] >= 5.0:
        return PREDIABETES
    else:
        return NORMAL

# Create labels using weighted voting
def create_labels(data):
    labels = []
    weights = {
        'glyhb': 0.4,
        'stab_glu': 0.3,
        'bmi': 0.15,
        'chol': 0.1,
        'ratio': 0.05
    }
    for _, row in data.iterrows():
        lf_scores = [
            glyhb_lf(row),
            stab_glu_lf(row),
            bmi_lf(row),
            chol_lf(row),
            ratio_lf(row)
        ]
        valid_scores = [s for s in lf_scores if s != -1]
        if not valid_scores:
            labels.append(np.nan)
            continue
        weighted_score = (
            (lf_scores[0] * weights['glyhb'] if lf_scores[0] != -1 else 0) +
            (lf_scores[1] * weights['stab_glu'] if lf_scores[1] != -1 else 0) +
            (lf_scores[2] * weights['bmi'] if lf_scores[2] != -1 else 0) +
            (lf_scores[3] * weights['chol'] if lf_scores[3] != -1 else 0) +
            (lf_scores[4] * weights['ratio'] if lf_scores[4] != -1 else 0)
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
    'max_depth': [3, 5, 7],
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
best_model.save_model('diabetes_severity_model.json')

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
output_path = '/content/drive/MyDrive/diabetes_filled.csv'
data.to_csv(output_path, index=False)
