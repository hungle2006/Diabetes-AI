import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, confusion_matrix
import uuid
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/kaggle_data/Data_other/TrainingWiDS2021.csv'

# Load data
data = pd.read_csv(file_path)
print(data.head())

# Key features
key_features = [
    'age', 'bmi', 'pre_icu_los_days', 'elective_surgery', 'apache_post_operative',
    'map_apache', 'heart_rate_apache', 'resprate_apache', 'temp_apache',
    'd1_spo2_max', 'd1_spo2_min', 'd1_heartrate_max', 'd1_heartrate_min',
    'd1_resprate_max', 'd1_resprate_min', 'd1_mbp_max', 'd1_mbp_min',
    'h1_spo2_max', 'h1_spo2_min', 'h1_mbp_max', 'h1_mbp_min',
    'bun_apache', 'creatinine_apache', 'albumin_apache',
    'd1_bun_max', 'd1_bun_min', 'd1_creatinine_max', 'd1_creatinine_min',
    'd1_albumin_max', 'd1_albumin_min', 'd1_lactate_max', 'd1_lactate_min',
    'd1_hco3_max', 'd1_hco3_min',
    'ph_apache', 'pao2_apache', 'fio2_apache', 'paco2_apache',
    'd1_arterial_ph_max', 'd1_arterial_ph_min', 'd1_arterial_po2_max', 'd1_arterial_po2_min',
    'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min', 'd1_arterial_pco2_max', 'd1_arterial_pco2_min',
    'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache', 'gcs_unable_apache',
    'intubated_apache', 'ventilated_apache', 'urineoutput_apache',
    'aids', 'cirrhosis', 'hepatic_failure', 'immunosuppression', 'leukemia',
    'lymphoma', 'solid_tumor_with_metastasis', 'diabetes_mellitus',
    'apache_2_diagnosis', 'apache_3j_diagnosis', 'arf_apache'
]

# Feature engineering
data['gcs_total'] = data['gcs_eyes_apache'] + data['gcs_motor_apache'] + data['gcs_verbal_apache']
data['pao2_fio2_ratio'] = data['pao2_apache'] / data['fio2_apache'].where(data['fio2_apache'] > 0)
key_features += ['gcs_total', 'pao2_fio2_ratio']

# Define labels
STABLE = 0
MILD = 1
MODERATE = 2
SEVERE = 3
CRITICAL = 4

# Labeling Functions
def gcs_lf(row):
    if pd.isna(row['gcs_total']):
        return -1
    if row['gcs_total'] < 8 or row['gcs_unable_apache'] == 1:
        return CRITICAL
    elif row['gcs_total'] <= 10:
        return SEVERE
    elif row['gcs_total'] <= 13:
        return MODERATE
    else:
        return STABLE

def map_lf(row):
    if pd.isna(row['map_apache']):
        return -1
    if row['map_apache'] < 65:
        return SEVERE
    elif row['map_apache'] <= 70:
        return MODERATE
    elif row['map_apache'] > 100:
        return MILD
    else:
        return STABLE

def ventilation_lf(row):
    if pd.isna(row['ventilated_apache']) or pd.isna(row['intubated_apache']):
        return -1
    if row['intubated_apache'] == 1:
        return CRITICAL
    elif row['ventilated_apache'] == 1:
        return SEVERE
    else:
        return STABLE

def kidney_lf(row):
    if pd.isna(row['bun_apache']) or pd.isna(row['creatinine_apache']):
        return -1
    if row['bun_apache'] > 50 or row['creatinine_apache'] > 3 or row['arf_apache'] == 1:
        return SEVERE
    elif row['bun_apache'] > 20 or row['creatinine_apache'] > 1.5:
        return MODERATE
    else:
        return STABLE

def abg_lf(row):
    if pd.isna(row['ph_apache']) or pd.isna(row['pao2_fio2_ratio']):
        return -1
    if row['ph_apache'] < 7.2 or row['pao2_fio2_ratio'] < 200:
        return CRITICAL
    elif row['pao2_fio2_ratio'] < 300:
        return SEVERE
    else:
        return STABLE

def lactate_lf(row):
    if pd.isna(row['d1_lactate_max']):
        return -1
    if row['d1_lactate_max'] > 4:
        return CRITICAL
    elif row['d1_lactate_max'] > 2:
        return SEVERE
    else:
        return STABLE

# Create labels using weighted voting
def create_labels(data):
    labels = []
    weights = {
        'gcs': 0.35,
        'map': 0.2,
        'ventilation': 0.2,
        'kidney': 0.1,
        'abg': 0.1,
        'lactate': 0.05
    }
    for _, row in data.iterrows():
        lf_scores = [
            gcs_lf(row),
            map_lf(row),
            ventilation_lf(row),
            kidney_lf(row),
            abg_lf(row),
            lactate_lf(row)
        ]
        valid_scores = [s for s in lf_scores if s != -1]
        if not valid_scores:
            labels.append(np.nan)
            continue
        weighted_score = (
            (lf_scores[0] * weights['gcs'] if lf_scores[0] != -1 else 0) +
            (lf_scores[1] * weights['map'] if lf_scores[1] != -1 else 0) +
            (lf_scores[2] * weights['ventilation'] if lf_scores[2] != -1 else 0) +
            (lf_scores[3] * weights['kidney'] if lf_scores[3] != -1 else 0) +
            (lf_scores[4] * weights['abg'] if lf_scores[4] != -1 else 0) +
            (lf_scores[5] * weights['lactate'] if lf_scores[5] != -1 else 0)
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
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=5, random_state=42)
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
best_model.save_model('severity_model_weighted_voting.json')

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
output_path = '/content/drive/MyDrive/TrainingWiDS2021_filled.csv'
data.to_csv(output_path, index=False)
