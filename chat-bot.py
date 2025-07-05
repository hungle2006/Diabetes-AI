
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Layer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging
import cv2
import discord
from discord.ext import commands, tasks
import asyncio
import nest_asyncio
from google.colab import drive
import time
import datetime
import requests
from io import BytesIO
from urllib.parse import urlparse
import gc
import shutil
from flask import Flask, request, render_template, jsonify
from threading import Thread
import json.decoder
from pyngrok import ngrok



public_url = ngrok.connect(5000, bind_tls=True)  # bind_tls=True để đảm bảo HTTPS
global ngrok_url
ngrok_url = public_url
print(f"Ngrok URL: {public_url}")

# Apply nest_asyncio
nest_asyncio.apply()

# Connect with Google Drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    raise Exception(f"Không thể mount Google Drive: {e}")

# set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Token Discord
DISCORD_TOKEN = "your Discord API"  # Thay bằng token Discord thực tế
NUM_CLASSES = 5  # Number of predicted classification

# Saving file
DATA_DIR = "/content/drive/MyDrive/"
LOCAL_DATA_DIR = "/content/drive/MyDrive"
GRAD_CAM_DIR = os.path.join(DATA_DIR, "gradcam_data")
FEATURE_SAVE_DIR = os.path.join(DATA_DIR, "working")
META_SAVE_DIR = os.path.join(FEATURE_SAVE_DIR, "meta_features")
OUTPUT_DIR = os.path.join(FEATURE_SAVE_DIR, "gradcam_output")
GRAD_CAM_SAVE_DIR = os.path.join(FEATURE_SAVE_DIR, "gradcam_meta")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")
DOCTORS_CSV = os.path.join(DATA_DIR, "doctors.csv")
LOCAL_PATIENTS_CSV = os.path.join(LOCAL_DATA_DIR, "patients.csv")
LOCAL_DOCTORS_CSV = os.path.join(LOCAL_DATA_DIR, "doctors.csv")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")

LOCAL_PATIENTS_CSV = '/content/local_patients.csv'
LOCAL_DOCTORS_CSV = '/content/local_doctors.csv'
PATIENTS_CSV = '/content/drive/MyDrive/patients.csv'
DOCTORS_CSV = '/content/drive/MyDrive/doctors.csv'
DATA_DIR = '/content/drive/MyDrive/'

# Function to sync files from Google Drive
def sync_from_drive():
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(LOCAL_PATIENTS_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(LOCAL_DOCTORS_CSV), exist_ok=True)

        # Check for same file paths
        if PATIENTS_CSV == LOCAL_PATIENTS_CSV:
            raise ValueError(f"Source and destination paths for patients are identical: {PATIENTS_CSV}")
        if DOCTORS_CSV == LOCAL_DOCTORS_CSV:
            raise ValueError(f"Source and destination paths for doctors are identical: {DOCTORS_CSV}")

        # Sync patients.csv
        if os.path.exists(PATIENTS_CSV):
            # Validate file before copying
            try:
                df = pd.read_csv(PATIENTS_CSV)
                required_columns = [
                    "patient_id", "name", "gender", "age", "address", "phone", "clinical_data",
                    "condition_label", "id_doctor", "prescription", "medication_schedule", "followup_date"
                ]
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"Missing columns in {PATIENTS_CSV}. Adding missing columns.")
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = pd.Series(dtype=str if col == "followup_date" else object)
                df.to_csv(LOCAL_PATIENTS_CSV, index=False)
                logger.info(f"Copied {PATIENTS_CSV} to {LOCAL_PATIENTS_CSV}")
            except pd.errors.EmptyDataError:
                logger.warning(f"{PATIENTS_CSV} is empty. Creating empty DataFrame.")
                pd.DataFrame(columns=required_columns).to_csv(LOCAL_PATIENTS_CSV, index=False)
            except pd.errors.ParserError:
                logger.error(f"{PATIENTS_CSV} is corrupted or invalid CSV format.")
                raise
        else:
            logger.info(f"{PATIENTS_CSV} not found. Creating empty {LOCAL_PATIENTS_CSV}.")
            pd.DataFrame(columns=[
                "patient_id", "name", "gender", "age", "address", "phone", "clinical_data",
                "condition_label", "id_doctor", "prescription", "medication_schedule", "followup_date"
            ]).to_csv(LOCAL_PATIENTS_CSV, index=False)

        # Sync doctors.csv
        if os.path.exists(DOCTORS_CSV):
            try:
                df = pd.read_csv(DOCTORS_CSV)
                required_columns = ["doctor_id", "name", "specialty", "contact", "hospital"]
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"Missing columns in {DOCTORS_CSV}. Adding missing columns.")
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = pd.Series(dtype=object)
                df.to_csv(LOCAL_DOCTORS_CSV, index=False)
                logger.info(f"Copied {DOCTORS_CSV} to {LOCAL_DOCTORS_CSV}")
            except pd.errors.EmptyDataError:
                logger.warning(f"{DOCTORS_CSV} is empty. Creating empty DataFrame.")
                pd.DataFrame(columns=required_columns).to_csv(LOCAL_DOCTORS_CSV, index=False)
            except pd.errors.ParserError:
                logger.error(f"{DOCTORS_CSV} is corrupted or invalid CSV format.")
                raise
        else:
            logger.info(f"{DOCTORS_CSV} not found. Creating empty {LOCAL_DOCTORS_CSV}.")
            pd.DataFrame(columns=["doctor_id", "name", "specialty", "contact", "hospital"]).to_csv(LOCAL_DOCTORS_CSV, index=False)

        logger.info("Đã đồng bộ dữ liệu từ Google Drive")
    except Exception as e:
        logger.error(f"Lỗi đồng bộ từ Drive: {str(e)}")
        raise

# Push file up Google Drive
def sync_to_drive():
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(PATIENTS_CSV), exist_ok=True)
        os.makedirs(os.path.dirname(DOCTORS_CSV), exist_ok=True)

        # Check for same file paths
        if LOCAL_PATIENTS_CSV == PATIENTS_CSV:
            raise ValueError(f"Source and destination paths for patients are identical: {LOCAL_PATIENTS_CSV}")
        if LOCAL_DOCTORS_CSV == DOCTORS_CSV:
            raise ValueError(f"Source and destination paths for doctors are identical: {LOCAL_DOCTORS_CSV}")

        # Sync patients.csv
        if os.path.exists(LOCAL_PATIENTS_CSV):
            try:
                df = pd.read_csv(LOCAL_PATIENTS_CSV)
                required_columns = [
                    "patient_id", "name", "gender", "age", "address", "phone", "clinical_data",
                    "condition_label", "id_doctor", "prescription", "medication_schedule", "followup_date"
                ]
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"Missing columns in {LOCAL_PATIENTS_CSV}. Adding missing columns.")
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = pd.Series(dtype=str if col == "followup_date" else object)
                df.to_csv(PATIENTS_CSV, index=False)
                logger.info(f"Copied {LOCAL_PATIENTS_CSV} to {PATIENTS_CSV}")
            except pd.errors.EmptyDataError:
                logger.warning(f"{LOCAL_PATIENTS_CSV} is empty. Creating empty DataFrame.")
                pd.DataFrame(columns=required_columns).to_csv(PATIENTS_CSV, index=False)
            except pd.errors.ParserError:
                logger.error(f"{LOCAL_PATIENTS_CSV} is corrupted or invalid CSV format.")
                raise
        else:
            logger.error(f"{LOCAL_PATIENTS_CSV} not found. Cannot sync to Drive.")
            raise FileNotFoundError(f"{LOCAL_PATIENTS_CSV} not found")

        # Sync doctors.csv
        if os.path.exists(LOCAL_DOCTORS_CSV):
            try:
                df = pd.read_csv(LOCAL_DOCTORS_CSV)
                required_columns = ["doctor_id", "name", "specialty", "contact", "hospital"]
                if not all(col in df.columns for col in required_columns):
                    logger.warning(f"Missing columns in {LOCAL_DOCTORS_CSV}. Adding missing columns.")
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = pd.Series(dtype=object)
                df.to_csv(DOCTORS_CSV, index=False)
                logger.info(f"Copied {LOCAL_DOCTORS_CSV} to {DOCTORS_CSV}")
            except pd.errors.EmptyDataError:
                logger.warning(f"{LOCAL_DOCTORS_CSV} is empty. Creating empty DataFrame.")
                pd.DataFrame(columns=required_columns).to_csv(DOCTORS_CSV, index=False)
            except pd.errors.ParserError:
                logger.error(f"{LOCAL_DOCTORS_CSV} is corrupted or invalid CSV format.")
                raise
        else:
            logger.error(f"{LOCAL_DOCTORS_CSV} not found. Cannot sync to Drive.")
            raise FileNotFoundError(f"{LOCAL_DOCTORS_CSV} not found")

        logger.info("Đã đẩy dữ liệu lên Google Drive")
    except Exception as e:
        logger.error(f"Lỗi đẩy dữ liệu lên Drive: {str(e)}")
        raise

# Synchronous function on startup
sync_from_drive()

column_descriptions = {
  "age": "Patient's age (years). Input by user or predicted if missing.",
  "bmi": "Body Mass Index (kg/m²). Input by user or predicted if missing.",
  "gender": "Gender (Male/Female). Input by user or predicted if missing.",
  "glucose_apache": "APACHE glucose score (mg/dL), assesses acute blood glucose levels. Input or predicted.",
  "d1_glucose_max": "Maximum glucose level on day 1 (mg/dL). Input or predicted.",
  "d1_glucose_min": "Minimum glucose level on day 1 (mg/dL). Usually predicted if not entered.",
  "creatinine_apache": "APACHE creatinine score (mg/dL), assesses kidney function. Input or predicted.",
  "d1_creatinine_max": "Maximum creatinine level on day 1 (mg/dL). Usually predicted.",
  "d1_creatinine_min": "Minimum creatinine level on day 1 (mg/dL). Usually predicted.",
  "bun_apache": "APACHE BUN score (mg/dL), assesses kidney function. Usually predicted.",
  "d1_bun_max": "Maximum BUN level on day 1 (mg/dL). Usually predicted.",
  "d1_bun_min": "Minimum BUN level on day 1 (mg/dL). Usually predicted.",
  "sodium_apache": "APACHE sodium score (mEq/L), assesses electrolyte balance. Input or predicted.",
  "d1_sodium_max": "Maximum sodium level on day 1 (mEq/L). Usually predicted.",
  "d1_sodium_min": "Minimum sodium level on day 1 (mEq/L). Usually predicted.",
  "d1_potassium_max": "Maximum potassium level on day 1 (mEq/L). Usually predicted.",
  "d1_potassium_min": "Minimum potassium level on day 1 (mEq/L). Usually predicted.",
  "albumin_apache": "APACHE albumin score (g/dL), assesses nutrition and liver function. Input or predicted.",
  "d1_albumin_max": "Maximum albumin level on day 1 (g/dL). Usually predicted.",
  "d1_albumin_min": "Minimum albumin level on day 1 (g/dL). Usually predicted.",
  "d1_heartrate_max": "Maximum heart rate on day 1 (bpm). Input or predicted.",
  "d1_mbp_max": "Maximum mean blood pressure on day 1 (mmHg). Input or predicted.",
  "severity_label": "Diabetes status 0: Very Mild, 1: Mild, 2: Moderate, 3: Severe, 4: Very Severe. Predicted by the main model."
}


# Clinical Column Mapping
column_mapping = {
    'age': 'age',
    'blood_pressure': 'd1_mbp_max',
    'glucose_level': 'd1_glucose_max',
    'heart_rate': 'd1_heartrate_max',
    'bmi': 'bmi',
    'glucose_apache': 'glucose_apache',
    'creatinine_apache': 'creatinine_apache',
    'sodium_apache': 'sodium_apache',
    'albumin_apache': 'albumin_apache',
    'gender': 'gender',
    'blood_sugar': 'd1_glucose_min'  # Giữ lại, nhưng cần thêm trường nhập liệu
}

# Edit TensorFlow
class MemoryAugmentedLayer(tf.keras.layers.Layer):
    def __init__(self, memory_size=20, memory_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim

    def build(self, input_shape):
        self.memory = self.add_weight(
            name='memory', shape=(self.memory_size, self.memory_dim),
            initializer='random_normal', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        attention_weights = tf.nn.softmax(tf.matmul(inputs, self.memory, transpose_b=True))
        memory_output = tf.matmul(attention_weights, self.memory)
        return tf.concat([inputs, memory_output], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"memory_size": self.memory_size, "memory_dim": self.memory_dim})
        return config

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, inputs, training=None):
        if training:
            return tf.keras.backend.stop_gradient(inputs) * (-self.lambda_)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'lambda_': self.lambda_})
        return config

# PyTorch
class AdvancedMLP(nn.Module):
    def __init__(self, input_size=20, hidden_sizes=[243, 128], num_classes=5, dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size), nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(), nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Hàm xử lý ảnh
def crop_image_from_gray_to_color(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    cropped_img = img[np.ix_(rows, cols)]
    return cropped_img

def load_ben_color(img, sigmaX=10, IMG_SIZE=224):
    try:
        image = img
        image = crop_image_from_gray_to_color(image, tol=7)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
        return image
    except Exception as e:
        logger.error(f"error with load_ben_color: {str(e)}")
        raise ValueError(f"Don't processing image: {str(e)}")

def load_image_from_path(image_path):
    try:
        if image_path.startswith(('http://', 'https://', 'drive.google.com')):
            if 'drive.google.com' in image_path:
                file_id = image_path.split('id=')[-1] if 'id=' in image_path else image_path.split('/d/')[1].split('/')[0]
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            else:
                download_url = image_path
            response = requests.get(download_url, stream=True, timeout=10)
            if response.status_code != 200:
                logger.error(f"Don't upload {image_path}. Status: {response.status_code}")
                return None
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img = np.array(img)
        else:
            if not os.path.exists(image_path):
                logger.error(f"Don't find: {image_path}")
                return None
            img = Image.open(image_path).convert('RGB')
            img = np.array(img)
        img = load_ben_color(img, sigmaX=10, IMG_SIZE=224)
        return img
    except Exception as e:
        logger.error(f"Error download image: {str(e)}")
        return None

def extract_image_id_from_path(image_path):
    if image_path.startswith(('http://', 'https://')):
        parsed_url = urlparse(image_path)
        filename = os.path.basename(parsed_url.path)
    else:
        filename = os.path.basename(image_path)
    image_id = os.path.splitext(filename)[0]
    return image_id if image_id else "unknown"

# Load model
def load_meta_model(config_path, weights_path):
    try:
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            logger.error(f"Don't find the file config or weights: {config_path}, {weights_path}")
            return None, None, None
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        custom_objects = {
            'MemoryAugmentedLayer': MemoryAugmentedLayer,
            'GradientReversalLayer': GradientReversalLayer
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            meta_model = model_from_json(json.dumps(model_config), custom_objects=custom_objects)
            meta_model.load_weights(weights_path)
            classification_output = meta_model.get_layer('classification').output
            feature_output = meta_model.layers[-3].output
            inputs = meta_model.input
            meta_classification_model = tf.keras.Model(inputs=inputs, outputs=classification_output)
            feature_model = tf.keras.Model(inputs=inputs, outputs=feature_output)
        logger.info(f"Loaded model from {weights_path}")
        return meta_model, meta_classification_model, feature_model
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        try:
            logger.info("Create fallback model...")
            fallback_model = create_fallback_model(NUM_CLASSES)
            return fallback_model, fallback_model, fallback_model
        except Exception as fallback_error:
            logger.error(f"Don't create fallback model: {str(fallback_error)}")
            return None, None, None

def create_fallback_model(num_classes=5):
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification')(x)
    model = tf.keras.Model(inputs, outputs)
    logger.info("Successfull create fallback model")
    return model

def load_model_from_config(config_path, weights_path, base_model_class):
    try:
        if os.path.exists(config_path) and os.path.exists(weights_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            custom_objects = {
                'MemoryAugmentedLayer': MemoryAugmentedLayer,
                'GradientReversalLayer': GradientReversalLayer
            }
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = tf.keras.models.model_from_json(json.dumps(model_config), custom_objects=custom_objects)
                model.load_weights(weights_path)
            return model
        logger.warning(f"Missing config or weights, using default model.")
        return base_model_class(weights='imagenet', include_top=False, pooling='avg')
    except Exception as e:
        logger.warning(f"Error loading model from config: {str(e)}. Using default model.")
        return base_model_class(weights='imagenet', include_top=False, pooling='avg')

# Calculate Grad-CAM
def compute_gradcam_4d(model, img_array, class_idx, layer_name, img_size=(224, 224)):
    try:
        img_array_resized = tf.image.resize(img_array, img_size, method='bilinear')
        img_array_resized = tf.ensure_shape(img_array_resized, [1, img_size[0], img_size[1], 3])
        if hasattr(model, 'preprocess_input'):
            img_array_resized = model.preprocess_input(img_array_resized)
        try:
            conv_layer = model.get_layer(layer_name)
        except ValueError:
            conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
            if conv_layers:
                conv_layer = conv_layers[-1]
                layer_name = conv_layer.name
            else:
                logger.error("Convolutional layer not found.")
                return None
        grad_model = tf.keras.Model(inputs=[model.input], outputs=[conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array_resized)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            logger.error(f"Gradient None cho layer {layer_name}")
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()
        if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
            logger.warning(f"Heatmap contain NaN/inf")
            return None
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap
    except Exception as e:
        logger.error(f"Error compute_gradcam_4d: {str(e)}")
        return None

# Function temperature scaling and Laplace smoothing
def apply_temperature_scaling(logits, temperature=2.0):
    return tf.nn.softmax(logits / temperature)

def laplace_smoothing(probs, epsilon=1e-5):
    return (probs + epsilon) / tf.reduce_sum(probs + epsilon, axis=1, keepdims=True)

# Predict the image
def predict_and_gradcam_from_path(image_path, meta_model, meta_classification_model, feature_model, model_configs):
    image_id = extract_image_id_from_path(image_path)
    img = load_image_from_path(image_path)
    if img is None:
        logger.warning(f"Unable to load or process image from {image_path}")
        return []
    img_array = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    if meta_model is None or meta_classification_model is None:
        logger.warning("Meta model not available, use prediction from EfficientNetB0")
        simple_model = tf.keras.applications.EfficientNetB0(weights='imagenet', classes=NUM_CLASSES)
        preds = simple_model.predict(img_array, batch_size=1)
        pred_class = np.argmax(preds, axis=1)[0]
        probs = np.max(preds, axis=1)[0]
    else:
        features_dict = {}
        for model_name, config in model_configs.items():
            try:
                preprocess = config['preprocess']
                img_size = config['img_size']
                img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                img_preprocessed = preprocess(np.expand_dims(img_resized, axis=0).astype(np.float32))
                base_model = load_model_from_config(config['config_path'], config['weights_path'], config['base_model'])
                try:
                    feature_layer_2d = base_model.layers[-2].output
                    feature_extract = tf.keras.Model(inputs=base_model.input, outputs=feature_layer_2d)
                except:
                    feature_extract = base_model
                features_2d = feature_extract.predict(img_preprocessed, batch_size=1)
                features_dict[model_name] = features_2d
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception as e:
                logger.warning(f"Error extracting features from {model_name}: {str(e)}")
                continue
        if not features_dict:
            logger.error("Unable to extract features from any model")
            return []
        combined_features = np.concatenate([features_dict[model_name] for model_name in model_configs], axis=1)
        try:
            scaler = StandardScaler()
            combined_features_scaled = scaler.fit_transform(combined_features)
            pca = PCA(n_components=min(50, combined_features_scaled.shape[1]), random_state=42)
            meta_features = pca.fit_transform(combined_features_scaled)
            preds = meta_classification_model.predict(meta_features, batch_size=1)
            preds = apply_temperature_scaling(preds, temperature=2.0)
            preds = laplace_smoothing(preds, epsilon=1e-6)
            pred_class = np.argmax(preds, axis=1)[0]
            probs = np.max(preds, axis=1)[0]
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            pred_class = np.random.randint(0, NUM_CLASSES)
            probs = 0.2
    results = []
    for model_name, config in model_configs.items():
        try:
            class_dir = os.path.join(GRAD_CAM_DIR, f"class_label_{pred_class}")
            os.makedirs(class_dir, exist_ok=True)
            model = load_model_from_config(config['config_path'], config['weights_path'], config['base_model'])
            model.preprocess_input = config['preprocess']
            heatmap = compute_gradcam_4d(
                model=model,
                img_array=img_array,
                class_idx=pred_class,
                layer_name=config['feature_layer_name'],
                img_size=(config['img_size'], config['img_size'])
            )
            if heatmap is not None:
                original_img = (img * 255).astype(np.uint8)
                superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
                timestamp = int(time.time())
                original_path = os.path.join(class_dir, f"sample_{image_id}_processed_{timestamp}.png")
                heatmap_path = os.path.join(class_dir, f"sample_{image_id}_heatmap_{model_name}_{timestamp}.png")
                gradcam_path = os.path.join(class_dir, f"sample_{image_id}_gradcam_pred_{pred_class}_{model_name}_{timestamp}.png")
                cv2.imwrite(original_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(heatmap_path, heatmap)
                cv2.imwrite(gradcam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
                results.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'pred_class': int(pred_class),
                    'prob': float(probs),
                    'processed_image_path': original_path,
                    'heatmap_path': heatmap_path,
                    'gradcam_path': gradcam_path,
                    'model_name': model_name
                })
                logger.info(f"Saved Grad-CAM for {image_id} ({model_name}) at: {gradcam_path}")
            else:
                logger.warning(f"Unable to generate Grad-CAM for {image_id} ({model_name})")
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception as e:
            logger.error(f"Error creating Grad-CAM for {model_name}: {str(e)}")
            continue
    return results

# Processing clinical data
def load_data(file_path, usecols=None, max_rows=None):
    logger = logging.getLogger(__name__)
    try:
        # Đọc dữ liệu, loại bỏ cột Unnamed: 0 nếu có
        df = pd.read_csv(file_path, usecols=usecols, nrows=max_rows, index_col=None)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        logger.info(f"Loaded data from {file_path} with columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def validate_clinical_data(data):
    errors = []

    if not isinstance(data, dict):
        return ["Data must be a valid JSON object (Python dictionary)"]

    for key, value in data.items():
        if key in ['age', 'blood_pressure', 'glucose_level', 'heart_rate', 'bmi', 'cholesterol', 'triglycerides']:
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"The value of '{key}' must be a non-negative number")
        elif key == 'gender' and value not in ['Male', 'Female']:
            errors.append("Gender must be 'Male' or 'Female'")

    return errors


def impute_missing_values(clinical_df, reference_df):
    """Impute missing values using RandomForest."""
    df_impute = clinical_df.copy().rename(columns={k: v for k, v in column_mapping.items() if v})
    numeric_cols = reference_df.select_dtypes(include=['float32', 'int32']).columns
    categorical_cols = reference_df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        if col not in df_impute.columns or not df_impute[col].isna().any():
            continue
        known_cols = [c for c in numeric_cols if c != col and c in df_impute.columns]
        if not known_cols:
            logger.warning(f"Skipping {col}: no valid features available")
            continue
        logger.info(f"Imputing missing values for numeric column: {col}")
        X_train = reference_df[known_cols].dropna()
        y_train = reference_df[col].loc[X_train.index]
        X_test = df_impute[known_cols].loc[df_impute[col].isna()]
        if len(X_test) == 0 or len(X_train) == 0:
            continue
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', RobustScaler())
                ]), known_cols)
            ]
        )
        try:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(X_train_processed, y_train)
            df_impute.loc[df_impute[col].isna(), col] = rf.predict(X_test_processed)
        except Exception as e:
            logger.error(f"Error imputing values for {col}: {e}")

    for col in categorical_cols:
        if col not in df_impute.columns or not df_impute[col].isna().any():
            continue
        known_cols = [c for c in numeric_cols if c in df_impute.columns]
        if not known_cols:
            logger.warning(f"Skipping {col}: no valid features available")
            continue
        logger.info(f"Imputing missing values for categorical column: {col}")
        X_train = reference_df[known_cols].dropna()
        y_train = reference_df[col].loc[X_train.index]
        X_test = df_impute[known_cols].loc[df_impute[col].isna()]
        if len(X_test) == 0 or len(X_train) == 0:
            continue
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', RobustScaler())
                ]), known_cols)
            ]
        )
        try:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(X_train_processed, pd.Categorical(y_train).codes)
            predicted_codes = rf.predict(X_test_processed).astype(int)
            valid_categories = pd.Categorical(reference_df[col].dropna().unique())
            df_impute.loc[df_impute[col].isna(), col] = [
                valid_categories[i % len(valid_categories)] for i in predicted_codes
            ]
        except Exception as e:
            logger.error(f"Error imputing values for {col}: {e}")

    return df_impute



def predict_new_columns(training_df, predict_df):
    logger = logging.getLogger(__name__)
    try:
        # Identify new columns to predict, only those present in training_df
        new_columns = [col for col in training_df.columns if col not in predict_df.columns]
        if not new_columns:
            logger.info("No new columns to predict.")
            return pd.DataFrame(index=predict_df.index)

        # Create DataFrame to store only the predicted new columns
        result_df = pd.DataFrame(index=predict_df.index)

        # Identify numeric and categorical columns in training_df
        numeric_cols = training_df.select_dtypes(include=['int32', 'float32']).columns
        categorical_cols = training_df.select_dtypes(include=['object']).columns

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', RobustScaler())
                ]), numeric_cols.intersection(training_df.columns)),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_cols.intersection(training_df.columns))
            ],
            remainder='passthrough'
        )

        # Iterate through each new column to predict
        for column in new_columns:
            if column not in training_df.columns:
                logger.warning(f"Skipping column {column}: not found in training_df")
                result_df[column] = np.nan
                continue
            logger.info(f"Predicting column: {column}")

            # Prepare training and test data
            X_train = training_df.drop(columns=[column])
            y_train = training_df[column].dropna()
            X_test = predict_df.copy()

            # Ensure X_test has same columns as X_train
            for col in X_train.columns:
                if col not in X_test.columns:
                    X_test[col] = np.nan
            X_test = X_test[X_train.columns]

            # Check for valid data
            if X_train.empty or X_test.empty or y_train.empty:
                logger.warning(f"Skipping column {column} due to empty data.")
                result_df[column] = np.nan
                continue

            try:
                # Preprocess data
                X_train_processed = preprocessor.fit_transform(X_train).astype('float32')
                X_test_processed = preprocessor.transform(X_test).astype('float32')

                # Choose model based on column type
                if column in numeric_cols:
                    model = RandomForestRegressor(
                        n_estimators=10, max_depth=10, random_state=42, n_jobs=1
                    )
                    model.fit(X_train_processed, y_train)
                    result_df[column] = model.predict(X_test_processed)
                else:
                    model = RandomForestClassifier(
                        n_estimators=10, max_depth=10, random_state=42, n_jobs=1
                    )
                    y_train_codes = pd.Categorical(y_train).codes
                    if len(np.unique(y_train_codes)) < 2:
                        logger.warning(f"Skipping column {column} due to only one unique class.")
                        result_df[column] = np.nan
                        continue
                    model.fit(X_train_processed, y_train_codes)
                    predicted_codes = model.predict(X_test_processed).astype(int)
                    valid_categories = pd.Categorical(training_df[column].dropna().unique())
                    result_df[column] = [
                        valid_categories[i % len(valid_categories)] for i in predicted_codes
                    ]
            except Exception as e:
                logger.error(f"Error predicting column {column}: {e}")
                result_df[column] = np.nan

        return result_df

    except Exception as e:
        logger.error(f"Error in predict_new_columns: {e}")
        return pd.DataFrame(index=predict_df.index)


def preprocess_clinical_data(df):
    if df.empty:
        return torch.FloatTensor(np.zeros((1, 0))), None
    numeric_cols = df.select_dtypes(include=['float32', 'int32']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', RobustScaler())]), numeric_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_cols)
    ])
    X_processed = preprocessor.fit_transform(df).astype('float32')
    return torch.FloatTensor(X_processed), preprocessor

def prepare_input_data(clinical_df=None, image_features=None, preprocessor=None, expected_input_size=20):
    """Prepare input data for the model, supporting image-only input."""
    if clinical_df is not None and not clinical_df.empty and preprocessor is not None:
        X_processed = preprocessor.transform(clinical_df).astype('float32')
        X_processed = torch.FloatTensor(X_processed)
        logger.info(f"Processed clinical features: {X_processed.shape[1]}")
    else:
        X_processed = torch.FloatTensor(np.zeros((1, 0)))  # No clinical data
        logger.info("No clinical data, using image features only")

    if image_features is not None and image_features.shape[0] == 1:
        image_features_tensor = torch.FloatTensor(image_features.astype('float32'))
        X_processed = torch.cat((X_processed, image_features_tensor), dim=1) if X_processed.shape[1] > 0 else image_features_tensor
        logger.info(f"Image feature dimensions: {image_features.shape[1]}")

    if X_processed.shape[1] != expected_input_size:
        if X_processed.shape[1] > expected_input_size:
            X_processed = X_processed[:, :expected_input_size]
            logger.info(f"Trimmed to {expected_input_size} dimensions")
        else:
            padding = torch.zeros(X_processed.shape[0], expected_input_size - X_processed.shape[1])
            X_processed = torch.cat((X_processed, padding), dim=1)
            logger.info(f"Added padding for {expected_input_size - X_processed.shape[1]} dimensions")

    logger.info(f"Final input size: {X_processed.shape}")
    return X_processed


import torch.nn as nn

class AdvancedMLP_clinical(nn.Module):
    """MLP model for predicting severity levels."""

    def __init__(self, input_size=20, hidden_dims=[125, 102], num_classes=5, dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_dims:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_clinical_model(model_path='/content/drive/MyDrive/main_clinical_model.pth'):
    """Load the clinical model from a .pth file."""
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model = AdvancedMLP_clinical(
            input_size=20,
            hidden_dims=[125, 102],
            num_classes=checkpoint['num_classes'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
        logger.info(f"Loaded clinical model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading clinical model: {e}")
        raise


def predict_clinical(clinical_data, training_file_path, clinical_model_path='/content/drive/MyDrive/main_clinical_model.pth', preprocessor=None, expected_input_size=20):
    """
    Predicts the label from clinical data and returns all usecols with source and description.

    Args:
        clinical_data (dict): Input data as a dictionary.
        training_file_path (str): Path to the training data.
        clinical_model_path (str): Path to the trained clinical model.
        preprocessor: Preprocessing pipeline.
        expected_input_size (int): Expected input size of the model.

    Returns:
        tuple: (predicted label, softmax probabilities, DataFrame with usecols, source, and description)
    """
    logger = logging.getLogger(__name__)

    if not clinical_data:
        raise ValueError("Clinical data is required")

    try:
        # Convert dict to DataFrame
        clinical_df = pd.DataFrame([clinical_data])

        # Apply column mapping
        clinical_df = clinical_df.rename(columns={k: v for k, v in column_mapping.items() if k in clinical_df.columns})

        # Load training data
        training_df = load_data(training_file_path, max_rows=5000)

        # Impute missing values
        imputed_df = impute_missing_values(clinical_df, training_df)

        # Columns to use
        usecols = [
            # Basic info
            'age', 'bmi', 'gender',
            # Glucose - most important for diabetes
            'glucose_apache', 'd1_glucose_max', 'd1_glucose_min',
            # Kidney function (diabetes complication)
            'creatinine_apache', 'd1_creatinine_max', 'd1_creatinine_min',
            'bun_apache', 'd1_bun_max', 'd1_bun_min',
            # Electrolytes - Sodium and Potassium
            'sodium_apache', 'd1_sodium_max', 'd1_sodium_min',
            'd1_potassium_max', 'd1_potassium_min',
            # Blood protein
            'albumin_apache', 'd1_albumin_max', 'd1_albumin_min',
            # Cardiovascular vitals
            'd1_heartrate_max', 'd1_mbp_max',
            # Target variable
            'severity_label'
        ]

        # Identify input columns
        input_columns = [col for col in imputed_df.columns if col in usecols]

        # Determine which columns need prediction (excluding target)
        columns_to_predict = [col for col in usecols if col not in input_columns and col != 'severity_label']

        # Initialize full DataFrame with usecols
        extended_df = pd.DataFrame(index=imputed_df.index, columns=usecols)

        # Fill available columns
        for col in input_columns:
            extended_df[col] = imputed_df[col]

        # Predict missing columns
        if columns_to_predict:
            logger.info(f"Predicting columns: {columns_to_predict}")
            training_df_filtered = training_df[[col for col in training_df.columns if col in usecols or col in imputed_df.columns]]
            predicted_df = predict_new_columns(training_df_filtered, imputed_df)
            for col in columns_to_predict:
                if col in predicted_df.columns:
                    extended_df[col] = predicted_df[col]

        # Preprocess input for main model
        X_input = prepare_input_data(
            clinical_df=extended_df,
            image_features=None,
            preprocessor=preprocessor,
            expected_input_size=expected_input_size
        )

        logger.info(f"Predicting with input shape: {X_input.shape}")

        # Load model
        clinical_model = load_clinical_model(clinical_model_path)

        # Predict severity_label
        with torch.no_grad():
            outputs = clinical_model(X_input)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        # Add prediction to DataFrame
        extended_df['severity_label'] = predicted.numpy()[0]

        # Add data source column
        extended_df['source'] = extended_df.apply(
            lambda row: 'Entered by user' if row.name in input_columns else 'Predicted by model', axis=1)
        for col in usecols:
            if col in input_columns:
                extended_df.loc[extended_df.index, 'source'] = 'Entered by user'
            else:
                extended_df.loc[extended_df.index, 'source'] = 'Predicted by model'

        # Add description column
        extended_df['description'] = extended_df.apply(
            lambda row: column_descriptions.get(row.name, 'No description available'), axis=1)
        for col in usecols:
            extended_df.loc[extended_df.index, 'description'] = column_descriptions.get(col, 'No description available')

        # Rearrange columns
        final_df = extended_df[usecols + ['source', 'description']]

        return predicted.numpy()[0], probs.numpy()[0], final_df

    except Exception as e:
        logger.error(f"Error during clinical prediction: {e}")
        raise


# Function to save a JSON template
def save_json_template(data, template_type, user_id):
    try:
        timestamp = int(time.time())
        template_path = os.path.join(TEMPLATES_DIR, f"{template_type}_{user_id}_{timestamp}.json")
        with open(template_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Template saved at: {template_path}")
        sync_to_drive()
        return template_path
    except Exception as e:
        logger.error(f"Error saving template: {str(e)}")
        return None

# Function to validate JSON with detailed error hints
def validate_json(json_str):
    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {e.msg} at line {e.lineno}, column {e.colno}"
        if "Expecting ','" in e.msg:
            error_msg += "\nHint: Check for missing commas between fields."
        elif "Expecting property name" in e.msg:
            error_msg += "\nHint: Make sure property names are enclosed in double quotes."
        elif "Expecting value" in e.msg:
            error_msg += "\nHint: Check for missing or invalid field values."
        return None, error_msg


# Hàm lưu vào CSV
def save_to_csv(df, csv_path):
    try:
        df.to_csv(csv_path, index=False)
        sync_to_drive()
        logger.info(f"Data saved to {csv_path}")
    except Exception as e:
        logger.error(f"Error saving CSV: {str(e)}")
        raise e


# Cấu hình mô hình
model_configs = {
    "efficientnetb0": {
        "model_type": "efficientnetb0",
        "config_path": os.path.join(FEATURE_SAVE_DIR, "EfficientNetB0_bestqwk", "config.json"),
        "weights_path": os.path.join(FEATURE_SAVE_DIR, "EfficientNetB0_bestqwk", "model.weights.h5"),
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "img_size": 224,
        "base_model": tf.keras.applications.EfficientNetB0,
        "feature_layer_name": "top_conv"
    },
    "xception": {
        "model_type": "xception",
        "config_path": os.path.join(FEATURE_SAVE_DIR, "Xception_bestqwk", "config.json"),
        "weights_path": os.path.join(FEATURE_SAVE_DIR, "Xception_bestqwk", "model.weights.h5"),
        "preprocess": tf.keras.applications.xception.preprocess_input,
        "img_size": 299,
        "base_model": tf.keras.applications.Xception,
        "feature_layer_name": "block14_sepconv2_act"
    },
    "inceptionv3": {
        "model_type": "inceptionv3",
        "config_path": os.path.join(FEATURE_SAVE_DIR, "InceptionV3_bestqwk", "config.json"),
        "weights_path": os.path.join(FEATURE_SAVE_DIR, "InceptionV3_bestqwk", "model.weights.h5"),
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
        "img_size": 299,
        "base_model": tf.keras.applications.InceptionV3,
        "feature_layer_name": "mixed10"
    },
    "resnet50": {
        "model_type": "resnet50",
        "config_path": os.path.join(FEATURE_SAVE_DIR, "ResNet50_bestqwk", "config.json"),
        "weights_path": os.path.join(FEATURE_SAVE_DIR, "ResNet50_bestqwk", "model.weights.h5"),
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
        "img_size": 224,
        "base_model": tf.keras.applications.ResNet50,
        "feature_layer_name": "conv5_block3_out"
    },
    "densenet121": {
        "model_type": "densenet121",
        "config_path": os.path.join(FEATURE_SAVE_DIR, "DenseNet121_bestqwk", "config.json"),
        "weights_path": os.path.join(FEATURE_SAVE_DIR, "DenseNet121_bestqwk", "model.weights.h5"),
        "preprocess": tf.keras.applications.densenet.preprocess_input,
        "img_size": 224,
        "base_model": tf.keras.applications.DenseNet121,
        "feature_layer_name": "conv5_block16_concat"
    }
}

config_filepath = os.path.join(FEATURE_SAVE_DIR, "config.json")
weights_filepath = os.path.join(FEATURE_SAVE_DIR, "meta_model_maml_fomaml_best_weights.weights.h5")
clinical_model_path = os.path.join(DATA_DIR, "main_clinical_model.pth")
training_file_path = os.path.join("/content/drive/MyDrive/TrainingWiDS2021_filled.csv")

# Flask app cho giao diện web
app = Flask(__name__)
api_data = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_patient', methods=['GET', 'POST'])
def web_add_patient():
    if request.method == 'POST':
        try:
            logger.info("Form data received: %s", request.form)

            # Get number of medications
            num_meds = request.form.get('num_meds', '').strip()
            try:
                num_meds = int(num_meds)
                if num_meds <= 0:
                    raise ValueError("Number of medications must be a positive integer")
            except ValueError:
                error_msg = "Number of medications must be a positive integer"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400

            # Get medication list
            prescription = []
            for i in range(num_meds):
                med_name = request.form.get(f'med_name_{i}', '').strip()
                used_tablets = request.form.get(f'used_tablets_{i}', '').strip()
                if not med_name or not used_tablets:
                    error_msg = f"Medication {i+1}: Name and tablet count must not be empty"
                    logger.error(error_msg)
                    return jsonify({"status": "error", "message": error_msg}), 400
                if not used_tablets.endswith("viên"):
                    error_msg = f"Medication {i+1}: Tablet count must end with 'viên' (e.g., 60 viên)"
                    logger.error(error_msg)
                    return jsonify({"status": "error", "message": error_msg}), 400
                try:
                    tablet_count = int(used_tablets.replace("viên", "").strip())
                    if tablet_count <= 0:
                        raise ValueError
                except ValueError:
                    error_msg = f"Medication {i+1}: Tablet count must be a positive integer (e.g., 60 viên)"
                    logger.error(error_msg)
                    return jsonify({"status": "error", "message": error_msg}), 400
                prescription.append({"name": med_name, "used_tablets": used_tablets})

            # Get follow-up date
            followup_date = request.form.get('followup_date', '').strip()
            if followup_date:
                try:
                    datetime.datetime.strptime(followup_date, '%Y-%m-%d %H:%M')
                except ValueError:
                    error_msg = "Follow-up date must be in format YYYY-MM-DD HH:MM (e.g., 2025-07-01 09:00)"
                    logger.error(error_msg)
                    return jsonify({"status": "error", "message": error_msg}), 400
            else:
                condition_label = int(request.form.get('condition_label', -1))
                if 0 <= condition_label <= 4:
                    followup_date = schedule_followup(condition_label)
                else:
                    error_msg = "Invalid condition level for auto-generating follow-up date"
                    logger.error(error_msg)
                    return jsonify({"status": "error", "message": error_msg}), 400

            # Get other fields
            data = {
                "patient_id": request.form.get('patient_id', '').strip(),
                "name": request.form.get('name', '').strip(),
                "gender": request.form.get('gender', '').strip(),
                "age": int(request.form.get('age', -1)) if request.form.get('age') and request.form.get('age').isdigit() else -1,
                "address": request.form.get('address', '').strip(),
                "phone": request.form.get('phone', '').strip(),
                "id_doctor": request.form.get('id_doctor', '').strip(),
                "condition_label": int(request.form.get('condition_label', -1)) if request.form.get('condition_label') and request.form.get('condition_label').isdigit() else -1,
                "prescription": prescription,
                "medication_schedule": {
                    "morning": [t.strip() for t in request.form.get('medication_morning', '').split(',') if t.strip()],
                    "noon": [t.strip() for t in request.form.get('medication_noon', '').split(',') if t.strip()],
                    "evening": [t.strip() for t in request.form.get('medication_evening', '').split(',') if t.strip()]
                },
                "followup_date": followup_date
            }

            # Validate required fields
            required_fields = ["patient_id", "name", "gender", "age", "address", "phone", "id_doctor", "condition_label"]
            missing_fields = [field for field in required_fields if not data[field] or data[field] == -1]
            if missing_fields:
                error_msg = f"Missing or invalid required fields: {', '.join(missing_fields)}"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400

            # Format checks
            if data['gender'] not in ['Male', 'Female']:
                error_msg = "Gender must be 'Male' or 'Female'"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400
            if data['age'] < 0:
                error_msg = "Age must be a non-negative number"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400
            if not (0 <= data['condition_label'] <= 4):
                error_msg = "Condition level must be between 0 and 4"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400

            # Validate medication times
            for time_slot, times in data['medication_schedule'].items():
                for time_str in times:
                    try:
                        datetime.datetime.strptime(time_str, '%H:%M')
                    except ValueError:
                        error_msg = f"Time {time_str} in {time_slot} is not in HH:MM format"
                        logger.error(error_msg)
                        return jsonify({"status": "error", "message": error_msg}), 400

            # Check patient and doctor
            patients_df = pd.read_csv(LOCAL_PATIENTS_CSV)
            if data['patient_id'] in patients_df['patient_id'].values:
                error_msg = f"Patient ID {data['patient_id']} already exists"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400
            doctors_df = pd.read_csv(LOCAL_DOCTORS_CSV)
            if data['id_doctor'] not in doctors_df['doctor_id'].values:
                error_msg = f"Doctor ID {data['id_doctor']} does not exist"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400

            # Save to patients.csv
            new_patient = {
                "patient_id": data['patient_id'],
                "name": data['name'],
                "gender": data['gender'],
                "age": data['age'],
                "address": data['address'],
                "phone": data['phone'],
                "clinical_data": json.dumps({}),
                "condition_label": data['condition_label'],
                "id_doctor": data['id_doctor'],
                "prescription": json.dumps(data['prescription']),
                "medication_schedule": json.dumps(data['medication_schedule']),
                "followup_date": data['followup_date']
            }
            patients_df = pd.concat([patients_df, pd.DataFrame([new_patient])], ignore_index=True)
            save_to_csv(patients_df, LOCAL_PATIENTS_CSV)

            # Save template
            template_path = save_json_template(data, "patient", "web")

            # Add to api_data for bot processing
            api_data.append({"type": "patient", "data": data})
            logger.info(f"Patient added to api_data: {data['patient_id']}")
            return jsonify({
                "status": "success",
                "message": f"Patient {data['patient_id']} added successfully. Follow-up: {data['followup_date']}"
            })
        except Exception as e:
            logger.error(f"Error in web_add_patient: {str(e)}")
            return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500
    return render_template('add_patient.html')


@app.route('/add_clinical', methods=['GET', 'POST'])
def web_add_clinical():
    if request.method == 'POST':
        try:
            data = {
                "age": float(request.form.get('age')) if request.form.get('age') and request.form.get('age').replace('.', '', 1).isdigit() else None,
                "blood_pressure": float(request.form.get('blood_pressure')) if request.form.get('blood_pressure') and request.form.get('blood_pressure').replace('.', '', 1).isdigit() else None,
                "glucose_level": float(request.form.get('glucose_level')) if request.form.get('glucose_level') and request.form.get('glucose_level').replace('.', '', 1).isdigit() else None,
                "heart_rate": float(request.form.get('heart_rate')) if request.form.get('heart_rate') and request.form.get('heart_rate').replace('.', '', 1).isdigit() else None,
                "bmi": float(request.form.get('bmi')) if request.form.get('bmi') and request.form.get('bmi').replace('.', '', 1).isdigit() else None,
                "glucose_apache": float(request.form.get('glucose_apache')) if request.form.get('glucose_apache') and request.form.get('glucose_apache').replace('.', '', 1).isdigit() else None,
                "creatinine_apache": float(request.form.get('creatinine_apache')) if request.form.get('creatinine_apache') and request.form.get('creatinine_apache').replace('.', '', 1).isdigit() else None,
                "sodium_apache": float(request.form.get('sodium_apache')) if request.form.get('sodium_apache') and request.form.get('sodium_apache').replace('.', '', 1).isdigit() else None,
                "albumin_apache": float(request.form.get('albumin_apache')) if request.form.get('albumin_apache') and request.form.get('albumin_apache').replace('.', '', 1).isdigit() else None,
                "blood_sugar": float(request.form.get('blood_sugar')) if request.form.get('blood_sugar') and request.form.get('blood_sugar').replace('.', '', 1).isdigit() else None,
                "gender": request.form.get('gender')
            }

            # Ensure at least one valid field
            valid_data = {k: v for k, v in data.items() if v is not None and v != ''}
            if not valid_data:
                error_msg = "At least one valid clinical field is required."
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400

            # Validate gender
            if data['gender'] and data['gender'] not in ['Male', 'Female']:
                error_msg = "Gender must be 'Male' or 'Female'."
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400

            # Validate ranges
            for field in ['age', 'blood_pressure', 'glucose_level', 'heart_rate', 'bmi',
                          'glucose_apache', 'creatinine_apache', 'sodium_apache',
                          'albumin_apache', 'blood_sugar']:
                if data[field] is not None:
                    if data[field] < 0:
                        error_msg = f"Value of '{field}' must be non-negative."
                        logger.error(error_msg)
                        return jsonify({"status": "error", "message": error_msg}), 400
                    if field == 'blood_pressure' and not (50 <= data[field] <= 250):
                        error_msg = "Blood pressure must be in the range 50–250 mmHg."
                        logger.error(error_msg)
                        return jsonify({"status": "error", "message": error_msg}), 400
                    if field == 'heart_rate' and not (30 <= data[field] <= 200):
                        error_msg = "Heart rate must be in the range 30–200 bpm."
                        logger.error(error_msg)
                        return jsonify({"status": "error", "message": error_msg}), 400
                    if field == 'bmi' and not (10 <= data[field] <= 60):
                        error_msg = "BMI must be in the range 10–60."
                        logger.error(error_msg)
                        return jsonify({"status": "error", "message": error_msg}), 400
                    if field == 'blood_sugar' and not (30 <= data[field] <= 500):
                        error_msg = "Blood sugar must be in the range 30–500 mg/dL."
                        logger.error(error_msg)
                        return jsonify({"status": "error", "message": error_msg}), 400

            # Run prediction
            severity, probs, result_df = predict_clinical(valid_data, training_file_path, clinical_model_path)
            severity_map = {
                0: "Very Mild",
                1: "Mild",
                2: "Moderate",
                3: "Severe",
                4: "Critical"
            }

            # Convert results to HTML table
            result_html = result_df.to_html(index=False, classes='table table-striped', border=1)

            # Render result page
            return render_template(
                'result_clinical.html',
                severity=severity_map.get(severity, 'Unknown'),
                probability=np.max(probs),
                result_table=result_html
            )

        except Exception as e:
            logger.error(f"Error in web_add_clinical: {str(e)}")
            return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

    return render_template('add_clinical.html')


@app.route('/api/data', methods=['GET'])
def get_api_data():
    try:
        logger.info(f"Accessed /api/data, api_data status: {len(api_data)} items")
        if api_data:
            data = api_data.pop(0)
            logger.info(f"Returning data: {data}")
            return jsonify(data)
        return jsonify({"status": "empty"})
    except Exception as e:
        logger.error(f"Error in get_api_data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# HTML templates
TEMPLATES = {
    "index.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Bot</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        }
    </style>
</head>
<body class="font-sans text-gray-900 flex items-center justify-center min-h-screen p-6">
    <div class="bg-white bg-opacity-90 backdrop-blur-lg rounded-3xl shadow-2xl max-w-2xl w-full p-10 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-4xl font-extrabold text-center mb-10 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">Medical Bot Interface</h1>
        <div class="flex flex-col gap-5 max-w-sm mx-auto">
            <a href="/add_patient" class="block bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-center py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 shadow-md">Add Patient</a>
            <a href="/add_doctor" class="block bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-center py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 shadow-md">Add Doctor</a>
            <a href="/add_clinical" class="block bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-center py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 shadow-md">Add Clinical Data</a>
            <a href="/view_codes" class="block bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-center py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 shadow-md">View Codes</a>
        </div>
    </div>
</body>
</html>
""",
    "add_patient.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add Patient</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        }
    </style>
</head>
<body class="font-sans text-gray-900 flex items-center justify-center min-h-screen p-6">
    <div class="bg-white bg-opacity-90 backdrop-blur-lg rounded-3xl shadow-2xl max-w-2xl w-full p-10 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-4xl font-extrabold text-center mb-10 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">Add Patient</h1>
        <form method="POST" id="patient_form" class="flex flex-col gap-6 max-w-lg mx-auto">
            <div>
                <label for="patient_id" class="block text-sm font-semibold text-gray-700">Patient ID</label>
                <input type="text" id="patient_id" name="patient_id" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="name" class="block text-sm font-semibold text-gray-700">Name</label>
                <input type="text" id="name" name="name" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="gender" class="block text-sm font-semibold text-gray-700">Gender</label>
                <select id="gender" name="gender" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            <div>
                <label for="age" class="block text-sm font-semibold text-gray-700">Age</label>
                <input type="number" id="age" name="age" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="address" class="block text-sm font-semibold text-gray-700">Address</label>
                <input type="text" id="address" name="address" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="phone" class="block text-sm font-semibold text-gray-700">Phone (Discord ID)</label>
                <input type="text" id="phone" name="phone" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="id_doctor" class="block text-sm font-semibold text-gray-700">Doctor ID</label>
                <input type="text" id="id_doctor" name="id_doctor" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="condition_label" class="block text-sm font-semibold text-gray-700">Condition Label (0-4)</label>
                <input type="number" id="condition_label" name="condition_label" min="0" max="4" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="followup_date" class="block text-sm font-semibold text-gray-700">Followup Date (YYYY-MM-DD HH:MM, optional)</label>
                <input type="text" id="followup_date" name="followup_date" placeholder="e.g., 2025-07-01 09:00" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="num_meds" class="block text-sm font-semibold text-gray-700">Number of Medications</label>
                <input type="number" id="num_meds" name="num_meds" min="1" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
                <button type="button" onclick="generateMedFields()" class="mt-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white px-6 py-2 rounded-xl hover:from-indigo-600 hover:to-purple-600 transition-all duration-200 shadow-md">Generate Medication Fields</button>
            </div>
            <div id="medication_fields" class="space-y-4"></div>
            <div>
                <label for="medication_morning" class="block text-sm font-semibold text-gray-700">Morning Medication (comma-separated, HH:MM)</label>
                <input type="text" id="medication_morning" name="medication_morning" placeholder="e.g., 08:00,09:00" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="medication_noon" class="block text-sm font-semibold text-gray-700">Noon Medication (comma-separated, HH:MM)</label>
                <input type="text" id="medication_noon" name="medication_noon" placeholder="e.g., 12:00" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="medication_evening" class="block text-sm font-semibold text-gray-700">Evening Medication (comma-separated, HH:MM)</label>
                <input type="text" id="medication_evening" name="medication_evening" placeholder="e.g., 19:00" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <input type="submit" value="Submit" class="bg-gradient-to-r from-blue-500 to-indigo-500 text-white py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 shadow-md">
        </form>
    </div>
    <script>
        function generateMedFields() {
            const numMeds = parseInt(document.getElementById('num_meds').value);
            if (isNaN(numMeds) || numMeds < 1) {
                alert("Please enter a positive integer for the number of medications!");
                return;
            }
            const medFields = document.getElementById('medication_fields');
            medFields.innerHTML = '';
            for (let i = 0; i < numMeds; i++) {
                const div = document.createElement('div');
                div.className = 'med_field space-y-3 p-4 bg-gray-100 rounded-xl shadow';
                div.innerHTML = `
                    <div>
                        <label class="block text-sm font-semibold text-gray-700">Medication ${i+1} - Name</label>
                        <input type="text" name="med_name_${i}" required placeholder="e.g., Metformin"
                            class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
                    </div>
                    <div>
                        <label class="block text-sm font-semibold text-gray-700">Medication ${i+1} - Tablets Used</label>
                        <input type="text" name="used_tablets_${i}" required placeholder="e.g., 60 viên"
                            class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
                    </div>
                `;
                medFields.appendChild(div);
            }
        }
    </script>
</body>
</html>
""",
    "add_doctor.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add Doctor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        }
    </style>
</head>
<body class="font-sans text-gray-900 flex items-center justify-center min-h-screen p-6">
    <div class="bg-white bg-opacity-90 backdrop-blur-lg rounded-3xl shadow-2xl max-w-2xl w-full p-10 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-4xl font-extrabold text-center mb-10 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">Add Doctor</h1>
        <form method="POST" class="flex flex-col gap-6 max-w-lg mx-auto">
            <div>
                <label for="doctor_id" class="block text-sm font-semibold text-gray-700">Doctor ID</label>
                <input type="text" id="doctor_id" name="doctor_id" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="name" class="block text-sm font-semibold text-gray-700">Name</label>
                <input type="text" id="name" name="name" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="specialty" class="block text-sm font-semibold text-gray-700">Specialty</label>
                <input type="text" id="specialty" name="specialty" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="contact" class="block text-sm font-semibold text-gray-700">Contact</label>
                <input type="text" id="contact" name="contact" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="hospital" class="block text-sm font-semibold text-gray-700">Hospital</label>
                <input type="text" id="hospital" name="hospital" required class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <input type="submit" value="Submit" class="bg-gradient-to-r from-blue-500 to-indigo-500 text-white py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 shadow-md">
        </form>
    </div>
</body>
</html>
""",
    "add_clinical.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Add Clinical Data</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        }
    </style>
</head>
<body class="font-sans text-gray-900 flex items-center justify-center min-h-screen p-6">
    <div class="bg-white bg-opacity-90 backdrop-blur-lg rounded-3xl shadow-2xl max-w-2xl w-full p-10 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-4xl font-extrabold text-center mb-10 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">Add Clinical Data</h1>
        <div id="errors" class="text-red-500 text-sm mb-6"></div>
        <form method="POST" onsubmit="return validateForm()" class="flex flex-col gap-6 max-w-lg mx-auto">
            <div>
                <label for="age" class="block text-sm font-semibold text-gray-700">Age</label>
                <input type="number" id="age" name="age" step="1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="blood_pressure" class="block text-sm font-semibold text-gray-700">Blood Pressure (mmHg)</label>
                <input type="number" id="blood_pressure" name="blood_pressure" step="1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="glucose_level" class="block text-sm font-semibold text-gray-700">Max Glucose Level (mg/dL)</label>
                <input type="number" id="glucose_level" name="glucose_level" step="0.1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="heart_rate" class="block text-sm font-semibold text-gray-700">Heart Rate (bpm)</label>
                <input type="number" id="heart_rate" name="heart_rate" step="1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="bmi" class="block text-sm font-semibold text-gray-700">BMI</label>
                <input type="number" id="bmi" name="bmi" step="0.1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="glucose_apache" class="block text-sm font-semibold text-gray-700">Glucose APACHE (mg/dL)</label>
                <input type="number" id="glucose_apache" name="glucose_apache" step="0.1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="creatinine_apache" class="block text-sm font-semibold text-gray-700">Creatinine APACHE (mg/dL)</label>
                <input type="number" id="creatinine_apache" name="creatinine_apache" step="0.1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="sodium_apache" class="block text-sm font-semibold text-gray-700">Sodium APACHE (mEq/L)</label>
                <input type="number" id="sodium_apache" name="sodium_apache" step="0.1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="albumin_apache" class="block text-sm font-semibold text-gray-700">Albumin APACHE (g/dL)</label>
                <input type="number" id="albumin_apache" name="albumin_apache" step="0.1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="blood_sugar" class="block text-sm font-semibold text-gray-700">Min Blood Sugar (mg/dL)</label>
                <input type="number" id="blood_sugar" name="blood_sugar" step="0.1" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
            </div>
            <div>
                <label for="gender" class="block text-sm font-semibold text-gray-700">Gender</label>
                <select id="gender" name="gender" class="mt-2 block w-full p-3 border border-gray-200 rounded-xl bg-gray-50 focus:border-indigo-500 focus:ring focus:ring-indigo-200 transition-all duration-200">
                    <option value="">Select</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            <input type="submit" value="Submit" class="bg-gradient-to-r from-blue-500 to-indigo-500 text-white py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 shadow-md">
        </form>
    </div>
    <script>
        function validateForm() {
            const fields = ['age', 'blood_pressure', 'glucose_level', 'heart_rate', 'bmi', 'glucose_apache', 'creatinine_apache', 'sodium_apache', 'albumin_apache', 'blood_sugar'];
            let hasValue = false;
            let errors = [];
            for (let field of fields) {
                let value = document.getElementById(field).value;
                if (value) {
                    hasValue = true;
                    let num = parseFloat(value);
                    if (isNaN(num) || num < 0) {
                        errors.push(`${field.replace('_', ' ').toUpperCase()} must be a non-negative number`);
                    } else {
                        if (field === 'blood_pressure' && (num < 50 || num > 250)) {
                            errors.push('Blood pressure must be between 50-250 mmHg');
                        }
                        if (field === 'heart_rate' && (num < 30 || num > 200)) {
                            errors.push('Heart rate must be between 30-200 bpm');
                        }
                        if (field === 'bmi' && (num < 10 || num > 60)) {
                            errors.push('BMI must be between 10-60');
                        }
                        if (field === 'blood_sugar' && (num < 30 || num > 500)) {
                            errors.push('Min blood sugar must be between 30-500 mg/dL');
                        }
                    }
                }
            }
            let gender = document.getElementById('gender').value;
            if (gender && !['Male', 'Female'].includes(gender)) {
                errors.push('Gender must be Male or Female');
            }
            if (!hasValue && !gender) {
                errors.push('At least one field must be filled');
            }
            let errorDiv = document.getElementById('errors');
            errorDiv.innerHTML = errors.map(err => `<p>${err}</p>`).join('');
            return errors.length === 0;
        }
    </script>
</body>
</html>
""",
    "view_codes.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>View Codes</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        }
        .content-cell {
            max-width: 400px; /* Giới hạn chiều rộng cột Content */
            overflow-x: auto; /* Cho phép cuộn ngang */
        }
        .content-cell pre {
            white-space: pre-wrap; /* Cho phép xuống dòng tự động */
            text-overflow: ellipsis; /* Hiển thị dấu ... khi nội dung bị cắt */
            overflow: hidden; /* Ẩn nội dung tràn */
        }
    </style>
</head>
<body class="font-sans text-gray-900 flex items-center justify-center min-h-screen p-6">
    <div class="bg-white bg-opacity-90 backdrop-blur-lg rounded-3xl shadow-2xl max-w-5xl w-full p-10 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-4xl font-extrabold text-center mb-10 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">List of Codes</h1>
        <div class="overflow-x-auto">
            <table class="w-full border-collapse">
                <thead>
                    <tr class="bg-gradient-to-r from-blue-500 to-indigo-500 text-white">
                        <th class="p-4 text-left font-semibold">Type</th>
                        <th class="p-4 text-left font-semibold">File Name</th>
                        <th class="p-4 text-left font-semibold">Content</th>
                    </tr>
                </thead>
                <tbody>
                    {% for file in files %}
                    <tr class="hover:bg-gray-100 transition-colors duration-200">
                        <td class="p-4 border-b border-gray-200">{{ file.type }}</td>
                        <td class="p-4 border-b border-gray-200">{{ file.name }}</td>
                        <td class="p-4 border-b border-gray-200 content-cell">
                            {% if file.content %}
                            <pre class="bg-gray-50 p-4 rounded-xl max-h-64 overflow-auto text-sm font-mono">{{ file.content | safe }}</pre>
                            {% else %}
                            Unable to display content
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        <a href="/" class="block mt-8 bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-center py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-indigo-600 transition-all duration-200 shadow-md max-w-sm mx-auto">Back to Home</a>
    </div>
</body>
</html>
"""
}

TEMPLATES["result_clinical.html"] = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Prediction Results</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        }

        /* Bảng đẹp hơn và dễ đọc */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
            text-align: left;
        }

        thead {
            background-color: #e0e7ff; /* Indigo-100 */
            font-weight: 600;
        }

        th, td {
            padding: 0.75rem 1rem;
            border: 1px solid #d1d5db; /* Gray-300 */
            white-space: nowrap; /* Không cho xuống dòng */
        }

        tbody tr:nth-child(even) {
            background-color: #f9fafb; /* Gray-50 */
        }
    </style>
</head>
<body class="font-sans text-gray-900 flex items-center justify-center min-h-screen p-6">
    <div class="bg-white bg-opacity-90 backdrop-blur-lg rounded-3xl shadow-2xl max-w-3xl w-full p-10 transform hover:scale-105 transition-transform duration-300">
        <h1 class="text-4xl font-extrabold text-center mb-10 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
            Clinical Prediction Results
        </h1>

        <div class="space-y-8">
            <!-- Summary -->
            <div class="bg-gray-50 p-6 rounded-xl shadow-sm">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Prediction Summary</h2>
                <div class="space-y-3">
                    <p class="text-sm"><span class="font-medium">Severity Level:</span> {{ severity }}</p>
                    <p class="text-sm"><span class="font-medium">Probability:</span> {{ probability | round(4) }}</p>
                </div>
            </div>

            <!-- Detailed Table -->
            <div class="bg-gray-50 p-6 rounded-xl shadow-sm">
                <h2 class="text-xl font-semibold text-gray-800 mb-4">Detailed Data</h2>
                <div class="overflow-x-auto max-w-full">
                    {{ result_table | safe }}
                </div>
            </div>
        </div>

        <!-- Buttons -->
        <div class="mt-10 flex justify-center gap-4">
            <a href="/add_clinical" class="inline-block bg-gradient-to-r from-indigo-500 to-purple-500 text-white px-6 py-3 rounded-xl hover:from-indigo-600 hover:to-purple-600 transition-all duration-200 shadow-md">
                Back to Data Entry
            </a>
            <a href="/" class="inline-block bg-gradient-to-r from-gray-500 to-gray-600 text-white px-6 py-3 rounded-xl hover:from-gray-600 hover:to-gray-700 transition-all duration-200 shadow-md">
                Home
            </a>
        </div>
    </div>
</body>
</html>

"""

# Lưu templates HTML
os.makedirs('templates', exist_ok=True)
for filename, content in TEMPLATES.items():
    with open(os.path.join('templates', filename), 'w') as f:
        f.write(content)

from flask import Flask, render_template
import os
import json
import pandas as pd

@app.route('/view_codes')
def view_codes():
    files = []
    # Thu thập file từ TEMPLATES_DIR (JSON templates)
    for filename in os.listdir(TEMPLATES_DIR):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(TEMPLATES_DIR, filename), 'r') as f:
                    content = json.dumps(json.load(f), indent=2, ensure_ascii=False)
                file_type = filename.split('_')[0].capitalize()
                files.append({'type': file_type, 'name': filename, 'content': content})
            except Exception as e:
                files.append({'type': 'JSON', 'name': filename, 'content': None})

    # Thu thập file từ OUTPUT_DIR (JSON và CSV)
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(OUTPUT_DIR, filename), 'r') as f:
                    content = json.dumps(json.load(f), indent=2, ensure_ascii=False)
                files.append({'type': 'Image Result', 'name': filename, 'content': content})
            except Exception as e:
                files.append({'type': 'Image Result', 'name': filename, 'content': None})
        elif filename.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(OUTPUT_DIR, filename))
                content = df.to_string(index=False)
                files.append({'type': 'Clinical Data', 'name': filename, 'content': content})
            except Exception as e:
                files.append({'type': 'Clinical Data', 'name': filename, 'content': None})

    return render_template('view_codes.html', files=files)

@app.route('/add_doctor', methods=['GET', 'POST'])
def web_add_doctor():
    if request.method == 'POST':
        try:
            data = {
                "doctor_id": request.form.get('doctor_id'),
                "name": request.form.get('name'),
                "specialty": request.form.get('specialty'),
                "contact": request.form.get('contact'),
                "hospital": request.form.get('hospital')
            }
            # Kiểm tra dữ liệu bắt buộc
            required_fields = ["doctor_id", "name", "specialty", "contact", "hospital"]
            missing_fields = [field for field in required_fields if not data[field]]
            if missing_fields:
                error_msg = f"Thiếu các trường bắt buộc: {', '.join(missing_fields)}"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400
            # Kiểm tra doctor_id đã tồn tại
            doctors_df = pd.read_csv(LOCAL_DOCTORS_CSV)
            if data['doctor_id'] in doctors_df['doctor_id'].values:
                error_msg = f"Bác sĩ ID {data['doctor_id']} đã tồn tại"
                logger.error(error_msg)
                return jsonify({"status": "error", "message": error_msg}), 400
            # Lưu bác sĩ mới
            new_doctor = pd.DataFrame([data])
            doctors_df = pd.concat([doctors_df, new_doctor], ignore_index=True)
            save_to_csv(doctors_df, LOCAL_DOCTORS_CSV)
            # Lưu template
            template_path = save_json_template(data, "doctor", "web")
            logger.info(f"Đã thêm bác sĩ {data['doctor_id']} từ web")
            return jsonify({"status": "success", "message": "Đã thêm bác sĩ thành công!"})
        except Exception as e:
            logger.error(f"Lỗi trong web_add_doctor: {str(e)}")
            return jsonify({"status": "error", "message": f"Lỗi: {str(e)}"}), 500
    return render_template('add_doctor.html')

# Run Flask in a separate thread
def run_flask():
    app.run(host='0.0.0.0', port=5000)

flask_thread = Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    logger.info(f'Bot connected: {bot.user}')
    await bot.change_presence(activity=discord.Game(name="Managing patients! Use !start to begin."))
    check_schedules.start()
    check_api_data.start()

@bot.command()
async def start(ctx):
    role = determine_role(ctx.channel.name)
    if role == 'patient':
        await ctx.send(
            "👋 **Hello!** I’m your patient support bot.\n\n"
            "📌 I will automatically send you reminders for medications and follow-up visits.\n"
            "📩 Please wait for my notifications!"
        )
    elif role == 'doctor':
        await ctx.send(
            "👋 **Hello Doctor!** I’m your medical assistant bot.\n\n"
            "📋 **Available commands:**\n"
            "- `!add_patient <JSON>`: Add a new patient.\n"
            "- `!add_patient_interactive`: Add patient via DM.\n"
            "- `!image`: Analyze medical image.\n"
            "- `!clinical <JSON>`: Analyze clinical data.\n"
            "- `!clinical_interactive`: Analyze clinical data via DM.\n"
            "- `!add_cv <JSON>`: Add doctor information.\n"
            "- `!add_cv_interactive`: Add doctor via DM.\n"
            "- `!template <patient|doctor|clinical>`: View input template.\n"
            "- `!list_templates <patient|doctor|clinical>`: List all templates.\n"
            "- `!view_clinical_history <patient_id>`: View patient's clinical history.\n\n"
            f"🌐 **Web Interface**: Visit {ngrok_url} to enter information via form."
        )
    else:
        await ctx.send("⚠️ Please use the bot in a patient or doctor-specific channel.")



@bot.command()
async def template(ctx, template_type: str):
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Only doctors can use this command!")
        return

    template_type = template_type.lower()

    if template_type == "patient":
        response = (
            "📋 **Template for adding a patient (!add_patient)**\n\n"
            "| Field | Required | Data Type | Description | Example |\n"
            "|-------|----------|-----------|-------------|---------|\n"
            "| patient_id | Yes | String | Unique patient identifier | P001 |\n"
            "| name | Yes | String | Patient name | Nguyen Van A |\n"
            "| gender | Yes | String (Male/Female) | Gender | Male |\n"
            "| age | Yes | Integer | Age | 45 |\n"
            "| address | Yes | String | Address | 123 Hanoi |\n"
            "| phone | Yes | String | Phone number (Discord ID) | 123456789 |\n"
            "| id_doctor | Yes | String | Doctor ID (must exist) | D001 |\n"
            "| condition_label | Yes | Integer (0-4) | Condition severity (0: very mild, 4: very severe) | 2 |\n"
            "| prescription | Yes | JSON list | Medication list (name, used tablets) | [{\"name\": \"Metformin\", \"used_tablets\": \"60 tablets\"}] |\n"
            "| medication_schedule | Yes | JSON object | Medication schedule (morning, noon, evening) | {\"morning\": [\"08:00\"], \"noon\": [], \"evening\": [\"19:00\"]} |\n"
            "| followup_date | No | String (YYYY-MM-DD HH:MM) | Follow-up date, auto-generated if not provided | 2025-07-01 09:00 |\n\n"
            "📌 **How to use**:\n1. Copy the sample JSON below.\n2. Fill in the fields.\n3. Send the command: `!add_patient <JSON>`.\n\n"
            "**Sample JSON**:\n```json\n"
            "{\"patient_id\": \"P001\", \"name\": \"Nguyen Van A\", \"gender\": \"Male\", \"age\": 45, "
            "\"address\": \"123 Hanoi\", \"phone\": \"123456789\", \"id_doctor\": \"D001\", "
            "\"condition_label\": 2, \"prescription\": [{\"name\": \"Metformin\", \"used_tablets\": \"60 tablets\"}, {\"name\": \"Amlodipine\", \"used_tablets\": \"30 tablets\"}], "
            "\"medication_schedule\": {\"morning\": [\"08:00\"], \"noon\": [], \"evening\": [\"19:00\"]}, "
            "\"followup_date\": \"2025-07-01 09:00\"}\n```"
        )
    elif template_type == "doctor":
        response = (
            "📋 **Template for adding a doctor (!add_cv)**\n\n"
            "| Field | Required | Data Type | Description | Example |\n"
            "|--------|----------|------------|-------------|---------|\n"
            "| doctor_id | Yes | String | Unique doctor identifier | D001 |\n"
            "| name | Yes | String | Doctor name | Tran Thi B |\n"
            "| specialty | Yes | String | Specialty | Internal Medicine |\n"
            "| contact | Yes | String | Contact info (email/phone) | doctorB@email.com |\n"
            "| hospital | Yes | String | Affiliated hospital | Central Hospital |\n\n"
            "📌 **How to use**:\n1. Copy the sample JSON below.\n2. Fill in the fields.\n3. Send the command: `!add_cv <JSON>`.\n\n"
            "**Sample JSON**:\n```json\n"
            "{\"doctor_id\": \"D001\", \"name\": \"Tran Thi B\", \"specialty\": \"Internal Medicine\", "
            "\"contact\": \"doctorB@email.com\", \"hospital\": \"Central Hospital\"}\n```"
        )
    elif template_type == "clinical":
        response = (
            "📋 **Template for clinical data analysis (!clinical)**\n\n"
            "| Field | Required | Data Type | Description | Example |\n"
            "|--------|----------|------------|-------------|---------|\n"
            "| age | No | Float | Age | 45 |\n"
            "| blood_pressure | No | Float | Blood pressure (mmHg) | 120 |\n"
            "| glucose_level | No | Float | Glucose level (mg/dL) | 90 |\n"
            "| heart_rate | No | Float | Heart rate (bpm) | 80 |\n"
            "| bmi | No | Float | Body Mass Index | 25.5 |\n"
            "| cholesterol | No | Float | Total cholesterol (mg/dL) | 200 |\n"
            "| triglycerides | No | Float | Triglycerides (mg/dL) | 150 |\n"
            "| gender | No | String (Male/Female) | Gender | Male |\n\n"
            "📌 **How to use**:\n1. Copy the sample JSON below.\n2. Fill in at least one field.\n3. Send the command: `!clinical <JSON>`.\n\n"
            "**Sample JSON**:\n```json\n"
            "{\"age\": 45, \"blood_pressure\": 120, \"glucose_level\": 90, \"heart_rate\": 80, "
            "\"bmi\": 25.5, \"cholesterol\": 200, \"triglycerides\": 150, \"gender\": \"Male\"}\n```"
        )
    else:
        response = "⚠️ Invalid template type. Please choose one of: `patient`, `doctor`, or `clinical`."
    await ctx.send(response)


@bot.command()
async def list_templates(ctx, template_type: str):
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Only doctors can use this command!")
        return

    template_type = template_type.lower()
    if template_type not in ['patient', 'doctor', 'clinical']:
        await ctx.send("⚠️ Invalid template type. Please choose one of: `patient`, `doctor`, or `clinical`.")
        return

    templates = [f for f in os.listdir(TEMPLATES_DIR) if f.startswith(template_type) and f.endswith('.json')]
    if not templates:
        await ctx.send(f"⚠️ No templates found for type `{template_type}`.")
        return

    response = f"📋 **List of `{template_type}` templates**:\n"
    for template in templates:
        with open(os.path.join(TEMPLATES_DIR, template), 'r') as f:
            data = json.load(f)
        response += f"- `{template}`: ```json\n{json.dumps(data, indent=2)}\n```\n"

    await ctx.send(response)


@bot.command()
async def add_patient(ctx, *, json_data=None):
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Only doctors can use this command!")
        return

    if json_data:
        data, error = validate_json(json_data)
        if error:
            await ctx.send(f"❌ {error}")
            return

        try:
            required_fields = ["patient_id", "name", "gender", "age", "address", "phone", "id_doctor", "condition_label", "medication_schedule", "prescription"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                await ctx.send(f"⚠️ Missing required fields: {', '.join(missing_fields)}")
                return

            if data['gender'] not in ['Male', 'Female']:
                await ctx.send("⚠️ Gender must be either 'Male' or 'Female'")
                return

            if not isinstance(data['age'], (int, float)) or data['age'] < 0:
                await ctx.send("⚠️ Age must be a non-negative number")
                return

            if not isinstance(data['condition_label'], int) or not (0 <= data['condition_label'] < NUM_CLASSES):
                await ctx.send("⚠️ Condition label must be an integer from 0 to 4")
                return

            if not isinstance(data['medication_schedule'], dict):
                await ctx.send("⚠️ Medication schedule must be a JSON object")
                return

            if not isinstance(data['prescription'], list) or not data['prescription']:
                await ctx.send("⚠️ Prescription must be a non-empty list of medications")
                return

            for med in data['prescription']:
                if not all(key in med for key in ['name', 'used_tablets']):
                    await ctx.send("⚠️ Each medication must include 'name' and 'used_tablets'")
                    return

            for time_slot in data['medication_schedule']:
                if time_slot not in ['morning', 'noon', 'evening']:
                    await ctx.send("⚠️ Time slots must be 'morning', 'noon', or 'evening'")
                    return
                if not isinstance(data['medication_schedule'][time_slot], list):
                    await ctx.send(f"⚠️ Value of '{time_slot}' must be a list of time strings")
                    return
                for time in data['medication_schedule'][time_slot]:
                    try:
                        datetime.datetime.strptime(time, '%H:%M')
                    except ValueError:
                        await ctx.send(f"⚠️ Invalid time format '{time}'. Use HH:MM format.")
                        return

            followup_date = data.get('followup_date', schedule_followup(data['condition_label']))
            try:
                datetime.datetime.strptime(followup_date, '%Y-%m-%d %H:%M')
            except ValueError:
                await ctx.send("⚠️ Follow-up date must be in format YYYY-MM-DD HH:MM or left blank for auto-generation.")
                return

            patients_df = pd.read_csv(LOCAL_PATIENTS_CSV)
            if data['patient_id'] in patients_df['patient_id'].values:
                await ctx.send(f"⚠️ Patient ID '{data['patient_id']}' already exists.")
                return

            doctors_df = pd.read_csv(LOCAL_DOCTORS_CSV)
            if data['id_doctor'] not in doctors_df['doctor_id'].values:
                await ctx.send(f"⚠️ Doctor ID '{data['id_doctor']}' does not exist.")
                return

            clinical_data = data.get('clinical_data', {})
            new_patient = {
                "patient_id": data['patient_id'], "name": data['name'], "gender": data['gender'],
                "age": data['age'], "address": data['address'], "phone": data['phone'],
                "clinical_data": json.dumps(clinical_data), "condition_label": data['condition_label'],
                "id_doctor": data['id_doctor'], "prescription": json.dumps(data['prescription']),
                "medication_schedule": json.dumps(data['medication_schedule']),
                "followup_date": followup_date
            }

            patients_df = pd.concat([patients_df, pd.DataFrame([new_patient])], ignore_index=True)
            save_to_csv(patients_df, LOCAL_PATIENTS_CSV)

            template_path = save_json_template(data, "patient", str(ctx.author.id))

            response = (
                f"✅ Patient `{data['patient_id']}` has been added successfully.\n"
                f"📅 **Follow-up date**: {followup_date}\n"
                f"**Prescription**: ```json\n{json.dumps(data['prescription'], indent=2)}\n```\n"
                f"**Assigned Doctor Info**:\n{get_doctor_info(data['id_doctor'])}"
            )

            if template_path:
                response += f"\n💾 Template saved at: `{template_path}`"

            await ctx.send(response)

        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")
            logger.error(f"Error while adding patient: {e}")
    else:
        await ctx.send("⚠️ Please provide JSON data or use `!add_patient_interactive` to enter via DM.")


@bot.command()
async def add_patient_interactive(ctx):
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Only doctors can use this command!")
        return
    await ctx.send("📩 Please check your DMs to enter patient information.")
    user = ctx.author
    data = {}
    fields = [
        ("patient_id", "Enter the patient's unique ID (e.g., P001):"),
        ("name", "Enter the patient's name (e.g., Nguyen Van A):"),
        ("gender", "Enter gender (Male/Female):"),
        ("age", "Enter age (non-negative number):"),
        ("address", "Enter address (e.g., 123 Hanoi):"),
        ("phone", "Enter phone number or Discord ID (e.g., 123456789):"),
        ("id_doctor", "Enter doctor ID (e.g., D001):"),
        ("condition_label", "Enter condition severity (integer from 0-4):"),
        ("followup_date", "Enter follow-up date (YYYY-MM-DD HH:MM, e.g., 2025-07-01 09:00, leave blank to auto-generate):"),
        ("medication_morning", "Enter morning medication times (HH:MM, separated by commas, e.g., 08:00,09:00):"),
        ("medication_noon", "Enter noon medication times (HH:MM, separated by commas, leave blank if none):"),
        ("medication_evening", "Enter evening medication times (HH:MM, separated by commas, e.g., 19:00):"),
    ]

    # Ask for number of medications
    await user.send("Enter the number of medications in the prescription (positive integer):")
    def check(m):
        return m.author == user and m.channel == user.dm_channel
    try:
        msg = await bot.wait_for('message', check=check, timeout=300)
        num_meds = msg.content.strip()
        try:
            num_meds = int(num_meds)
            if num_meds <= 0:
                await user.send("⚠️ Medication count must be a positive integer. Please try again.")
                return
        except ValueError:
            await user.send("⚠️ Medication count must be an integer. Please try again.")
            return
    except asyncio.TimeoutError:
        await user.send("⚠️ Time expired. Please try again with `!add_patient_interactive`.")
        return

    # Enter each medication's details
    prescription = []
    for i in range(num_meds):
        await user.send(f"Enter details for medication #{i+1}:\n- Medication name (e.g., Metformin):")
        try:
            msg = await bot.wait_for('message', check=check, timeout=300)
            med_name = msg.content.strip()
            if not med_name:
                await user.send("⚠️ Medication name cannot be empty. Please try again.")
                return
            await user.send(f"- Number of tablets used (e.g., 60 viên):")
            msg = await bot.wait_for('message', check=check, timeout=300)
            used_tablets = msg.content.strip()
            if not used_tablets:
                await user.send("⚠️ Tablet count cannot be empty. Please try again.")
                return
            if not used_tablets.endswith("viên"):
                await user.send("⚠️ Tablet count must end with 'viên' (e.g., 60 viên). Please try again.")
                return
            try:
                tablet_count = int(used_tablets.replace("viên", "").strip())
                if tablet_count <= 0:
                    raise ValueError
            except ValueError:
                await user.send("⚠️ Tablet count must be a positive integer (e.g., 60 viên). Please try again.")
                return
            prescription.append({"name": med_name, "used_tablets": used_tablets})
            await user.send(f"Added medication: {med_name}, {used_tablets}.")
        except asyncio.TimeoutError:
            await user.send("⚠️ Time expired. Please try again with `!add_patient_interactive`.")
            return
    data['prescription'] = prescription

    # Enter other patient fields
    for field, prompt in fields:
        await user.send(prompt)
        try:
            msg = await bot.wait_for('message', check=check, timeout=300)
            value = msg.content.strip()
            if field == 'gender' and value not in ['Male', 'Female']:
                await user.send("⚠️ Gender must be 'Male' or 'Female'. Please try again.")
                return
            if field == 'age':
                try:
                    value = int(value)
                    if value < 0:
                        await user.send("⚠️ Age must be a non-negative number. Please try again.")
                        return
                except ValueError:
                    await user.send("⚠️ Age must be an integer. Please try again.")
                    return
            if field == 'condition_label':
                try:
                    value = int(value)
                    if value < 0 or value >= NUM_CLASSES:
                        await user.send("⚠️ Condition severity must be an integer from 0 to 4. Please try again.")
                        return
                except ValueError:
                    await user.send("⚠️ Condition severity must be an integer. Please try again.")
                    return
            if field == 'followup_date' and value:
                try:
                    datetime.datetime.strptime(value, '%Y-%m-%d %H:%M')
                except ValueError:
                    await user.send("⚠️ Follow-up date must be in format YYYY-MM-DD HH:MM (e.g., 2025-07-01 09:00). Please try again.")
                    return
            if field.startswith('medication_'):
                if value:
                    times = [t.strip() for t in value.split(',')]
                    for t in times:
                        try:
                            datetime.datetime.strptime(t, '%H:%M')
                        except ValueError:
                            await user.send(f"⚠️ Time '{t}' is not in correct format HH:MM. Please try again.")
                            return
                    data[field] = times
                else:
                    data[field] = []
            else:
                data[field] = value
        except asyncio.TimeoutError:
            await user.send("⚠️ Time expired. Please try again with `!add_patient_interactive`.")
            return

    # Create medication_schedule
    data['medication_schedule'] = {
        'morning': data.pop('medication_morning'),
        'noon': data.pop('medication_noon'),
        'evening': data.pop('medication_evening')
    }

    # Generate followup_date if not provided
    if not data['followup_date']:
        data['followup_date'] = schedule_followup(data['condition_label'])

    # Check and save data
    patients_df = pd.read_csv(LOCAL_PATIENTS_CSV)
    if data['patient_id'] in patients_df['patient_id'].values:
        await user.send(f"⚠️ Patient ID {data['patient_id']} already exists.")
        return
    doctors_df = pd.read_csv(LOCAL_DOCTORS_CSV)
    if data['id_doctor'] not in doctors_df['doctor_id'].values:
        await user.send(f"⚠️ Doctor ID {data['id_doctor']} does not exist.")
        return
    clinical_data = data.get('clinical_data', {})
    new_patient = {
        "patient_id": data['patient_id'], "name": data['name'], "gender": data['gender'],
        "age": data['age'], "address": data['address'], "phone": data['phone'],
        "clinical_data": json.dumps(clinical_data), "condition_label": data['condition_label'],
        "id_doctor": data['id_doctor'], "prescription": json.dumps(data['prescription']),
        "medication_schedule": json.dumps(data['medication_schedule']),
        "followup_date": data['followup_date']
    }
    patients_df = pd.concat([patients_df, pd.DataFrame([new_patient])], ignore_index=True)
    save_to_csv(patients_df, LOCAL_PATIENTS_CSV)
    template_path = save_json_template(data, "patient", str(ctx.author.id))
    response = (
        f"✅ Patient {data['patient_id']} added.\n"
        f"📅 **Follow-up**: {data['followup_date']}\n"
        f"**Prescription**: {json.dumps(data['prescription'], indent=2)}\n"
        f"**Assigned Doctor Info**\n{get_doctor_info(data['id_doctor'])}"
    )
    if template_path:
        response += f"\n💾 Template saved at: `{template_path}`"
    await user.send(response)


@bot.command()
async def add_cv(ctx, *, json_data=None):
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Chỉ bác sĩ mới có thể sử dụng lệnh này!")
        return
    if json_data:
        data, error = validate_json(json_data)
        if error:
            await ctx.send(f"❌ {error}")
            return
        try:
            required_fields = ["doctor_id", "name", "specialty", "contact", "hospital"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                await ctx.send(f"⚠️ Thiếu các trường bắt buộc: {', '.join(missing_fields)}")
                return
            doctors_df = pd.read_csv(LOCAL_DOCTORS_CSV)
            if data['doctor_id'] in doctors_df['doctor_id'].values:
                await ctx.send(f"⚠️ Bác sĩ ID {data['doctor_id']} đã tồn tại")
                return
            new_doctor = {
                "doctor_id": data['doctor_id'], "name": data['name'], "specialty": data['specialty'],
                "contact": data['contact'], "hospital": data['hospital']
            }
            doctors_df = pd.concat([doctors_df, pd.DataFrame([new_doctor])], ignore_index=True)
            save_to_csv(doctors_df, LOCAL_DOCTORS_CSV)
            template_path = save_json_template(data, "doctor", str(ctx.author.id))
            response = f"✅ Đã thêm bác sĩ {data['doctor_id']}."
            if template_path:
                response += f"\n💾 Template đã lưu tại: `{template_path}`"
            await ctx.send(response)
        except Exception as e:
            await ctx.send(f"❌ Lỗi: {str(e)}")
            logger.error(f"Lỗi thêm bác sĩ: {e}")
    else:
        await ctx.send("⚠️ Vui lòng cung cấp JSON hoặc dùng `!add_cv_interactive` để nhập qua DM.")


@bot.command()
async def add_cv_interactive(ctx):
    # Only doctors can use this command
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Only doctors can use this command!")
        return

    await ctx.send("📩 Please check your DM to enter doctor information.")
    user = ctx.author
    data = {}

    # Define required fields and prompts
    fields = [
        ("doctor_id", "Enter doctor ID (unique, e.g., D001):"),
        ("name", "Enter doctor’s name (e.g., Tran Thi B):"),
        ("specialty", "Enter specialty (e.g., Internal Medicine):"),
        ("contact", "Enter contact information (email/phone, e.g., doctorB@email.com):"),
        ("hospital", "Enter affiliated hospital (e.g., Central Hospital):")
    ]

    # Collect user input for each field via DM
    for field, prompt in fields:
        await user.send(prompt)
        def check(m):
            return m.author == user and m.channel == user.dm_channel
        try:
            msg = await bot.wait_for('message', check=check, timeout=300)
            data[field] = msg.content.strip()
        except asyncio.TimeoutError:
            await user.send("⚠️ Time out. Please try again with `!add_cv_interactive`.")
            return

    # Check for duplicate doctor ID
    doctors_df = pd.read_csv(LOCAL_DOCTORS_CSV)
    if data['doctor_id'] in doctors_df['doctor_id'].values:
        await user.send(f"⚠️ Doctor ID {data['doctor_id']} already exists.")
        return

    # Append new doctor to the CSV
    new_doctor = {
        "doctor_id": data['doctor_id'],
        "name": data['name'],
        "specialty": data['specialty'],
        "contact": data['contact'],
        "hospital": data['hospital']
    }
    doctors_df = pd.concat([doctors_df, pd.DataFrame([new_doctor])], ignore_index=True)
    save_to_csv(doctors_df, LOCAL_DOCTORS_CSV)

    # Save as template (optional)
    template_path = save_json_template(data, "doctor", str(ctx.author.id))
    response = f"✅ Doctor {data['doctor_id']} has been added."
    if template_path:
        response += f"\n💾 Template saved at: `{template_path}`"
    await user.send(response)



@bot.command()
async def clinical(ctx, *, json_data=None):
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Only doctors can use this command!")
        return
    if json_data:
        data, error = validate_json(json_data)
        if error:
            await ctx.send(f"❌ {error}")
            return
        try:
            errors = validate_clinical_data(data)
            if errors:
                await ctx.send(f"⚠️ Invalid data:\n" + "\n".join(errors))
                return
            await ctx.send("✅ Processing clinical data...")
            severity, probs, extended_df = predict_clinical(data, training_file_path, clinical_model_path)
            severity_map = {0: "Very Mild", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Very Severe"}
            response = (
                f"🎯 **Clinical Prediction Result**\n"
                f"🔹 Severity: **{severity}** ({severity_map.get(severity, 'Unknown')})\n"
                f"🔹 Probability: {np.max(probs):.4f}\n"
            )
            if not extended_df.empty:
                extended_df['predicted_severity'] = severity
                # Generate table with all columns
                table = "**Full Data**\n| " + " | ".join(extended_df.columns) + " |\n"
                table += "|---" * len(extended_df.columns) + "|\n"
                for _, row in extended_df.iterrows():
                    row_values = []
                    for val in row:
                        if pd.isna(val):
                            row_values.append("N/A")
                        else:
                            row_values.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))
                    table += "| " + " | ".join(row_values) + " |\n"
                response += f"\n{table}"
                output_path = os.path.join(OUTPUT_DIR, f"clinical_{ctx.author.id}_{int(time.time())}.csv")
                extended_df.to_csv(output_path, index=False)
                shutil.copy(output_path, os.path.join(DATA_DIR, os.path.basename(output_path)))
                response += f"\n💾 Data saved at: `{output_path}`"
            template_path = save_json_template(data, "clinical", str(ctx.author.id))
            if template_path:
                response += f"\n💾 Template saved at: `{template_path}`"
            # Split the response if too long
            if len(response) > 2000:
                parts = [response[i:i+1900] for i in range(0, len(response), 1900)]
                for part in parts:
                    await ctx.send(part)
            else:
                await ctx.send(response)
        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")
            logger.error(f"Clinical processing error: {e}")
    else:
        await ctx.send("⚠️ Please provide JSON data or use `!clinical_interactive` to input via DM.")


@bot.command()
async def clinical_interactive(ctx):
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Only doctors can use this command!")
        return
    await ctx.send("📩 Please check your DMs to input clinical data. Type `cancel` at any time to abort.")
    user = ctx.author
    data = {}
    fields = [
        ("age", "Enter age (float, leave blank if unknown):"),
        ("blood_pressure", "Enter blood pressure (mmHg, float, leave blank if unknown):"),
        ("glucose_level", "Enter max glucose level (mg/dL, float, leave blank if unknown):"),
        ("heart_rate", "Enter heart rate (bpm, float, leave blank if unknown):"),
        ("bmi", "Enter BMI (float, leave blank if unknown):"),
        ("glucose_apache", "Enter APACHE glucose (mg/dL, float, leave blank if unknown):"),
        ("creatinine_apache", "Enter APACHE creatinine (mg/dL, float, leave blank if unknown):"),
        ("sodium_apache", "Enter APACHE sodium (mEq/L, float, leave blank if unknown):"),
        ("albumin_apache", "Enter APACHE albumin (g/dL, float, leave blank if unknown):"),
        ("blood_sugar", "Enter minimum blood glucose (mg/dL, float, leave blank if unknown):"),
        ("gender", "Enter gender (Male/Female, leave blank if unknown):")
    ]

    for field, prompt in fields:
        while True:
            await user.send(prompt)
            def check(m):
                return m.author == user and m.channel == user.dm_channel
            try:
                msg = await bot.wait_for('message', check=check, timeout=300)
                value = msg.content.strip()
                if value.lower() == 'cancel':
                    await user.send("❌ Clinical data entry canceled.")
                    return
                if value:
                    if field in ['age', 'blood_pressure', 'glucose_level', 'heart_rate', 'bmi',
                                 'glucose_apache', 'creatinine_apache', 'sodium_apache',
                                 'albumin_apache', 'blood_sugar']:
                        try:
                            value = float(value)
                            if value < 0:
                                await user.send(f"⚠️ {field} must be a non-negative number. Please try again or type `cancel` to exit.")
                                continue
                            if field == 'blood_pressure' and not (50 <= value <= 250):
                                await user.send("⚠️ Blood pressure must be between 50-250 mmHg. Please try again or type `cancel`.")
                                continue
                            if field == 'heart_rate' and not (30 <= value <= 200):
                                await user.send("⚠️ Heart rate must be between 30-200 bpm. Please try again or type `cancel`.")
                                continue
                            if field == 'bmi' and not (10 <= value <= 60):
                                await user.send("⚠️ BMI must be between 10-60. Please try again or type `cancel`.")
                                continue
                            if field == 'blood_sugar' and not (30 <= value <= 500):
                                await user.send("⚠️ Blood sugar must be between 30-500 mg/dL. Please try again or type `cancel`.")
                                continue
                        except ValueError:
                            await user.send(f"⚠️ {field} must be a float number. Please try again or type `cancel`.")
                            continue
                    elif field == 'gender' and value not in ['Male', 'Female']:
                        await user.send("⚠️ Gender must be 'Male' or 'Female'. Please try again or type `cancel`.")
                        continue
                    data[field] = value
                break
            except asyncio.TimeoutError:
                await user.send("⚠️ Timeout. Please try again with `!clinical_interactive`.")
                return

    if not data:
        await user.send("⚠️ At least one field is required. Please try again.")
        return

    summary = "**Confirm Clinical Data**\n"
    for field, value in data.items():
        summary += f"- {field}: {value}\n"
    summary += "\nType `confirm` to continue or `cancel` to abort."
    await user.send(summary)

    def check_confirm(m):
        return m.author == user and m.channel == user.dm_channel and m.content.strip().lower() in ['confirm', 'cancel']
    try:
        msg = await bot.wait_for('message', check=check_confirm, timeout=300)
        if msg.content.strip().lower() == 'cancel':
            await user.send("❌ Clinical data entry canceled.")
            return
    except asyncio.TimeoutError:
        await user.send("⚠️ Confirmation timeout. Please try again with `!clinical_interactive`.")
        return

    await user.send("✅ Processing clinical data...")
    try:
        # Define usecols explicitly, matching predict_clinical
        usecols = [
            'age', 'bmi', 'gender',
            'glucose_apache', 'd1_glucose_max', 'd1_glucose_min',
            'creatinine_apache', 'd1_creatinine_max', 'd1_creatinine_min',
            'bun_apache', 'd1_bun_max', 'd1_bun_min',
            'sodium_apache', 'd1_sodium_max', 'd1_sodium_min',
            'd1_potassium_max', 'd1_potassium_min',
            'albumin_apache', 'd1_albumin_max', 'd1_albumin_min',
            'd1_heartrate_max', 'd1_mbp_max',
            'severity_label'
        ]

        # Call predict_clinical
        severity, probs, result_df = predict_clinical(data, training_file_path, clinical_model_path)
        severity_map = {0: "Very Mild", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Very Severe"}

        input_columns = list(data.keys())

        # Create results table for Discord
        table = "**Clinical Prediction Result**\n"
        table += f"🔹 **Severity**: **{severity_map.get(severity, 'Unknown')}** ({severity})\n"
        table += f"🔍 **Probability**: {np.max(probs):.4f}\n"
        table += "\n**Detailed Data**:\n"
        table += "| Column | Value | Source | Description |\n"
        table += "|---|---|---|---|\n"
        for col in usecols:
            if col not in result_df.columns:
                logger.warning(f"Column {col} not in result_df")
                continue
            value = result_df[col].iloc[0]
            source = 'Manual' if col in input_columns else 'Predicted'
            description = result_df.get('description', pd.Series('No description')).iloc[0]
            table += f"| {col} | {value} | {source} | {description} |\n"

        # Send in chunks
        for i in range(0, len(table), 1900):
            await user.send(table[i:i+1900])

    except Exception as e:
        logger.error(f"Error in clinical_interactive: {str(e)}")
        await user.send(f"❌ An error occurred: {str(e)}")


@bot.command()
async def image(ctx):
    if determine_role(ctx.channel.name) != 'doctor':
        await ctx.send("⚠️ Only doctors are allowed to use this command!")
        return
    if not ctx.message.attachments:
        await ctx.send("⚠️ Please attach a medical image or provide an image URL!")
        return
    await ctx.send("✅ Processing the medical image...")

    meta_model, meta_classification_model, feature_model = load_meta_model(config_filepath, weights_filepath)
    attachment = ctx.message.attachments[0]
    image_path = attachment.url

    try:
        results = predict_and_gradcam_from_path(image_path, meta_model, meta_classification_model, feature_model, model_configs)
        if not results:
            await ctx.send("❌ Unable to process the image. Please check the format or the image URL.")
            return

        severity_map = {0: "Very Mild", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Very Severe"}

        for result in results:
            response = (
                f"🎯 **Medical Image Analysis Result**\n"
                # f"🔹 Image ID: {result['image_id']}\n"
                f"🔹 Severity Level: **{result['pred_class']}** ({severity_map.get(result['pred_class'], 'Unknown')})\n"
                # f"🔹 Probability: {result['prob']:.4f}\n"
                f"🔹 Model: {result['model_name']}\n"
            )
            files = []
            for path_key in ['processed_image_path', 'heatmap_path', 'gradcam_path']:
                if os.path.exists(result[path_key]):
                    files.append(discord.File(result[path_key]))
                else:
                    response += f"⚠️ File not found: {path_key}\n"
            await ctx.send(response, files=files)

        output_filepath = os.path.join(OUTPUT_DIR, f"image_results_{ctx.author.id}_{int(time.time())}.json")
        save_results(results, output_filepath)

    except Exception as e:
        await ctx.send(f"❌ Error processing image: {str(e)}")
        logger.error(f"Image processing error: {e}")


@tasks.loop(minutes=1)
async def check_schedules():
    try:
        patients_df = pd.read_csv(LOCAL_PATIENTS_CSV)
        current_time = datetime.datetime.now()
        for _, patient in patients_df.iterrows():
            medication_schedule = json.loads(patient['medication_schedule'])
            patient_name = patient['name']
            id_doctor = patient['id_doctor']
            discord_id = patient['phone']
            followup_date = datetime.datetime.strptime(patient['followup_date'], '%Y-%m-%d %H:%M')
            user = None
            try:
                user = await bot.fetch_user(int(discord_id))
            except (ValueError, discord.errors.NotFound):
                logger.warning(f"Discord ID {discord_id} not found for patient {patient_name}")
                continue
            if user:
                # Follow-up reminder
                if current_time.date() == followup_date.date() and current_time.hour == followup_date.hour:
                    message = (
                        f"🔔 **Follow-up Reminder for {patient_name}**\n"
                        f"📅 **Scheduled Time**: {patient['followup_date']}\n"
                        f"**Assigned Doctor Information**\n{get_doctor_info(id_doctor)}"
                    )
                    await user.send(message)
                    logger.info(f"Follow-up reminder sent to {patient_name}")

                # Medication reminders
                for time_slot, times in medication_schedule.items():
                    for time_str in times:
                        try:
                            schedule_time = datetime.datetime.strptime(time_str, '%H:%M').replace(
                                year=current_time.year, month=current_time.month, day=current_time.day
                            )
                            time_diff = (current_time - schedule_time).total_seconds() / 60
                            if -5 <= time_diff <= 5:  # Within 5-minute window
                                message = format_medication_schedule(medication_schedule, patient_name, id_doctor)
                                await user.send(message)
                                logger.info(f"Medication reminder sent to {patient_name} at {time_str}")
                        except ValueError:
                            logger.error(f"Invalid time format: {time_str} for patient {patient_name}")
    except Exception as e:
        logger.error(f"Error while checking schedules: {str(e)}")


@tasks.loop(seconds=10)
async def check_api_data():
    try:
        response = requests.get('http://localhost:5000/api/data', timeout=5)
        logger.info(f"Request to /api/data, status code: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Request to /api/data failed with status code: {response.status_code}")
            return

        try:
            data = response.json()
            logger.info(f"Response from /api/data: {data}")
        except ValueError as e:
            logger.error(f"Invalid JSON response from /api/data: {str(e)}")
            return

        if 'status' in data and data['status'] == 'empty':
            return

        if 'type' not in data or 'data' not in data:
            logger.error(f"API data missing 'type' or 'data' fields: {data}")
            return

        data_type = data['type']
        payload = data['data']

        # Find doctor-related channel
        channel = None
        for guild in bot.guilds:
            for ch in guild.text_channels:
                if 'doctor' in ch.name.lower() or 'bác sĩ' in ch.name.lower():
                    channel = ch
                    break
            if channel:
                break

        if not channel:
            logger.error("No doctor channel found to send API data")
            return

        if data_type == 'patient':
            try:
                if not isinstance(payload['prescription'], list) or not payload['prescription']:
                    await channel.send("⚠️ Prescription must be a non-empty list")
                    return
                for med in payload['prescription']:
                    if not all(key in med for key in ['name', 'used_tablets']):
                        await channel.send("⚠️ Each medication must contain 'name' and 'used_tablets'")
                        return

                patients_df = pd.read_csv(LOCAL_PATIENTS_CSV)
                if payload['patient_id'] in patients_df['patient_id'].values:
                    await channel.send(f"⚠️ Patient ID {payload['patient_id']} already exists")
                    return

                doctors_df = pd.read_csv(LOCAL_DOCTORS_CSV)
                if payload['id_doctor'] not in doctors_df['doctor_id'].values:
                    await channel.send(f"⚠️ Doctor ID {payload['id_doctor']} does not exist")
                    return

                followup_date = payload.get('followup_date', schedule_followup(payload['condition_label']))
                try:
                    datetime.datetime.strptime(followup_date, '%Y-%m-%d %H:%M')
                except ValueError:
                    await channel.send("⚠️ Invalid follow-up date, using default date")
                    followup_date = schedule_followup(payload['condition_label'])

                new_patient = {
                    "patient_id": payload['patient_id'], "name": payload['name'], "gender": payload['gender'],
                    "age": payload['age'], "address": payload['address'], "phone": payload['phone'],
                    "clinical_data": json.dumps({}), "condition_label": payload['condition_label'],
                    "id_doctor": payload['id_doctor'], "prescription": json.dumps(payload['prescription']),
                    "medication_schedule": json.dumps(payload['medication_schedule']),
                    "followup_date": followup_date
                }

                patients_df = pd.concat([patients_df, pd.DataFrame([new_patient])], ignore_index=True)
                save_to_csv(patients_df, LOCAL_PATIENTS_CSV)
                template_path = save_json_template(payload, "patient", "web")

                response = (
                    f"✅ Added patient {payload['patient_id']} from web.\n"
                    f"📅 **Follow-up**: {followup_date}\n"
                    f"**Prescription**: {json.dumps(payload['prescription'], indent=2)}\n"
                    f"**Doctor Info**\n{get_doctor_info(payload['id_doctor'])}"
                )
                if template_path:
                    response += f"\n💾 Template saved at: `{template_path}`"

                await channel.send(response)
                logger.info(f"Processed patient {payload['patient_id']} from API")

            except Exception as e:
                await channel.send(f"❌ Error adding patient from web: {str(e)}")
                logger.error(f"Error processing patient data from API: {str(e)}")

        elif data_type == 'clinical':
            try:
                errors = validate_clinical_data(payload)
                if errors:
                    await channel.send(f"⚠️ Invalid clinical data:\n" + "\n".join(errors))
                    return

                severity, probs, extended_df = predict_clinical(payload, training_file_path, clinical_model_path)
                severity_map = {0: "Very Mild", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Critical"}

                response = (
                    f"🎯 **Clinical Prediction from Web**\n"
                    f"🔹 Severity Level: **{severity}** ({severity_map.get(severity, 'Unknown')})\n"
                    f"🔹 Probability: {np.max(probs):.4f}\n"
                )

                if not extended_df.empty:
                    extended_df['predicted_severity'] = severity
                    table = "**Additional Data**\n| Metric | Value |\n|---|---|\n"
                    for col, val in extended_df.iloc[0].items():
                        if pd.isna(val):
                            continue
                        val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                        table += f"| {col} | {val_str} |\n"
                    response += f"\n{table}"

                    output_path = os.path.join(OUTPUT_DIR, f"clinical_web_{int(time.time())}.csv")
                    extended_df.to_csv(output_path, index=False)
                    shutil.copy(output_path, os.path.join(DATA_DIR, os.path.basename(output_path)))
                    response += f"\n💾 Data saved at: `{output_path}`"

                template_path = save_json_template(payload, "clinical", "web")
                if template_path:
                    response += f"\n💾 Template saved at: `{template_path}`"

                await channel.send(response)
                logger.info("Processed clinical data from API")

            except Exception as e:
                await channel.send(f"❌ Error processing clinical data from web: {str(e)}")
                logger.error(f"Error processing clinical data from API: {str(e)}")

    except Exception as e:
        logger.error(f"Error checking API data: {str(e)}")


# Function to generate follow-up schedule
def schedule_followup(severity):
    followup_days = {0: 30, 1: 14, 2: 7, 3: 3, 4: 1}
    days = followup_days.get(severity, 7)
    followup_date = datetime.datetime.now() + datetime.timedelta(days=days)
    return followup_date.strftime('%Y-%m-%d %H:%M')

# Function to get doctor information
def get_doctor_info(id_doctor):
    doctors_df = pd.read_csv(LOCAL_DOCTORS_CSV)
    doctor = doctors_df[doctors_df['doctor_id'] == id_doctor]
    if doctor.empty:
        return "Doctor information not found."
    doctor = doctor.iloc[0]
    return (
        f"- Name: {doctor['name']}\n"
        f"- Specialty: {doctor['specialty']}\n"
        f"- Contact: {doctor['contact']}\n"
        f"- Hospital: {doctor['hospital']}"
    )

# Function to format medication schedule
def format_medication_schedule(medication_schedule, patient_name, id_doctor):
    if not medication_schedule:
        return "No medication schedule available."

    message = f"🔔 **Medication Reminder for {patient_name}**\n"
    times = ['morning', 'noon', 'evening']

    patients_df = pd.read_csv(LOCAL_PATIENTS_CSV)
    patient = patients_df[patients_df['name'] == patient_name]

    if not patient.empty:
        prescription = json.loads(patient['prescription'].iloc[0])
        message += "**Prescribed Medications**:\n"
        for med in prescription:
            message += f"- {med['name']}: {med['used_tablets']}\n"

    for time_slot in times:
        if time_slot in medication_schedule and medication_schedule[time_slot]:
            times_list = medication_schedule[time_slot]
            if isinstance(times_list, str):
                times_list = json.loads(times_list)
            message += f"⏰ **{time_slot.capitalize()}**: {', '.join(times_list)}\n"

    message += f"\n**Assigned Doctor Information**\n{get_doctor_info(id_doctor)}"
    return message

# Function to save results to file
def save_results(all_results, output_filepath):
    try:
        with open(output_filepath, 'w') as f:
            json.dump(all_results, f, indent=4)
        shutil.copy(output_filepath, os.path.join(DATA_DIR, os.path.basename(output_filepath)))
        logger.info(f"Results saved at: {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

# Function to determine role based on channel name
def determine_role(channel_name):
    channel_name = channel_name.lower()
    if 'patient' in channel_name or 'bệnh nhân' in channel_name:
        return 'patient'
    elif 'doctor' in channel_name or 'bác sĩ' in channel_name:
        return 'doctor'
    return None

# Flask app setup
from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# Declare favicon route before running the app
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

# Create necessary directories
for directory in [LOCAL_DATA_DIR, DATA_DIR, GRAD_CAM_DIR, FEATURE_SAVE_DIR, META_SAVE_DIR, OUTPUT_DIR, GRAD_CAM_SAVE_DIR, TEMPLATES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Run the bot
if __name__ == "__main__":
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")
