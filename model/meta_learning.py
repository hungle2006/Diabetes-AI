import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Layer, Dropout
from tensorflow.keras.applications import EfficientNetB0, Xception, InceptionV3, ResNet50, DenseNet121
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet121_preprocess
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, confusion_matrix, precision_score, accuracy_score
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import gc
from google.colab import drive
import time
import subprocess
import tensorflow_probability as tfp
from albumentations.core.composition import OneOf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("Enabled GPU memory growth")
    except RuntimeError as e:
        logging.error(f"Error setting GPU memory growth: {e}")
else:
    logging.warning("No GPU found, running on CPU may be slow.")

# Mount Google Drive
drive.mount('/content/drive')

# Set up directories
feature_save_dir = "/content/drive/MyDrive/working"
log_dir = os.path.join(feature_save_dir, "logs")
meta_save_dir = os.path.join(feature_save_dir, "meta_features")
gradcam_save_dir = os.path.join(feature_save_dir, "gradcam_meta")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(feature_save_dir, exist_ok=True)
os.makedirs(meta_save_dir, exist_ok=True)
os.makedirs(gradcam_save_dir, exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Data directories
drive_folder = "/content/drive/MyDrive/kaggle_data/aptos2019"
processed_folder = "/content/processed_train_images"
TEMP_AUGMENT_DIR = "/tmp/temp_augmented_images"

# Paths to ZIP file and processed image directory
zip_path = "/content/drive/MyDrive/kaggle_data/aptos2019/train_images.zip"
extract_dir = "/content/aptos_extracted"
processed_folder = "/content/processed_train_images"

os.makedirs(extract_dir, exist_ok=True)
os.makedirs(TEMP_AUGMENT_DIR, exist_ok=True)
SIZE = 224
NUM_CLASSES = 5

# Create df_train_processed
df_train_processed = df_train[df_train['id_code'].isin(processed_ids)].copy()
if df_train_processed.empty:
    logging.error("df_train_processed is empty. Check processed_ids or df_train['id_code'].")
    raise ValueError("No valid data to process.")
logging.info(f"df_train_processed shape: {df_train_processed.shape}")

# 6. Update DataFrame with processed images
df_train_processed['image_path'] = [os.path.join(processed_folder, f"{id_code}") for id_code in df_train_processed['id_code']]

# 7. Split data into train, validation, test
x = df_train_processed['image_path']
y = df_train_processed['diagnosis']

# Shuffle data
x, y = shuffle(x, y, random_state=42)

# Split into train+validation and test (80% - 20%)
x_temp, test_x, y_temp, test_y = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)

# Split into train and validation (85% train, 15% val in the 80% initial data)
train_x, valid_x, train_y, valid_y = train_test_split(x_temp, y_temp, test_size=0.15/0.80, stratify=y_temp, random_state=42)

# Print verification info
print("Train X size:", len(train_x))
print("Train y size:", len(train_y))
print("Valid X size:", len(valid_x))
print("Valid y size:", len(valid_y))
print("Test X size:", len(test_x))

# Convert labels to one-hot
train_y_multi = tf.keras.utils.to_categorical(train_y, num_classes=NUM_CLASSES)
valid_y_multi = tf.keras.utils.to_categorical(valid_y, num_classes=NUM_CLASSES)
test_y_multi = tf.keras.utils.to_categorical(test_y, num_classes=NUM_CLASSES)

# Function to load original image from extracted directory
def load_original_image(image_id, extract_dir):
    """
    Load original image from the extracted directory based on image_id.

    Args:
        image_id (str): Image ID (excluding file extension).
        extract_dir (str): Directory containing images extracted from ZIP.

    Returns:
        numpy.ndarray: RGB image or None if not found/error.
    """
    try:
        image_id = image_id.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().startswith(image_id.lower()) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        logging.error(f"Could not read image at: {img_path}")
                        return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return img
        logging.warning(f"Image not found for ID: {image_id} in {extract_dir}")
        return None
    except Exception as e:
        logging.error(f"Error loading original image {image_id}: {str(e)}")
        return None

# Load processed image
def load_processed_image(image_id, processed_folder, size=224):
    try:
        img_path = os.path.join(processed_folder, f"{image_id}.png")
        if not os.path.exists(img_path):
            logging.error(f"Image does not exist: {img_path}")
            return None
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Could not read image: {img_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        logging.error(f"Error loading image {image_id}: {str(e)}")
        return None

resized_train_x = np.array([img for img in [load_processed_image(id_code, processed_folder, size=SIZE) for id_code in train_x] if img is not None])
resized_valid_x = np.array([img for img in [load_processed_image(id_code, processed_folder, size=SIZE) for id_code in valid_x] if img is not None])
resized_test_x = np.array([img for img in [load_processed_image(id_code, processed_folder, size=SIZE) for id_code in test_x] if img is not None])

# Random Erasing function
def custom_random_erasing(image, scale=(0.01, 0.05), ratio=(0.5, 2.0), p=0.3):
    if np.random.random() > p:
        return image
    height, width, channels = image.shape
    area = height * width
    scale_factor = np.random.uniform(scale[0], scale[1])
    erase_area = area * scale_factor
    aspect_ratio = np.random.uniform(ratio[0], ratio[1])
    erase_height = int(np.sqrt(erase_area / aspect_ratio))
    erase_width = int(np.sqrt(erase_area * aspect_ratio))
    erase_height = min(erase_height, height)
    erase_width = min(erase_width, width)
    if erase_height < 1 or erase_width < 1:
        return image
    x = np.random.randint(0, width - erase_width + 1)
    y = np.random.randint(0, height - erase_height + 1)
    output = image.copy()
    value = np.mean(image, axis=(0, 1))
    output[y:y+erase_height, x:x+erase_width, :] = value
    return output

# Balance and augment data
def balance_and_augment_data(images, labels, target_classes=[0, 1, 2, 3, 4], samples_per_class=None):
    num_classes = labels.shape[1]
    label_indices = np.argmax(labels, axis=1)
    keep_indices = np.isin(label_indices, target_classes)
    filtered_images = images[keep_indices]
    filtered_labels = labels[keep_indices]
    filtered_label_indices = label_indices[keep_indices]

    # Check for NaN/inf in filtered_labels
    if np.any(np.isnan(filtered_labels)):
        logging.error("filtered_labels contain NaN values.")
        raise ValueError("filtered_labels contain NaN values.")
    if np.any(np.isinf(filtered_labels)):
        logging.error("filtered_labels contain inf values.")
        raise ValueError("filtered_labels contain inf values.")

    class_counts = np.bincount(filtered_label_indices, minlength=num_classes)
    print(f"Initial distribution: {dict(zip(range(num_classes), class_counts))}")

    max_count = samples_per_class or max(class_counts)
    print(f"Target samples per class: {max_count}")

    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(p=0.2),
        A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=0.2),
    ])

    new_images = []
    new_labels = []
    for cls in target_classes:  # Include class 0
        cls_indices = np.where(filtered_label_indices == cls)[0]
        cls_images = filtered_images[cls_indices]
        cls_labels = filtered_labels[cls_indices]
        current_count = len(cls_indices)
        new_images.extend(cls_images)
        new_labels.extend(cls_labels)
        print(f"Class {cls}: {current_count} initial samples")
        augment_count = max_count - current_count
        if augment_count > 0:
            print(f"Augmenting {augment_count} samples for class {cls}")
            for _ in range(augment_count):
                idx = np.random.choice(cls_indices)
                img = filtered_images[idx].astype(np.uint8)
                aug_img = augmenter(image=img)['image']
                aug_img = custom_random_erasing(aug_img, scale=(0.01, 0.05), ratio=(0.5, 2.0), p=0.3)
                new_images.append(aug_img)
                new_labels.append(filtered_labels[idx])

    new_images = np.array(new_images, dtype=np.float32)
    new_labels = np.array(new_labels, dtype=np.float32)
    new_images, new_labels = shuffle(new_images, new_labels, random_state=42)
    final_class_counts = np.bincount(np.argmax(new_labels, axis=1), minlength=num_classes)
    print(f"Label distribution after balancing: {dict(zip(range(num_classes), final_class_counts))}")

    # Check for NaN/inf in new_labels
    if np.any(np.isnan(new_labels)):
        logging.error("new_labels contain NaN values after augmentation.")
        raise ValueError("new_labels contain NaN values after augmentation.")
    if np.any(np.isinf(new_labels)):
        logging.error("new_labels contain inf values after augmentation.")
        raise ValueError("new_labels contain inf values after augmentation.")

    return new_images, new_labels

# Balance train data
class_counts = np.bincount(train_y)
class_0_count = class_counts[0]
print(f"Number of class 0 samples: {class_0_count}")
balanced_train_x, balanced_train_y_multi = balance_and_augment_data(
    resized_train_x, train_y_multi, target_classes=[1, 2, 3, 4], samples_per_class=class_0_count
)
class_0_indices = np.where(np.argmax(train_y_multi, axis=1) == 0)[0]
class_0_images = resized_train_x[class_0_indices]
class_0_labels = train_y_multi[class_0_indices]

# Check for NaN/inf in class_0_labels
if np.any(np.isnan(class_0_labels)):
    logging.error("class_0_labels contain NaN values.")
    raise ValueError("class_0_labels contain NaN values.")
if np.any(np.isinf(class_0_labels)):
    logging.error("class_0_labels contain inf values.")
    raise ValueError("class_0_labels contain inf values.")

balanced_train_x = np.concatenate([balanced_train_x, class_0_images], axis=0)
balanced_train_y_multi = np.concatenate([balanced_train_y_multi, class_0_labels], axis=0)
balanced_train_x, balanced_train_y_multi = shuffle(balanced_train_x, balanced_train_y_multi, random_state=42)
final_class_counts = np.bincount(np.argmax(balanced_train_y_multi, axis=1), minlength=5)
print(f"Label distribution after adding class 0: {dict(zip(range(5), final_class_counts))}")
print("balanced_train_x shape:", balanced_train_x.shape)
print("balanced_train_y_multi shape:", balanced_train_y_multi.shape)

# Check for NaN/inf after concatenation
if np.any(np.isnan(balanced_train_y_multi)):
    logging.error("balanced_train_y_multi contains NaN values after concatenation.")
    raise ValueError("balanced_train_y_multi contains NaN values after concatenation.")
if np.any(np.isinf(balanced_train_y_multi)):
    logging.error("balanced_train_y_multi contains inf values after concatenation.")
    raise ValueError("balanced_train_y_multi contains inf values after concatenation.")

# Check label distribution
if np.any(final_class_counts == 0):
    logging.error("One or more classes have no samples after balancing. Check balance_and_augment_data.")
    raise ValueError("One or more classes have no samples after balancing.")

# After balancing data
if len(train_x) != len(resized_train_x):
    logging.error(f"train_x ({len(train_x)}) does not match resized_train_x ({len(resized_train_x)}).")
    raise ValueError("train_x and resized_train_x are not synchronized.")

# Update train_x to synchronize with balanced_train_x
train_x_balanced = []
for i in range(len(balanced_train_x)):
    # Assume samples maintain order; otherwise, need to remap IDs
    train_x_balanced.append(f"sample_{i}")  # Replace with actual IDs if available
train_x = train_x_balanced

# Check synchronization
if len(train_x) != len(balanced_train_y_multi):
    logging.error(f"train_x ({len(train_x)}) does not match balanced_train_y_multi ({len(balanced_train_y_multi)}).")
    raise ValueError("train_x and balanced_train_y_multi are not synchronized.")

# In main section
train_x = train_x.tolist() if isinstance(train_x, (pd.Series, pd.DataFrame)) else train_x
valid_x = valid_x.tolist() if isinstance(valid_x, (pd.Series, pd.DataFrame)) else valid_x
test_x = test_x.tolist() if isinstance(test_x, (pd.Series, pd.DataFrame)) else test_x

# Check lengths
logging.info(f"train_x length: {len(train_x)}, balanced_train_y_multi shape: {balanced_train_y_multi.shape}")
logging.info(f"valid_x length: {len(valid_x)}, valid_y_multi shape: {valid_y_multi.shape}")
logging.info(f"test_x length: {len(test_x)}, test_y_multi shape: {test_y_multi.shape}")

if len(train_x) != len(balanced_train_y_multi):
    logging.error(f"train_x length mismatch: {len(train_x)}, expected: {len(balanced_train_y_multi)}.")
    raise ValueError("train_x length does not match balanced_train_y_multi.")
if len(valid_x) != len(valid_y_multi):
    logging.error(f"valid_x length mismatch: {len(valid_x)}, expected: {len(valid_y_multi)}.")
    raise ValueError("valid_x length does not match valid_y_multi.")
if len(test_x) != len(test_y_multi):
    logging.error(f"test_x length mismatch: {len(test_x)}, expected: {len(test_y_multi)}.")
    raise ValueError("test_x length does not match test_y_multi.")

# Balance validation data
class_counts_valid = np.bincount(valid_y)
class_0_count_valid = class_counts_valid[0]
print(f"Number of class 0 samples in valid: {class_0_count_valid}")

balanced_valid_x, balanced_valid_y_multi = balance_and_augment_data(
    resized_valid_x, valid_y_multi, target_classes=[1, 2, 3, 4], samples_per_class=class_0_count_valid
)

# Add back class 0
class_0_indices_valid = np.where(np.argmax(valid_y_multi, axis=1) == 0)[0]
class_0_images_valid = resized_valid_x[class_0_indices_valid]
class_0_labels_valid = valid_y_multi[class_0_indices_valid]

# Check for NaN/inf in class_0_labels_valid
if np.any(np.isnan(class_0_labels_valid)):
    logging.error("class_0_labels_valid contains NaN.")
    raise ValueError("class_0_labels_valid contains NaN.")
if np.any(np.isinf(class_0_labels_valid)):
    logging.error("class_0_labels_valid contains inf.")
    raise ValueError("class_0_labels_valid contains inf.")

balanced_valid_x = np.concatenate([balanced_valid_x, class_0_images_valid], axis=0)
balanced_valid_y_multi = np.concatenate([balanced_valid_y_multi, class_0_labels_valid], axis=0)
balanced_valid_x, balanced_valid_y_multi = shuffle(balanced_valid_x, balanced_valid_y_multi, random_state=42)
final_class_counts_valid = np.bincount(np.argmax(balanced_valid_y_multi, axis=1), minlength=NUM_CLASSES)
print(f"Validation label distribution after balancing: {dict(zip(range(NUM_CLASSES), final_class_counts_valid))}")
print("balanced_valid_x shape:", balanced_valid_x.shape)
print("balanced_valid_y_multi shape:", balanced_valid_y_multi.shape)

# Check for NaN/inf after concatenation
if np.any(np.isnan(balanced_valid_y_multi)):
    logging.error("balanced_valid_y_multi contains NaN after concatenation.")
    raise ValueError("balanced_valid_y_multi contains NaN after concatenation.")
if np.any(np.isinf(balanced_valid_y_multi)):
    logging.error("balanced_valid_y_multi contains inf after concatenation.")
    raise ValueError("balanced_valid_y_multi contains inf after concatenation.")

# Update valid_x to synchronize
valid_x_balanced = [f"valid_sample_{i}" for i in range(len(balanced_valid_x))]
valid_x = valid_x_balanced

# Check synchronization
if len(valid_x) != len(balanced_valid_y_multi):
    logging.error(f"valid_x ({len(valid_x)}) does not match balanced_valid_y_multi ({len(balanced_valid_y_multi)}).")
    raise ValueError("valid_x and balanced_valid_y_multi are not synchronized.")

# Define model_configs
model_configs = {
    "efficientnetb0": {
        "model_type": "efficientnetb0",
        "config_path": "/content/drive/MyDrive/working/EfficientNetB0_bestqwk_aptos/config.json",
        "weights_path": "/content/drive/MyDrive/working/EfficientNetB0_bestqwk_aptos/model.weights.h5",
        "preprocess": efficientnet_preprocess,
        "img_size": 224,
        "base_model": EfficientNetB0,
        "feature_layer_name": "top_conv"  # Layer for 4D features
    },
    "xception": {
        "model_type": "xception",
        "config_path": "/content/drive/MyDrive/working/Xception_bestqwk_aptos/config.json",
        "weights_path": "/content/drive/MyDrive/working/Xception_bestqwk_aptos/model.weights.h5",
        "preprocess": xception_preprocess,
        "img_size": 299,
        "base_model": Xception,
        "feature_layer_name": "block14_sepconv2_act"
    },
    "inceptionv3": {
        "model_type": "inceptionv3",
        "config_path": "/content/drive/MyDrive/working/InceptionV3_bestqwk_aptos/config.json",
        "weights_path": "/content/drive/MyDrive/working/InceptionV3_bestqwk_aptos/model.weights.h5",
        "preprocess": inceptionv3_preprocess,
        "img_size": 299,
        "base_model": InceptionV3,
        "feature_layer_name": "mixed10"
    },
    "resnet50": {
        "model_type": "resnet50",
        "config_path": "/content/drive/MyDrive/working/ResNet50_bestqwk_aptos/config.json",
        "weights_path": "/content/drive/MyDrive/working/ResNet50_bestqwk_aptos/model.weights.h5",
        "preprocess": resnet50_preprocess,
        "img_size": 224,
        "base_model": ResNet50,
        "feature_layer_name": "conv5_block3_out"
    },
    "densenet121": {
        "model_type": "densenet121",
        "config_path": "/content/drive/MyDrive/working/DenseNet121_bestqwk_aptos/config.json",
        "weights_path": "/content/drive/MyDrive/working/DenseNet121_bestqwk_aptos/model.weights.h5",
        "preprocess": densenet121_preprocess,
        "img_size": 224,
        "base_model": DenseNet121,
        "feature_layer_name": "conv5_block16_concat"
    }
}

# My_Generator with mixup
class My_Generator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size, is_train=False, mix=True, augment=False, size1=224, size2=299, model_type="default", preprocess=None):
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        self.is_mix = mix
        self.model_type = str(model_type).lower()
        self.preprocess = preprocess
        self.temp_augment_dir = TEMP_AUGMENT_DIR
        os.makedirs(self.temp_augment_dir, exist_ok=True)
        self.target_size = (size2, size2) if 'inceptionv3' in self.model_type or 'xception' in self.model_type else (size1, size1)
        self.image_paths = []
        if isinstance(images, np.ndarray):
            for i, img in enumerate(images):
                img_path = os.path.join(self.temp_augment_dir, f"img_{i}_{np.random.randint(1000000)}.png")
                try:
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    self.image_paths.append(img_path)
                except Exception as e:
                    logging.error(f"Error saving image {img_path}: {str(e)}")
                    continue
        else:
            self.image_paths = list(images)
        unique_paths = []
        unique_indices = []
        seen = set()
        for i, path in enumerate(self.image_paths):
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
                unique_indices.append(i)
        self.image_paths = unique_paths
        self.labels = self.labels[unique_indices]
        if len(self.image_paths) != len(self.labels):
            logging.error(f"Number of image_paths ({len(self.image_paths)}) does not match number of labels ({len(self.labels)})")
            self.labels = self.labels[:len(self.image_paths)]
        if not self.image_paths:
            raise ValueError("No valid image_paths were created.")
        print(f"Initialized My_Generator: {len(self.image_paths)} samples, target_size={self.target_size}, is_train={is_train}")
        self.dataset = self._create_dataset()

    def _load_image(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, self.target_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.ensure_shape(img, [self.target_size[0], self.target_size[1], 3])
        img = tf.cast(img, tf.float32) / 255.0
        if self.preprocess is not None:
            img = self.preprocess(img)
        return img

    def _mixup(self, images, labels):
        batch_size = tf.shape(images)[0]
        lam = tf.random.uniform([], minval=0.2, maxval=0.4, dtype=tf.float32)
        indices = tf.random.shuffle(tf.range(batch_size, dtype=tf.int32))
        mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
        mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
        return mixed_images, mixed_labels

    def _create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))
        dataset = dataset.map(
            lambda img_path, label: (self._load_image(img_path), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if self.is_train:
            dataset = dataset.shuffle(buffer_size=len(self.image_paths))
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        if self.is_train and self.is_mix:
            dataset = dataset.map(
                self._mixup,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __iter__(self):
        self.iterator = iter(self.dataset)
        return self

    def __next__(self):
        batch = next(self.iterator)
        images, labels = batch
        # Ensure images is a 4D tensor: (batch_size, height, width, channels)
        if len(images.shape) != 4:
            logging.warning(f"Batch images have unexpected shape: {images.shape}, reshaping to 4D")
            images = tf.reshape(images, [-1, self.target_size[0], self.target_size[1], 3])
        # Ensure labels is a 2D tensor: (batch_size, num_classes)
        if len(labels.shape) != 2:
            logging.warning(f"Batch labels have unexpected shape: {labels.shape}, reshaping to 2D")
            labels = tf.reshape(labels, [-1, labels.shape[-1]])
        return images, labels

# Callback to calculate class weights
class ConfusionMatrixWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_features, valid_labels, classification_model, num_classes=5, class_counts=None):
        super().__init__()
        self.valid_features = tf.convert_to_tensor(valid_features, dtype=tf.float32)
        self.valid_labels = tf.convert_to_tensor(valid_labels, dtype=tf.float32)
        self.classification_model = classification_model
        self.num_classes = num_classes
        self.prev_cm = None
        self.class_weights = tf.ones(num_classes, dtype=tf.float32)
        self.class_counts = tf.cast(class_counts, dtype=tf.float32) if class_counts is not None else None
        self.history_dir = os.path.join(feature_save_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.weights_history = []
        logging.info(f"ConfusionMatrixWeightCallback init: valid_features shape={self.valid_features.shape}, valid_labels shape={self.valid_labels.shape}")

    def on_epoch_end(self, epoch, logs=None):
        # Predict on validation set
        y_pred = self.classification_model.predict(self.valid_features, verbose=0, batch_size=32)
        y_true = tf.argmax(self.valid_labels, axis=1, output_type=tf.int32)
        y_pred_classes = tf.argmax(y_pred, axis=1, output_type=tf.int32)

        # Calculate confusion matrix using TensorFlow
        cm = tf.math.confusion_matrix(
            y_true, y_pred_classes, num_classes=self.num_classes, dtype=tf.float32
        )
        print(f"Epoch {epoch+1} - Confusion Matrix:\n{cm.numpy()}")

        # Calculate errors and error rates
        eye = tf.eye(self.num_classes, dtype=tf.float32)
        errors = tf.reduce_sum(cm * (1 - eye), axis=1)
        total_samples_per_class = tf.reduce_sum(cm, axis=1)
        total_samples_per_class = tf.where(
            tf.equal(total_samples_per_class, 0), 1.0, total_samples_per_class
        )
        error_rates = errors / total_samples_per_class

        # Identify weak classes
        weak_classes = []
        if self.class_counts is not None:
            max_count = tf.reduce_max(self.class_counts)
            min_count = tf.reduce_min(tf.where(tf.math.is_finite(self.class_counts), self.class_counts, max_count))

            weak_classes = tf.where(self.class_counts <= min_count * 1.5)
            weak_classes = tf.cast(weak_classes, tf.int32)
        high_error_classes = tf.where(error_rates >= tfp.stats.percentile(error_rates, 75))
        high_error_classes = tf.cast(high_error_classes, tf.int32)
        weak_classes = tf.concat([
            tf.reshape(weak_classes, [-1]),
            tf.reshape(high_error_classes, [-1])
        ], axis=0)
        weak_classes = tf.unique(weak_classes)[0]
        weak_classes = tf.sort(weak_classes)
        print(f"Epoch {epoch+1} - Weak classes: {weak_classes.numpy()}")

        # Update class weights
        self.class_weights = 1.0 + error_rates
        for cls in weak_classes:
            self.class_weights = tf.tensor_scatter_nd_update(
                self.class_weights, [[cls]], [self.class_weights[cls] * 2.0]
            )
        self.class_weights = self.class_weights / tf.reduce_max(self.class_weights)
        print(f"Epoch {epoch+1} - Class weights: {self.class_weights.numpy()}")

        # Save weights history
        weights_history_entry = {
            "epoch": epoch + 1,
            "class_weights": self.class_weights.numpy().tolist(),
            "weak_classes": weak_classes.numpy().tolist(),
            "confusion_matrix": cm.numpy().tolist()
        }
        self.weights_history.append(weights_history_entry)
        weights_path = os.path.join(self.history_dir, f"class_weights_epoch_{epoch+1}.json")
        with open(weights_path, 'w') as f:
            json.dump(weights_history_entry, f, indent=4)
        print(f"Saved class weights at: {weights_path}")

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm.numpy(), annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=list(range(self.num_classes)),
                    yticklabels=list(range(self.num_classes)))
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(feature_save_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix at: {cm_path}")
        self.prev_cm = cm

    def get_class_weights(self):
        return self.class_weights

# Custom layers
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, inputs):
        return tf.keras.backend.stop_gradient(inputs) * (-self.lambda_)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'lambda_': self.lambda_})
        return config

class MemoryAugmentedLayer(tf.keras.layers.Layer):
    def __init__(self, memory_size, memory_dim, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.memory_dim = memory_dim

    def build(self, input_shape):
        # Ensure input_shape is defined
        if len(input_shape) != 2 or input_shape[-1] is None:
            raise ValueError(f"Expected 2D input with defined feature dimension, got {input_shape}")
        self.memory = self.add_weight(
            shape=(self.memory_size, self.memory_dim),
            initializer='zeros',
            trainable=False,
            dtype=tf.float32,
            name='memory'
        )
        super().build(input_shape)

    def call(self, inputs):
        # Ensure inputs is 2D
        if len(inputs.shape) != 2:
            logging.warning(f"MemoryAugmentedLayer inputs have unexpected shape: {inputs.shape}, reshaping to 2D")
            inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])

        batch_size = tf.shape(inputs)[0]
        memory_size = tf.shape(self.memory)[0]

        # Slice or tile memory to match batch size
        memory_sliced = tf.cond(
            tf.greater(batch_size, memory_size),
            lambda: tf.tile(self.memory, [(batch_size + memory_size - 1) // memory_size, 1])[:batch_size],
            lambda: self.memory[:batch_size]
        )

        # Combine inputs and memory
        output = tf.reduce_mean([tf.stack([inputs, memory_sliced], axis=0)], axis=0)

        # Ensure output is 2D with defined shape
        output = tf.reshape(output, [batch_size, self.memory_dim])

        return output

    def compute_output_shape(self, input_shape):
        # Define the output shape explicitly
        return (input_shape[0], self.memory_dim)

    def get_config(self):
        config = super().get_config()
        config.update({'memory_size': self.memory_size, 'memory_dim': self.memory_dim})
        return config

class CustomGridDropout(tf.keras.layers.Layer):
    def __init__(self, ratio, holes_number, p, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.holes_number = holes_number
        self.p = p

    def call(self, inputs, training=None):
        if not training:
            return inputs

        batch_size = tf.shape(inputs)[0]
        feature_dim = tf.shape(inputs)[1]

        # Example dropout logic (simplified)
        mask = tf.random.uniform([batch_size, feature_dim]) > self.p
        mask = tf.cast(mask, tf.float32)
        return inputs * mask / (1 - self.p)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio, 'holes_number': self.holes_number, 'p': self.p})
        return config

class CustomIsotonicRegression:
    def __init__(self):
        self.iso_reg = IsotonicRegression()
        self.X_min_ = None
        self.X_max_ = None
    def fit(self, X, y):
        self.X_min_ = np.min(X)
        self.X_max_ = np.max(X)
        self.iso_reg.fit(X, y)
        return self
    def predict(self, X):
        X_clipped = np.clip(X, self.X_min_, self.X_max_)
        return self.iso_reg.predict(X_clipped)

# Function to minimize memory usage
def minimize_memory_usage(array):
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    return array

# Function to extract and save 2D features from model_configs
def load_model_from_config(config_path, weights_path, base_model_class):
    try:
        if config_path and os.path.exists(config_path) and weights_path and os.path.exists(weights_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            model = tf.keras.models.model_from_json(json.dumps(model_config))
            model.load_weights(weights_path)
            return model
        raise FileNotFoundError
    except:
        return base_model_class(weights='imagenet', include_top=False, pooling='avg')

def extract_2d_features(model_name, config, generator, save_dir, sample_ids):
    expected_samples = len(generator.image_paths)
    base_model = load_model_from_config(
        config['config_path'], config['weights_path'], config['base_model']
    )
    feature_layer_2d = base_model.layers[-2].output if len(base_model.layers) > 1 else base_model.output
    feature_extract = Model(inputs=base_model.input, outputs=feature_layer_2d)
    feature_extract.trainable = False
    features_2d = []
    processed_samples = []
    steps = int(np.ceil(expected_samples / generator.batch_size))

    for step, batch_data in enumerate(generator):
        if step >= steps:
            break
        try:
            batch_images, _ = batch_data
            if batch_images.shape[0] == 0:
                logging.warning(f"Batch {step+1} is empty. Skipping.")
                continue
            batch_features_2d = feature_extract(batch_images, training=False)
            features_2d.append(batch_features_2d.numpy().astype(np.float32))
            batch_sample_ids = sample_ids[step * generator.batch_size : (step + 1) * generator.batch_size][:batch_images.shape[0]]
            processed_samples.extend(batch_sample_ids)
        except Exception as e:
            logging.error(f"Error at batch {step+1}: {str(e)}")
            continue

    features_2d = np.concatenate(features_2d, axis=0) if features_2d else np.zeros((expected_samples, 512), dtype=np.float32)
    if features_2d.shape[0] != expected_samples:
        logging.warning(f"Feature shape mismatch: expected {expected_samples}, got {features_2d.shape[0]}")
        features_2d = features_2d[:expected_samples] if features_2d.shape[0] > expected_samples else \
                      np.pad(features_2d, ((0, expected_samples - features_2d.shape[0]), (0, 0)), mode='edge')

    os.makedirs(save_dir, exist_ok=True)
    features_2d_path = os.path.join(save_dir, f"{model_name}_features_2d.npy")
    np.save(features_2d_path, features_2d)
    print(f"Saved 2D features at: {features_2d_path}, shape={features_2d.shape}")

    metadata = {
        "model_name": model_name,
        "features_2d_path": features_2d_path,
        "sample_ids": processed_samples,
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    metadata_path = os.path.join(save_dir, f"{model_name}_features_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata at: {metadata_path}")

    del base_model, feature_extract
    tf.keras.backend.clear_session()
    gc.collect()
    return features_2d

# Load 4D features from .npz file
def load_4d_features(model_name, split='train'):
    npz_path = os.path.join(meta_save_dir, f"{model_name}_{split}_features.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        features_4d = data['features_4d']
        features_4d_reduced = minimize_memory_usage(np.mean(features_4d, axis=(1, 2)))
        print(f"Loaded and reduced 4D features for {model_name} ({split}): shape {features_4d_reduced.shape}")
        return features_4d_reduced
    else:
        logging.error(f"File not found: {npz_path}")
        raise FileNotFoundError(f"File not found: {npz_path}")

# Combine and reduce features
def combine_and_reduce_features(features_2d_dict, features_4d_dict, labels, sample_ids, save_dir, n_components=50):
    logging.info(f"Combining features for {len(features_2d_dict)} models")
    if isinstance(sample_ids, (pd.Series, pd.DataFrame)):
        sample_ids = sample_ids.values.tolist()
    elif not isinstance(sample_ids, list):
        raise ValueError(f"sample_ids must be a list or pandas.Series/DataFrame, got {type(sample_ids)}")

    expected_samples = len(labels)
    logging.info(f"Expected samples: {expected_samples}, sample_ids length: {len(sample_ids)}")
    if len(sample_ids) != expected_samples:
        raise ValueError(f"Length of sample_ids ({len(sample_ids)}) does not match labels ({expected_samples})")

    # Convert labels to tensor
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    # Validate input labels using TensorFlow
    if tf.reduce_any(tf.math.is_nan(labels)):
        logging.error("Input labels contain NaN values.")
        raise ValueError("Input labels contain NaN values.")
    if tf.reduce_any(tf.math.is_inf(labels)):
        logging.error("Input labels contain inf values.")
        raise ValueError("Input labels contain inf values.")
    label_sums = tf.reduce_sum(labels, axis=1)
    if not tf.reduce_all(tf.logical_and(label_sums >= 0.9, label_sums <= 1.1)):
        logging.error("Input labels are not in valid one-hot format.")
        raise ValueError("Input labels are not in valid one-hot format.")
    if tf.reduce_any(labels < 0.0) or tf.reduce_any(labels > 1.0):
        logging.error("Input labels contain values outside [0.0, 1.0].")
        raise ValueError("Input labels contain values outside [0.0, 1.0].")

    # Log initial label distribution
    label_indices = tf.argmax(labels, axis=1, output_type=tf.int32)
    initial_label_counts = tf.math.bincount(label_indices, minlength=NUM_CLASSES)
    logging.info(f"Initial label distribution: {dict(zip(range(NUM_CLASSES), initial_label_counts.numpy()))}")
    if tf.reduce_all(tf.equal(initial_label_counts, 0)):
        raise ValueError("No samples found in labels for any class before processing.")

    # Save input labels for debugging
    os.makedirs(save_dir, exist_ok=True)
    input_labels_path = os.path.join(save_dir, 'input_labels.npy')
    np.save(input_labels_path, labels.numpy())
    logging.info(f"Saved input labels at: {input_labels_path}")

    # Normalize labels
    labels_normalized = tf.where(labels >= 0.5, 1.0, 0.0)
    logging.info("Normalized labels to one-hot.")

    # Validate normalized labels
    if tf.reduce_any(tf.math.is_nan(labels_normalized)):
        logging.error("labels_normalized contains NaN values.")
        raise ValueError("labels_normalized contains NaN values.")
    if tf.reduce_any(tf.math.is_inf(labels_normalized)):
        logging.error("labels_normalized contains inf values.")
        raise ValueError("labels_normalized contains inf values.")
    normalized_label_sums = tf.reduce_sum(labels_normalized, axis=1)
    if not tf.reduce_all(tf.logical_and(normalized_label_sums >= 0.9, normalized_label_sums <= 1.1)):
        logging.error("labels_normalized is not in valid one-hot format.")
        raise ValueError("labels_normalized is not in valid one-hot format.")

    # Save normalized labels for debugging
    normalized_labels_path = os.path.join(save_dir, 'normalized_labels.npy')
    np.save(normalized_labels_path, labels_normalized.numpy())
    logging.info(f"Saved normalized labels at: {normalized_labels_path}")

    # Validate feature shapes
    for model_name in features_2d_dict:
        features_2d = features_2d_dict[model_name]
        features_4d = features_4d_dict[model_name.lower()]
        logging.info(f"Model {model_name}: 2D shape={features_2d.shape}, 4D shape={features_4d.shape}")
        if features_2d.shape[0] != features_4d.shape[0]:
            raise ValueError(f"Mismatch in sample count: 2D ({features_2d.shape[0]}), 4D ({features_4d.shape[0]})")
        if features_2d.shape[0] != expected_samples:
            logging.warning(f"Feature count ({features_2d.shape[0]}) does not match expected samples ({expected_samples})")

    # Compute valid indices
    valid_indices = None
    for model_name in features_2d_dict:
        features_2d = features_2d_dict[model_name]
        features_4d = features_4d_dict[model_name.lower()]
        valid_2d = tf.reduce_all(tf.math.is_finite(features_2d), axis=1)
        valid_4d = tf.reduce_all(tf.math.is_finite(features_4d), axis=1)
        valid_model_indices = tf.logical_and(valid_2d, valid_4d)
        if valid_indices is None:
            valid_indices = valid_model_indices
        else:
            valid_indices = tf.logical_and(valid_indices, valid_model_indices)

    valid_indices = tf.where(valid_indices)[:, 0]
    n_valid_samples = tf.size(valid_indices)
    logging.info(f"Valid samples after alignment: {n_valid_samples.numpy()}")
    logging.info(f"Valid indices (first 10): {valid_indices[:10].numpy()}")
    logging.info(f"Valid indices shape: {valid_indices.shape}, max index: {tf.reduce_max(valid_indices).numpy()}")

    # Validate valid_indices
    if n_valid_samples == 0:
        for model_name in features_2d_dict:
            features_2d = features_2d_dict[model_name]
            features_4d = features_4d_dict[model_name.lower()]
            valid_2d = tf.reduce_all(tf.math.is_finite(features_2d), axis=1)
            valid_4d = tf.reduce_all(tf.math.is_finite(features_4d), axis=1)
            logging.info(f"Model {model_name}: valid 2D samples={tf.reduce_sum(tf.cast(valid_2d, tf.int32))}, valid 4D samples={tf.reduce_sum(tf.cast(valid_4d, tf.int32))}")
        raise ValueError("No valid samples after alignment. Check 2D and 4D features.")

    # Cast tf.shape(labels)[0] to int64 to match valid_indices
    labels_size = tf.cast(tf.shape(labels)[0], tf.int64)
    if tf.reduce_any(valid_indices >= labels_size):
        logging.error(f"valid_indices contains out-of-bounds indices. Max index: {tf.reduce_max(valid_indices).numpy()}, labels size: {labels_size.numpy()}")
        raise ValueError("valid_indices contains out-of-bounds indices.")

    # Validate labels before gather
    logging.info("Validating labels_normalized before tf.gather")
    if tf.reduce_any(tf.math.is_nan(labels_normalized)):
        logging.error("labels_normalized contains NaN values before tf.gather.")
        raise ValueError("labels_normalized contains NaN values before tf.gather.")
    if tf.reduce_any(tf.math.is_inf(labels_normalized)):
        logging.error("labels_normalized contains inf values before tf.gather.")
        raise ValueError("labels_normalized contains inf values before tf.gather.")

    # Align features and labels
    combined_features = []
    for model_name in features_2d_dict:
        features_2d = tf.gather(features_2d_dict[model_name], valid_indices)
        features_4d = tf.gather(features_4d_dict[model_name.lower()], valid_indices)
        if not tf.reduce_all(tf.math.is_finite(features_2d)):
            logging.error(f"Features 2D for {model_name} contain NaN/inf after gather.")
            raise ValueError(f"Features 2D for {model_name} contain NaN/inf after gather.")
        if not tf.reduce_all(tf.math.is_finite(features_4d)):
            logging.error(f"Features 4D for {model_name} contain NaN/inf after gather.")
            raise ValueError(f"Features 4D for {model_name} contain NaN/inf after gather.")
        combined_features.append(features_2d)
        combined_features.append(features_4d)
    combined_features = tf.concat(combined_features, axis=1)
    logging.info(f"Combined features shape before PCA: {combined_features.shape}")

    # Save features before PCA
    before_pca_path = os.path.join(save_dir, 'combined_features_before_pca.npy')
    np.save(before_pca_path, combined_features.numpy())
    logging.info(f"Saved combined features (before PCA) at: {before_pca_path}")

    # Apply PCA
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features.numpy())
    pca = PCA(n_components=n_components, random_state=42)
    features = pca.fit_transform(combined_features_scaled)
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    logging.info(f"Features shape after PCA: {features.shape}")

    # Save features after PCA
    after_pca_path = os.path.join(save_dir, 'combined_features_after_pca.npy')
    np.save(after_pca_path, features.numpy())
    logging.info(f"Saved features after PCA at: {after_pca_path}")

    # Align labels
    aligned_labels = tf.gather(labels_normalized, valid_indices)

    # Validate aligned_labels
    if tf.reduce_any(tf.math.is_nan(aligned_labels)):
        logging.error("aligned_labels contains NaN values after tf.gather.")
        raise ValueError("aligned_labels contains NaN values after tf.gather.")
    if tf.reduce_any(tf.math.is_inf(aligned_labels)):
        logging.error("aligned_labels contains inf values after tf.gather.")
        raise ValueError("aligned_labels contains inf values after tf.gather.")
    aligned_label_sums = tf.reduce_sum(aligned_labels, axis=1)
    if not tf.reduce_all(tf.logical_and(aligned_label_sums >= 0.9, aligned_label_sums <= 1.1)):
        logging.error("aligned_labels is not in valid one-hot format.")
        raise ValueError("aligned_labels is not in valid one-hot format.")

    # Save aligned_labels for debugging
    aligned_labels_path = os.path.join(save_dir, 'aligned_labels.npy')
    np.save(aligned_labels_path, aligned_labels.numpy())
    logging.info(f"Saved aligned labels at: {aligned_labels_path}")

    # Log aligned label distribution
    aligned_label_indices = tf.argmax(aligned_labels, axis=1, output_type=tf.int32)
    aligned_label_counts = tf.math.bincount(aligned_label_indices, minlength=NUM_CLASSES)
    logging.info(f"Aligned labels distribution: {dict(zip(range(NUM_CLASSES), aligned_label_counts.numpy()))}")
    if tf.reduce_all(tf.equal(aligned_label_counts, 0)):
        logging.error("No samples found in aligned_labels for any class.")
        raise ValueError("No samples found in aligned_labels for any class.")

    # Align sample_ids
    aligned_sample_ids = [sample_ids[i] for i in valid_indices.numpy()]

    # Validate alignment
    if features.shape[0] != aligned_labels.shape[0]:
        logging.error(f"Mismatch: features ({features.shape[0]}), aligned_labels ({aligned_labels.shape[0]})")
        raise ValueError(f"Mismatch: features ({features.shape[0]}), aligned_labels ({aligned_labels.shape[0]})")
    if features.shape[0] != len(aligned_sample_ids):
        logging.error(f"Mismatch: features ({features.shape[0]}), aligned_sample_ids ({len(aligned_sample_ids)})")
        raise ValueError(f"Mismatch: features ({features.shape[0]}), aligned_sample_ids ({len(aligned_sample_ids)})")

    # Save metadata
    metadata = {
        "before_pca_path": before_pca_path,
        "after_pca_path": after_pca_path,
        "aligned_labels_path": aligned_labels_path,
        "n_samples": int(n_valid_samples.numpy()),
        "n_components": int(n_components),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "valid_indices": [int(idx) for idx in valid_indices.numpy()],
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    metadata_path = os.path.join(save_dir, 'combined_features_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Saved metadata at: {metadata_path}")

    return features, pca, valid_indices.numpy().tolist(), features.shape[1], aligned_labels, aligned_sample_ids

# Function to reduce 4D to 2D (already provided in your code)
def reduce_4d_to_2d(features_4d):
    return np.mean(features_4d, axis=(1, 2))

def create_episode(features, labels,
                   n_support=15,
                   n_query=10,
                   hard_sample_ratio=0.3,
                   class_3_multiplier=2,
                   use_soft_labels=False,
                   seed=None,
                   return_metadata=True):
    """
    Create an episode for few-shot learning. Each class will have an equal number of support and query samples,
    except class 3 will have double the number. If there are insufficient samples, they will be replicated to meet the required number.

    Additionally, select a proportion of "hard" samples based on distance to class centroid for training.

    PARAMETERS:
        features: Array or tensor, shape [num_samples, feature_dim]
        labels: Array or tensor, one-hot (or soft labels), shape [num_samples, num_classes]
        n_support: Number of support samples per class
        n_query: Number of query samples per class
        hard_sample_ratio: Proportion of "hard" samples to select
        class_3_multiplier: Multiplier for the number of samples in class 3 (e.g., 2 means class 3 has double the samples)
        use_soft_labels: True if labels are not one-hot but soft (probability distribution)
        seed: Integer to set random seed for reproducibility
        return_metadata: True to return additional metadata: sample indices, class distribution...

    RETURNS:
        Tuple containing:
            support_features: Support samples, shape [?, feature_dim]
            support_labels: Labels of support samples, shape [?, num_classes]
            query_features: Query samples, shape [?, feature_dim]
            query_labels: Labels of query samples, shape [?, num_classes]
            hard_indices: Indices of hard samples in the entire set
            metadata (if return_metadata=True): Dict containing detailed episode information
    """

    # Step 1: Set seed for reproducibility if provided
    if seed is not None:
        tf.random.set_seed(seed)

    # Step 2: Ensure input data is Tensor
    features = tf.cast(tf.convert_to_tensor(features), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

    # Step 3: Check for NaN or Inf in labels
    if tf.reduce_any(tf.math.is_nan(labels)) or tf.reduce_any(tf.math.is_inf(labels)):
        raise ValueError("Labels contain NaN or Inf values, clean data first.")

    # Step 4: Ensure features have correct 2D shape
    if len(features.shape) != 2:
        print(f"Features have unexpected shape: {features.shape}, reshaping to 2D.")
        features = tf.reshape(features, [-1, features.shape[-1]])

    # Step 5: Determine number of classes
    num_classes = labels.shape[1]

    # Step 6: Prepare lists for support, query, and hard samples
    support_features_list = []
    support_labels_list = []
    query_features_list = []
    query_labels_list = []
    hard_indices_list = []

    # Metadata for tracking sample indices, distribution, etc.
    metadata = {
        "support_indices": [],
        "query_indices": [],
        "hard_sample_flags": [],
        "class_distribution": {}
    }

    # Step 7: Iterate through each class to create subsets
    for cls in range(num_classes):
        # Calculate number of support and query samples for this class
        if cls == 3:
            n_support_cls = n_support * class_3_multiplier
            n_query_cls = n_query * class_3_multiplier
        else:
            n_support_cls = n_support
            n_query_cls = n_query

        total_needed = n_support_cls + n_query_cls

        # Step 8: Filter indices for samples in the current class
        if use_soft_labels:
            cls_mask = tf.math.greater(labels[:, cls], 0.5)
        else:
            cls_mask = tf.equal(tf.argmax(labels, axis=1, output_type=tf.int32), cls)

        cls_indices = tf.where(cls_mask)[:, 0]
        cls_indices = tf.cast(cls_indices, tf.int32)
        n_cls_samples = tf.shape(cls_indices)[0]

        # Step 9: If insufficient samples, replicate to meet required number
        if n_cls_samples < total_needed:
            print(f"[WARNING] Insufficient samples for class {cls}: only {n_cls_samples.numpy()}, need {total_needed}")
            repeat_times = tf.math.ceil(total_needed / tf.cast(n_cls_samples, tf.float32))
            repeated_indices = tf.tile(cls_indices, [tf.cast(repeat_times, tf.int32)])
            shuffled_indices = tf.random.shuffle(repeated_indices)[:total_needed]
        else:
            shuffled_indices = tf.random.shuffle(cls_indices)[:total_needed]

        # Step 10: Split into support and query sets
        support_indices = shuffled_indices[:n_support_cls]
        query_indices = shuffled_indices[n_support_cls:n_support_cls + n_query_cls]

        # Step 11: Calculate distances and select hard samples
        selected_features = tf.gather(features, shuffled_indices)
        center = tf.reduce_mean(selected_features, axis=0)
        distances = tf.norm(selected_features - center, axis=1)
        n_hard_samples = max(1, int(n_support_cls * hard_sample_ratio))
        hard_sample_pos = tf.argsort(distances, direction='DESCENDING')[:n_hard_samples]
        hard_samples = tf.gather(shuffled_indices, hard_sample_pos)

        # Step 12: Store support, query, and hard samples
        support_features_list.append(tf.gather(features, support_indices))
        support_labels_list.append(tf.gather(labels, support_indices))
        query_features_list.append(tf.gather(features, query_indices))
        query_labels_list.append(tf.gather(labels, query_indices))
        hard_indices_list.append(hard_samples)

        # Record metadata if needed
        if return_metadata:
            metadata["support_indices"].append(support_indices.numpy())
            metadata["query_indices"].append(query_indices.numpy())
            metadata["hard_sample_flags"].append(hard_sample_pos.numpy())
            metadata["class_distribution"][cls] = {
                "support": int(n_support_cls),
                "query": int(n_query_cls),
                "hard": int(n_hard_samples)
            }

        # Print statistics for this class
        print(f"Class {cls}: {n_support_cls} support samples, {n_query_cls} query samples, {n_hard_samples} hard samples")

    # Step 13: Combine all classes into a single set
    if not support_features_list:
        raise ValueError("No classes have sufficient data to create episode.")

    support_features = tf.concat(support_features_list, axis=0)
    support_labels = tf.concat(support_labels_list, axis=0)
    query_features = tf.concat(query_features_list, axis=0)
    query_labels = tf.concat(query_labels_list, axis=0)
    hard_indices = tf.concat(hard_indices_list, axis=0)

    # Step 14: Calculate label distribution in support and query sets
    support_class_dist = tf.reduce_sum(support_labels, axis=0)
    query_class_dist = tf.reduce_sum(query_labels, axis=0)

    print(f"Episode summary:")
    print(f"  Support: {support_features.shape[0]} samples, query: {query_features.shape[0]} samples")
    print(f"  Class distribution (support): {support_class_dist.numpy()}")
    print(f"  Class distribution (query): {query_class_dist.numpy()}")

    # Step 15: Return results
    if return_metadata:
        return (
            support_features.numpy(),
            support_labels.numpy(),
            query_features.numpy(),
            query_labels.numpy(),
            hard_indices.numpy(),
            metadata
        )
    else:
        return (
            support_features.numpy(),
            support_labels.numpy(),
            query_features.numpy(),
            query_labels.numpy(),
            hard_indices.numpy()
        )

# Function to compute prototypes
def compute_prototypes(features, labels, feature_model):
    labels_arg = tf.argmax(labels, axis=1, output_type=tf.int32)
    prototypes = []
    for cls in range(NUM_CLASSES):
        cls_indices = tf.where(tf.equal(labels_arg, cls))[:, 0]
        if tf.size(cls_indices) == 0:
            prototypes.append(np.zeros(feature_model.output_shape[1], dtype=np.float32))
        else:
            cls_features = feature_model(tf.gather(features, cls_indices), training=False)
            prototype = tf.reduce_mean(cls_features, axis=0).numpy()
            prototypes.append(prototype)
    return np.array(prototypes, dtype=np.float32)

def prototypical_loss(features, labels, prototypes):
    labels_arg = tf.argmax(labels, axis=1, output_type=tf.int32)
    distances = tf.reduce_sum(tf.square(tf.expand_dims(features, 1) - prototypes), axis=2)
    log_probs = tf.nn.log_softmax(-distances, axis=1)
    loss = -tf.reduce_mean(tf.gather(log_probs, labels_arg, batch_dims=1))
    return loss

# Function for prototypical prediction
def prototypical_predict(features, prototypes):
    distances = tf.reduce_sum(tf.square(tf.expand_dims(features, 1) - prototypes), axis=2)
    probs = tf.nn.softmax(-distances, axis=1)
    return probs

# Temperature scaling function
def apply_temperature_scaling(logits, temperature=2.0):
    return tf.nn.softmax(logits / temperature)

# Laplace smoothing function
def laplace_smoothing(probs, epsilon=1e-5):
    return (probs + epsilon) / tf.reduce_sum(probs + epsilon, axis=1, keepdims=True)

# Function for learning rate reduction and early stopping
def reduce_lr_and_early_stop(episode, qwk, best_qwk, patience_lr, patience_stop, lr_patience_counter,
                             stop_patience_counter, inner_lr, outer_lr, fine_tune_lr, min_lr, reduce_factor):
    stop_training = False
    if qwk > best_qwk:
        lr_patience_counter = 0
        stop_patience_counter = 0
    else:
        lr_patience_counter += 1
        stop_patience_counter += 1
    if lr_patience_counter >= patience_lr:
        inner_lr = max(inner_lr * reduce_factor, min_lr)
        outer_lr = max(outer_lr * reduce_factor, min_lr)
        fine_tune_lr = max(fine_tune_lr * reduce_factor, min_lr)
        lr_patience_counter = 0
        print(f"Reduced learning rate at episode {episode+1}: inner_lr={inner_lr:.6f}, outer_lr={outer_lr:.6f}, fine_tune_lr={fine_tune_lr:.6f}")
    if stop_patience_counter >= patience_stop:
        stop_training = True
        print(f"Early stopping at episode {episode+1}")
    return inner_lr, outer_lr, fine_tune_lr, lr_patience_counter, stop_patience_counter, stop_training

# Function to save confusion matrix
def save_confusion_matrix(y_true, y_pred, episode, qwk, save_dir, prefix=''):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(range(NUM_CLASSES)),
                yticklabels=list(range(NUM_CLASSES)))
    plt.title(f'Confusion Matrix - {prefix}Episode {episode} (QWK: {qwk:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(save_dir, f'{prefix}confusion_matrix_episode_{episode}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix at: {cm_path}")
    return cm

# Function to save meta-learner features
def save_meta_learner_features(feature_model, features, labels, sample_ids, save_dir):
    """
    Extract and save 2D features from feature_model, ensuring dimensions match between features, labels, and sample_ids.

    Args:
        feature_model: Feature extraction model.
        features: Tensor or array containing input features (n, ...).
        labels: Tensor or array containing one-hot labels (n, num_classes).
        sample_ids: List of sample IDs (length n).
        save_dir: Directory to save features and labels.

    Returns:
        meta_features: Extracted 2D features (numpy array).
    """
    # Convert to Tensor and cast type
    features = tf.convert_to_tensor(features, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    n_features = features.shape[0]
    n_labels = labels.shape[0]
    n_sample_ids = len(sample_ids)

    # Log
    logging.info(f"save_meta_learner_features: features={features.shape}, labels={labels.shape}, sample_ids={n_sample_ids}")

    # Synchronize by trimming to min length
    min_len = min(n_features, n_labels, n_sample_ids)
    if n_features != min_len or n_labels != min_len or n_sample_ids != min_len:
        logging.warning(f"[TRIMMING] Size mismatch. Trimming to {min_len} samples.")
        features = features[:min_len]
        labels = labels[:min_len]
        sample_ids = sample_ids[:min_len]

    # Extract features
    meta_features = feature_model.predict(features, batch_size=32, verbose=0)

    # Ensure results are NumPy
    if isinstance(meta_features, tf.Tensor):
        meta_features = meta_features.numpy()
    if isinstance(labels, tf.Tensor):
        labels = labels.numpy()

    # Save data
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'meta_features_2d.npy'), meta_features)
    np.save(os.path.join(save_dir, 'meta_labels.npy'), labels)
    np.save(os.path.join(save_dir, 'meta_sample_ids.npy'), np.array(sample_ids))

    logging.info(f"✅ Saved features and labels at: {save_dir}")

    return meta_features

# Function to compute Grad-CAM
def compute_gradcam_4d(model, img_array, feature_4d, class_idx, layer_name, img_size=(224, 224)):
    """
    Compute Grad-CAM heatmap for an image.

    Args:
        model: Loaded Keras model with weights.
        img_array: Input image (shape: [1, height, width, 3]).
        feature_4d: 4D features (not used directly but kept for compatibility).
        class_idx: Predicted class index (int).
        layer_name: Name of the layer to extract features from (str).
        img_size: Input image size for the model (tuple).

    Returns:
        numpy.ndarray: Colored heatmap (RGB) or None if error occurs.
    """
    try:
        img_array_resized = tf.image.resize(img_array, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img_array_resized = tf.ensure_shape(img_array_resized, [1, img_size[0], img_size[1], 3])
        if hasattr(model, 'preprocess_input'):
            img_array_resized = model.preprocess_input(img_array_resized)
        grad_model = Model(
            inputs=[model.input],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array_resized)
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            logging.error(f"Gradient is None for layer {layer_name}")
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.reduce_max(heatmap)
        heatmap = tf.where(
            tf.math.logical_and(tf.math.is_finite(heatmap_max), heatmap_max > 0),
            heatmap / heatmap_max,
            tf.zeros_like(heatmap)
        )
        heatmap = heatmap.numpy()
        if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
            logging.warning(f"Heatmap contains NaN/inf for layer {layer_name}")
            return None
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap
    except Exception as e:
        logging.error(f"Error in compute_gradcam_4d: {str(e)}")
        return None

# Focal Loss for focusing on minority classes
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_factor = tf.pow(1 - y_pred, self.gamma)
        loss = focal_factor * cross_entropy

        if self.class_weights is not None:
            weights = tf.reduce_sum(y_true * self.class_weights, axis=1)
            loss = tf.reduce_sum(loss, axis=1) * weights
        else:
            loss = tf.reduce_sum(loss, axis=1)

        return tf.reduce_mean(loss)

def custom_random_erasing(data, p=0.3, sl=0.02, sh=0.4, r1=0.3, is_image=True):
    if np.random.rand() > p:
        return data

    if is_image:
        # Assume data is an image with shape [H, W, C]
        h, w, c = data.shape
        area = h * w
        for _ in range(100):
            target_area = np.random.uniform(sl, sh) * area
            aspect_ratio = np.random.uniform(r1, 1/r1)
            erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area / aspect_ratio)))
            if erase_w <= w and erase_h <= h:
                x1 = np.random.randint(0, h - erase_h + 1)
                y1 = np.random.randint(0, w - erase_w + 1)
                # Random color or noise for erasing
                if np.random.rand() < 0.5:
                    fill = np.random.uniform(0, 255, (erase_h, erase_w, c))
                else:
                    fill = np.random.normal(128, 32, (erase_h, erase_w, c)).clip(0, 255)
                data[x1:x1+erase_h, y1:y1+erase_w, :] = fill
                break
        # Add slight color jitter
        if np.random.rand() < 0.3:
            data = np.clip(data * np.random.uniform(0.8, 1.2), 0, 255)
    else:
        # For non-image data, add random noise
        noise = np.random.normal(0, 0.05, data.shape).astype(np.float32)
        data = data + noise

    return data

def create_augmenter(height=None, width=None, is_image=True):
    if is_image:
        # Image augmentations
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3)
            ], p=0.3),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.5
            ),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.RandomCrop(height=height, width=width, p=1.0) if height and width else A.NoOp(),
            A.OneOf([
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=2, p=0.3)
            ], p=0.2)
        ])
    else:
        # Non-image augmentations (e.g., feature noise or scaling)
        def non_image_augmenter(data):
            data = data.copy()
            if np.random.rand() < 0.5:
                # Add Gaussian noise
                noise = np.random.normal(0, 0.02, data.shape).astype(np.float32)
                data += noise
            if np.random.rand() < 0.3:
                # Random scaling
                scale = np.random.uniform(0.95, 1.05)
                data *= scale
            return data
        return non_image_augmenter
      
def sample_balanced_task(features, labels, n_support, n_query, n_way, min_required=2):
    is_image_input = features.ndim == 4 and features.shape[-1] in [1, 3]  # Support RGB or grayscale
    label_indices = tf.argmax(labels, axis=1).numpy()
    unique_classes, counts = np.unique(label_indices, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))

    # Adjust n_support/n_query if needed
    while n_support + n_query >= min_required:
        class_data = {cls: np.where(label_indices == cls)[0] for cls in unique_classes if class_counts[cls] >= min_required}
        if len(class_data) >= min(n_way, len(unique_classes)):
            break
        if n_support > 1:
            n_support -= 1
        elif n_query > 1:
            n_query -= 1
        else:
            break

    if not class_data:
        raise ValueError("No class has enough samples to create a balanced task.")

    # Select classes
    eligible_classes = list(class_data.keys())
    eligible_counts = np.array([class_counts[c] for c in eligible_classes])
    class_probs = 1.0 / (eligible_counts + 1e-6)
    class_probs /= class_probs.sum()
    replace_classes = len(eligible_classes) < n_way
    selected_classes = np.random.choice(eligible_classes, size=n_way, replace=replace_classes, p=class_probs)

    task_features, task_labels = [], []

    for cls in selected_classes:
        cls_indices = class_data[cls]
        total_needed = n_support + n_query
        replace_sampling = len(cls_indices) < total_needed
        selected = np.random.choice(cls_indices, size=total_needed, replace=replace_sampling)

        feats = tf.gather(features, selected).numpy()
        labs = tf.gather(labels, selected).numpy()

        if is_image_input:
            height, width = feats.shape[1:3]
            augmenter = create_augmenter(height, width, is_image=True)
            feats_aug = []
            for feat in feats:
                aug_img = augmenter(image=feat.astype(np.uint8))['image']
                aug_img = custom_random_erasing(aug_img.astype(np.float32), is_image=True)
                feats_aug.append(aug_img)
            feats = np.stack(feats_aug, axis=0)
        else:
            augmenter = create_augmenter(is_image=False)
            feats_aug = []
            for feat in feats:
                aug_feat = augmenter(feat.astype(np.float32))
                aug_feat = custom_random_erasing(aug_feat, is_image=False)
                feats_aug.append(aug_feat)
            feats = np.stack(feats_aug, axis=0)

        task_features.append(tf.convert_to_tensor(feats, dtype=tf.float32))
        task_labels.append(tf.convert_to_tensor(labs, dtype=tf.float32))

    features = tf.concat(task_features, axis=0)
    labels = tf.concat(task_labels, axis=0)

    if tf.math.reduce_any(tf.math.is_nan(labels)) or tf.math.reduce_any(tf.math.is_inf(labels)):
        raise ValueError("Labels contain NaN or inf after augmentation")

    return features, labels

def oversample_minority_classes(features, labels, min_count):
    is_image_input = features.ndim == 4 and features.shape[-1] in [1, 3]  # Support RGB or grayscale
    label_indices = tf.argmax(labels, axis=1).numpy()
    unique_classes = np.unique(label_indices)
    new_features, new_labels = [], []

    for cls in unique_classes:
        cls_indices = np.where(label_indices == cls)[0]
        current_count = len(cls_indices)
        num_needed = max(0, min_count - current_count)

        cls_feats = tf.gather(features, cls_indices)
        cls_labs = tf.gather(labels, cls_indices)

        new_features.append(cls_feats)
        new_labels.append(cls_labs)

        if num_needed > 0:
            print(f"Augmenting {num_needed} samples for class {cls}")
            if is_image_input:
                height, width = features.shape[1:3]
                augmenter = create_augmenter(height, width, is_image=True)
                for _ in range(num_needed):
                    idx = np.random.choice(cls_indices)
                    img = features[idx].numpy().astype(np.uint8)
                    label = labels[idx].numpy()

                    aug_img = augmenter(image=img)['image']
                    aug_img = custom_random_erasing(aug_img.astype(np.float32), is_image=True)

                    new_features.append(tf.convert_to_tensor([aug_img], dtype=tf.float32))
                    new_labels.append(tf.convert_to_tensor([label], dtype=tf.float32))
            else:
                augmenter = create_augmenter(is_image=False)
                for _ in range(num_needed):
                    idx = np.random.choice(cls_indices)
                    feat = features[idx].numpy()
                    label = labels[idx].numpy()

                    aug_feat = augmenter(feat.astype(np.float32))
                    aug_feat = custom_random_erasing(aug_feat, is_image=False)

                    new_features.append(tf.convert_to_tensor([aug_feat], dtype=tf.float32))
                    new_labels.append(tf.convert_to_tensor([label], dtype=tf.float32))

    features = tf.concat(new_features, axis=0)
    labels = tf.concat(new_labels, axis=0)

    if tf.math.reduce_any(tf.math.is_nan(labels)) or tf.math.reduce_any(tf.math.is_inf(labels)):
        raise ValueError("Labels contain NaN or inf after augmentation")
    if tf.math.reduce_any(tf.math.is_nan(features)) or tf.math.reduce_any(tf.math.is_inf(features)):
        raise ValueError("Features contain NaN or inf after augmentation")

    return features, labels

def oversample_validation_set(features, labels):
    is_image_input = features.ndim == 4 and features.shape[-1] in [1, 3]  # Support RGB or grayscale
    label_indices = tf.argmax(labels, axis=1).numpy()
    unique_classes, counts = np.unique(label_indices, return_counts=True)
    max_count = np.max(counts)

    new_features, new_labels = [], []

    for cls in unique_classes:
        cls_indices = np.where(label_indices == cls)[0]
        current_count = len(cls_indices)
        n_to_add = max_count - current_count

        cls_feats = tf.gather(features, cls_indices)
        cls_labs = tf.gather(labels, cls_indices)

        new_features.append(cls_feats)
        new_labels.append(cls_labs)

        if n_to_add > 0:
            print(f"Augmenting {n_to_add} samples for class {cls}")
            if is_image_input:
                height, width = features.shape[1:3]
                augmenter = create_augmenter(height, width, is_image=True)
                for _ in range(n_to_add):
                    idx = np.random.choice(cls_indices)
                    img = features[idx].numpy().astype(np.uint8)
                    label = labels[idx].numpy()

                    aug_img = augmenter(image=img)['image']
                    aug_img = custom_random_erasing(aug_img.astype(np.float32), is_image=True)

                    new_features.append(tf.convert_to_tensor([aug_img], dtype=tf.float32))
                    new_labels.append(tf.convert_to_tensor([label], dtype=tf.float32))
            else:
                augmenter = create_augmenter(is_image=False)
                for _ in range(n_to_add):
                    idx = np.random.choice(cls_indices)
                    feat = features[idx].numpy()
                    label = labels[idx].numpy()

                    aug_feat = augmenter(feat.astype(np.float32))
                    aug_feat = custom_random_erasing(aug_feat, is_image=False)

                    new_features.append(tf.convert_to_tensor([aug_feat], dtype=tf.float32))
                    new_labels.append(tf.convert_to_tensor([label], dtype=tf.float32))

    features = tf.concat(new_features, axis=0)
    labels = tf.concat(new_labels, axis=0)

    if tf.math.reduce_any(tf.math.is_nan(labels)) or tf.math.reduce_any(tf.math.is_inf(labels)):
        raise ValueError("Labels contain NaN or inf after augmentation")
    if tf.math.reduce_any(tf.math.is_nan(features)) or tf.math.reduce_any(tf.math.is_inf(features)):
        raise ValueError("Features contain NaN or inf after augmentation")

    return features, labels

def maml_fomaml_train_manual(
    features, labels, valid_features, valid_labels, input_dim, n_episodes=50,  # Increase number of episodes
    n_support=15, n_query=10, inner_lr=0.001, outer_lr=0.001, fine_tune_lr=0.0001,
    use_fomaml=True, memory_size=20, sample_ids=None, images=None, features_4d_dict=None
):
    """
    Function to train a meta-learning model using MAML or FOMAML.

    Args:
        features: Training features (Tensor, shape: [n_samples, input_dim]).
        labels: Training labels (Tensor, one-hot, shape: [n_samples, NUM_CLASSES]).
        valid_features: Validation features (Tensor, shape: [n_valid_samples, input_dim]).
        valid_labels: Validation labels (Tensor, one-hot, shape: [n_valid_samples, NUM_CLASSES]).
        input_dim: Input feature dimension (int).
        n_episodes: Number of meta-learning episodes (int).
        n_support: Number of support samples per class (int).
        n_query: Number of query samples per class (int).
        inner_lr: Inner loop learning rate (float).
        outer_lr: Outer loop learning rate (float).
        fine_tune_lr: Learning rate for fine-tuning (float).
        use_fomaml: Use FOMAML or MAML (bool).
        memory_size: Memory size for MemoryAugmentedLayer (int).
        sample_ids: Sample IDs for validation (list, optional).
        images: Original images for Grad-CAM (array, optional).
        features_4d_dict: Dictionary of 4D features (dict, optional).

    Returns:
        meta_model: Trained meta-learning model.
        meta_classification_model: Meta-learning classification model.
        feature_model: Feature extraction model.
        history: Training history (dict).
    """
    # Convert and validate data types
    features = tf.cast(tf.convert_to_tensor(features), dtype=tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), dtype=tf.float32)
    valid_features = tf.cast(tf.convert_to_tensor(valid_features), dtype=tf.float32)
    valid_labels = tf.cast(tf.convert_to_tensor(valid_labels), dtype=tf.float32)

    # Check dimensions
    if not isinstance(input_dim, int):
        raise ValueError(f"input_dim must be an integer, received {input_dim}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch in dimensions: features ({features.shape[0]}), labels ({labels.shape[0]})")
    if valid_features.shape[0] != valid_labels.shape[0]:
        raise ValueError(f"Mismatch in dimensions: valid_features ({valid_features.shape[0]}), valid_labels ({valid_labels.shape[0]})")
    if features.shape[1] != input_dim:
        raise ValueError(f"features.shape[1] ({features.shape[1]}) does not match input_dim ({input_dim})")
    if labels.shape[1] != NUM_CLASSES:
        raise ValueError(f"labels.shape[1] ({labels.shape[1]}) does not match NUM_CLASSES ({NUM_CLASSES})")
    if sample_ids is not None and len(sample_ids) != valid_features.shape[0]:
        raise ValueError(f"Mismatch in dimensions: sample_ids ({len(sample_ids)}), valid_features ({valid_features.shape[0]})")

    # Log dimensions
    logging.info(f"features shape: {features.shape}")
    logging.info(f"labels shape: {labels.shape}")
    logging.info(f"valid_features shape: {valid_features.shape}")
    logging.info(f"valid_labels shape: {valid_labels.shape}")

    # Check for NaN and inf
    if tf.reduce_any(tf.math.is_nan(labels)) or tf.reduce_any(tf.math.is_inf(labels)):
        raise ValueError("labels contain NaN or inf.")
    if tf.reduce_any(tf.math.is_nan(features)) or tf.reduce_any(tf.math.is_inf(features)):
        logging.warning("features contain NaN or inf, replacing with 0.")
        features = np.where(tf.math.logical_or(tf.math.is_nan(features), tf.math.is_inf(features)), 0.0, features)

    # Check one-hot format
    label_sums = tf.reduce_sum(labels, axis=1)
    if not tf.reduce_all(tf.logical_and(label_sums >= 0.9, label_sums <= 1.1)):
        raise ValueError("labels are not in valid one-hot format.")
    if tf.reduce_any(labels < 0.0) or tf.reduce_any(labels > 1.0):
        logging.warning("labels contain values outside [0.0, 1.0], normalizing to [0,1].")
        labels = np.where(labels >= 0.5, 1.0, 0.0)

    # Normalize labels
    labels = tf.where(labels >= 0.5, 1.0, 0.0)
    labels_normalized = labels  # or keep if separate operations are needed

    if tf.reduce_any(tf.math.is_nan(labels_normalized)) or tf.reduce_any(tf.math.is_inf(labels_normalized)):
        raise ValueError("labels contain NaN or inf after normalization.")

    # Label distribution
    label_indices = tf.argmax(labels, axis=1, output_type=tf.int32)
    class_counts = tf.cast(tf.math.bincount(label_indices, minlength=NUM_CLASSES), tf.float32)

    logging.info(f"Label distribution: {dict(zip(range(NUM_CLASSES), class_counts.numpy()))}")
    if tf.reduce_all(tf.equal(class_counts, 0)):
        raise ValueError("labels contain no samples for any class.")

    # Define MemoryAugmentedLayer
    class MemoryAugmentedLayer(tf.keras.layers.Layer):
        def __init__(self, memory_size, memory_dim, **kwargs):
            super().__init__(**kwargs)
            self.memory_size = memory_size
            self.memory_dim = memory_dim
            self.memory_projection = None  # Move Dense here

        def build(self, input_shape):
            self.input_dim = input_shape[-1]

            # Create memory
            self.memory = self.add_weight(
                shape=(self.memory_size, self.memory_dim),
                initializer='zeros',
                trainable=False,
                name='memory',
                dtype=tf.float32
            )

            # If memory_dim != input_dim → need projection
            if self.memory_dim != self.input_dim:
                self.memory_projection = tf.keras.layers.Dense(self.input_dim, use_bias=False, name="memory_projection")

            super().build(input_shape)

        def call(self, inputs):
            if len(inputs.shape) != 2:
                logging.warning(f"Input to MemoryAugmentedLayer is not 2D: {inputs.shape}, reshaping to 2D")
                inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])

            batch_size = tf.shape(inputs)[0]
            memory_size = tf.shape(self.memory)[0]

            memory_sliced = tf.cond(
                tf.greater(batch_size, memory_size),
                lambda: tf.tile(self.memory, [(batch_size + memory_size - 1) // memory_size, 1])[:batch_size],
                lambda: self.memory[:batch_size]
            )

            # Call Dense if needed
            if self.memory_projection is not None:
                memory_sliced = self.memory_projection(memory_sliced)

            # Concatenate and compute mean
            stacked = tf.stack([inputs, memory_sliced], axis=0)
            output = tf.reduce_mean(stacked, axis=0)
            return output

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])

        def get_config(self):
            config = super().get_config()
            config.update({
                'memory_size': self.memory_size,
                'memory_dim': self.memory_dim,
            })
            return config

    # Define create_model
    def create_model(input_dim):
        inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = CustomGridDropout(ratio=0.3, holes_number=10, p=0.3)(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = CustomGridDropout(ratio=0.3, holes_number=5, p=0.3)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        feature_output = x
        x = MemoryAugmentedLayer(memory_size=memory_size, memory_dim=64)(x)
        classification_output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='classification')(x)
        domain_inputs = GradientReversalLayer(lambda_=1.0)(x)
        domain_x = tf.keras.layers.Dense(32, activation='relu')(domain_inputs)
        domain_output = tf.keras.layers.Dense(2, activation='softmax', name='domain')(domain_x)

        model = tf.keras.Model(inputs=inputs, outputs=[classification_output, domain_output])
        classification_model = tf.keras.Model(inputs=inputs, outputs=classification_output)
        feature_model = tf.keras.Model(inputs=inputs, outputs=feature_output)

        memory_layers = [l for l in model.layers if isinstance(l, MemoryAugmentedLayer)]
        if not memory_layers:
            raise ValueError("No MemoryAugmentedLayer found in the model.")
        memory_layer = memory_layers[0]

        logging.info(f"Model created with input shape: {inputs.shape}, classification output shape: {classification_output.shape}")

        return model, classification_model, memory_layer, feature_model

    # Validate before calling create_model
    if not isinstance(NUM_CLASSES, int) or NUM_CLASSES <= 0:
        raise ValueError(f"NUM_CLASSES is invalid: {NUM_CLASSES}")
    if not isinstance(memory_size, int) or memory_size <= 0:
        raise ValueError(f"memory_size is invalid: {memory_size}")
    logging.info(f"input_dim: {input_dim}")

    # Initialize model
    tf.keras.backend.clear_session()
    meta_model, meta_classification_model, memory_layer, feature_model = create_model(input_dim)
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=outer_lr)
    fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    loss_fn = FocalLoss(gamma=2.0)  # High alpha to focus on minority classes
    domain_loss_fn = tf.keras.losses.CategoricalCrossentropy()

    class_counts = tf.cast(tf.math.bincount(tf.argmax(labels, axis=1), minlength=NUM_CLASSES), tf.float32)
    class_weights = tf.where(class_counts > 0, 1.0 / class_counts, 1.0)
    class_weights = class_weights / tf.reduce_sum(class_weights) * NUM_CLASSES
    class_weights = tf.tensor_scatter_nd_update(class_weights, [[3], [4]], [class_weights[3] * 3.0, class_weights[4] * 3.0])  # Increase weights for classes 3, 4
    logging.info(f"Class weights: {class_weights.numpy()}")

    # Accuracy computation function
    def compute_accuracy(y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

    # Initialize tracking variables
    best_qwk = -float('inf')
    history = {
        'qwk': [], 'support_loss': [], 'support_accuracy': [],
        'query_loss': [], 'query_accuracy': [], 'precision': [], 'recall': []
    }
    lr_patience_counter = 5
    stop_patience_counter = 10
    patience_lr = 5  # Reduced from 10 to 5
    patience_stop = 20  # Reduced from 50 to 20
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Reduced patience from 5 to 3
    min_lr = 1e-7
    reduce_factor = 0.5
    weights_filepath = os.path.join(feature_save_dir, "meta_model_maml_fomaml_best_weights.weights.h5")

    # Callbacks and domain labels
    cm_callback = ConfusionMatrixWeightCallback(
        valid_features=valid_features,
        valid_labels=valid_labels,
        classification_model=meta_classification_model,
        num_classes=NUM_CLASSES,
        class_counts=class_counts
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    iso_reg = CustomIsotonicRegression()
    source_domain_labels = tf.keras.utils.to_categorical(tf.zeros(tf.shape(features)[0], dtype=tf.int32), num_classes=2)
    target_domain_labels = tf.keras.utils.to_categorical(tf.ones(tf.shape(valid_features)[0], dtype=tf.int32), num_classes=2)
    class_weights = tf.ones(NUM_CLASSES, dtype=tf.float32)
    class_weights = tf.tensor_scatter_nd_update(class_weights, [[3]], [10 / (10 * 2)])

    # Episode loop
    for episode in range(n_episodes):
        tf.keras.backend.clear_session()
        support_features, support_labels, query_features, query_labels, hard_indices, metadata = create_episode(
            features=features,
            labels=labels,
            n_support=20,
            n_query=20,
            hard_sample_ratio=0.3,
            class_3_multiplier=2,
            return_metadata=True  # Ensure this
        )

        # ⚖️ Balance each class in support & query
        support_features, support_labels = sample_balanced_task(support_features, support_labels, n_support, n_query, n_way=5)
        query_features, query_labels = sample_balanced_task(query_features, query_labels, n_support, n_query, n_way=5)

        if tf.size(support_features) == 0 or tf.size(query_features) == 0:
            logging.warning(f"Episode {episode+1}: Empty episode, skipping.")
            history['qwk'].append(0.0)
            history['support_loss'].append(0.0)
            history['support_accuracy'].append(0.0)
            history['query_loss'].append(0.0)
            history['query_accuracy'].append(0.0)
            history['precision'].append(0.0)
            history['recall'].append(0.0)
            continue

        logging.info(f"Episode {episode+1}: support_features shape={support_features.shape}, support_labels shape={support_labels.shape}")
        logging.info(f"Episode {episode+1}: query_features shape={query_features.shape}, query_labels shape={query_labels.shape}")

        task_model, task_classification_model, task_memory_layer, task_feature_model = create_model(input_dim)
        task_model.set_weights(meta_model.get_weights())
        task_optimizer = tf.keras.optimizers.Adam(learning_rate=inner_lr)

        support_prototypes = compute_prototypes(support_features, support_labels, task_feature_model)

        # Inner loop
        for _ in range(15):
            with tf.GradientTape() as tape:
                class_preds, domain_preds = task_model(support_features, training=True)
                if class_preds.shape[-1] != NUM_CLASSES:
                    raise ValueError(f"Invalid class_preds shape: {class_preds.shape}, expected [batch_size, {NUM_CLASSES}]")
                min_size = tf.minimum(tf.shape(class_preds)[0], tf.shape(support_labels)[0])
                class_preds = class_preds[:min_size]
                support_labels_adj = support_labels[:min_size]
                min_size_domain = tf.minimum(tf.shape(domain_preds)[0], tf.shape(source_domain_labels)[0])
                domain_preds = domain_preds[:min_size_domain]
                source_domain_labels_slice = source_domain_labels[:min_size_domain]
                support_labels_indices = tf.argmax(support_labels_adj, axis=1, output_type=tf.int32)
                sample_weights = tf.gather(class_weights, support_labels_indices)
                class_loss = loss_fn(support_labels_adj, class_preds, sample_weight=sample_weights)
                domain_loss = domain_loss_fn(source_domain_labels_slice, domain_preds)
                support_features_task = task_feature_model(support_features, training=False)
                proto_loss = prototypical_loss(support_features_task, support_labels_adj, support_prototypes)
                total_loss = class_loss + 0.5 * domain_loss + 0.5 * proto_loss
            task_grads = tape.gradient(total_loss, task_model.trainable_variables)
            valid_grads = [(g, v) for g, v in zip(task_grads, task_model.trainable_variables) if g is not None]
            task_optimizer.apply_gradients(valid_grads)
            task_keys = task_feature_model(support_features, training=False)
            if task_keys.shape[1] != 128:
                task_keys = tf.keras.layers.Dense(128, use_bias=False, dtype=tf.float32)(task_keys)
            task_keys = tf.concat([task_keys, tf.zeros((memory_size - tf.shape(task_keys)[0], task_keys.shape[1]), dtype=tf.float32)], axis=0) if tf.shape(task_keys)[0] < memory_size else task_keys[:memory_size]
            task_keys = task_feature_model(support_features, training=False)

            # Fix: If task_keys.shape[1] != 64, project to 64
            if task_keys.shape[1] != 64:
                projector = tf.keras.layers.Dense(64, use_bias=False, dtype=tf.float32)
                task_keys = projector(task_keys)

            # Ensure correct size
            task_keys = tf.concat(
                [task_keys, tf.zeros((memory_size - tf.shape(task_keys)[0], 64), dtype=tf.float32)],
                axis=0
            ) if tf.shape(task_keys)[0] < memory_size else task_keys[:memory_size]

            task_memory_layer.memory.assign(task_keys)

            gc.collect()

        support_preds, _ = task_model(support_features, training=False)
        support_loss_value = float(loss_fn(support_labels_adj, support_preds).numpy())
        support_accuracy = float(compute_accuracy(support_labels_adj, support_preds).numpy())

        # Outer loop
        with tf.GradientTape() as outer_tape:
            query_preds, domain_preds = task_model(query_features, training=True)
            min_size = tf.minimum(tf.shape(query_preds)[0], tf.shape(query_labels)[0])
            query_preds = query_preds[:min_size]
            query_labels_adj = query_labels[:min_size]
            min_size_domain = tf.minimum(tf.shape(domain_preds)[0], tf.shape(source_domain_labels)[0])
            domain_preds = domain_preds[:min_size_domain]
            source_domain_labels_slice = source_domain_labels[:min_size_domain]
            query_labels_indices = tf.argmax(query_labels_adj, axis=1, output_type=tf.int32)
            sample_weights = tf.gather(class_weights, query_labels_indices)
            query_loss = loss_fn(query_labels_adj, query_preds, sample_weight=sample_weights)
            domain_loss = domain_loss_fn(source_domain_labels_slice, domain_preds)
            query_features_task = task_feature_model(query_features, training=False)
            proto_loss = prototypical_loss(query_features_task, query_labels_adj, support_prototypes)
            total_query_loss = query_loss + 0.5 * domain_loss + 0.5 * proto_loss
            query_accuracy = float(compute_accuracy(query_labels_adj, query_preds).numpy())
            query_loss_value = float(query_loss.numpy())

        meta_grads = outer_tape.gradient(total_query_loss, task_model.trainable_variables)
        valid_grads = [(g, v) for g, v in zip(meta_grads, meta_model.trainable_variables) if g is not None]
        meta_optimizer.apply_gradients(valid_grads)
        memory_keys = feature_model(support_features, training=False)
        if memory_keys.shape[1] != 128:
            memory_keys = tf.keras.layers.Dense(128, use_bias=False, dtype=tf.float32)(memory_keys)
        memory_keys = tf.concat([memory_keys, tf.zeros((memory_size - tf.shape(memory_keys)[0], memory_keys.shape[1]), dtype=tf.float32)], axis=0) if tf.shape(memory_keys)[0] < memory_size else memory_keys[:memory_size]
        memory_keys = feature_model(support_features, training=False)

        # Fix: If memory_keys.shape[1] != 64, project to 64
        if memory_keys.shape[1] != 64:
            projector = tf.keras.layers.Dense(64, use_bias=False, dtype=tf.float32)
            memory_keys = projector(memory_keys)

        # Ensure correct size
        memory_keys = tf.concat(
            [memory_keys, tf.zeros((memory_size - tf.shape(memory_keys)[0], 64), dtype=tf.float32)],
            axis=0
        ) if tf.shape(memory_keys)[0] < memory_size else memory_keys[:memory_size]

        memory_layer.memory.assign(memory_keys)

        gc.collect()

        # Fine-tune
        for _ in range(5):
            with tf.GradientTape() as fine_tune_tape:
                fine_tune_preds = meta_classification_model(query_features, training=True)
                min_size = tf.minimum(tf.shape(fine_tune_preds)[0], tf.shape(query_labels)[0])
                if min_size == 0:
                    continue
                fine_tune_preds = fine_tune_preds[:min_size]
                query_labels_adj = query_labels[:min_size]
                query_labels_indices = tf.argmax(query_labels_adj, axis=1, output_type=tf.int32)
                sample_weights = tf.gather(class_weights, query_labels_indices)
                fine_tune_loss = loss_fn(query_labels_adj, fine_tune_preds, sample_weight=sample_weights)
            fine_tune_grads = fine_tune_tape.gradient(fine_tune_loss, meta_classification_model.trainable_variables)
            valid_grads = [(g, v) for g, v in zip(fine_tune_grads, meta_classification_model.trainable_variables) if g is not None]
            fine_tune_optimizer.apply_gradients(valid_grads)
            gc.collect()

        # Evaluate on validation set
        valid_features, valid_labels = oversample_validation_set(valid_features, valid_labels)
        valid_preds_maml = meta_classification_model.predict(valid_features, batch_size=32)
        min_size = tf.minimum(tf.shape(valid_preds_maml)[0], tf.shape(valid_labels)[0])
        valid_preds_maml = valid_preds_maml[:min_size]
        valid_labels_adj = valid_labels[:min_size]
        valid_features_task = task_feature_model(valid_features, training=False)
        valid_prototypes = compute_prototypes(valid_features, valid_labels_adj, task_feature_model)
        valid_preds_proto = prototypical_predict(valid_features_task, valid_prototypes)
        valid_preds_ensemble = 0.7 * valid_preds_maml + 0.3 * valid_preds_proto
        valid_preds_scaled = apply_temperature_scaling(valid_preds_ensemble, temperature=2.0)
        valid_preds_scaled = laplace_smoothing(valid_preds_scaled, epsilon=1e-6)
        valid_probs = tf.reduce_max(valid_preds_scaled, axis=1)
        valid_probs = tf.clip_by_value(valid_probs, 0.0, 1.0)
        valid_probs_np = valid_probs.numpy()
        valid_preds_classes = tf.argmax(valid_preds_scaled, axis=1, output_type=tf.int32).numpy()
        valid_labels_classes = tf.argmax(valid_labels_adj, axis=1, output_type=tf.int32).numpy()

        if episode >= 10:
            valid_preds_calibrated = iso_reg.predict(valid_probs_np)
        else:
            iso_reg.fit(valid_probs_np, valid_labels_classes)
        qwk = cohen_kappa_score(valid_labels_classes, valid_preds_classes,
                                labels=list(range(NUM_CLASSES)), weights='quadratic')
        precision = precision_score(valid_labels_classes, valid_preds_classes, average='weighted')
        recall = recall_score(valid_labels_classes, valid_preds_classes, average='weighted')

        history['qwk'].append(float(qwk))
        history['support_loss'].append(support_loss_value)
        history['support_accuracy'].append(support_accuracy)
        history['query_loss'].append(query_loss_value)
        history['query_accuracy'].append(query_accuracy)
        history['precision'].append(float(precision))
        history['recall'].append(float(recall))

        cm_callback.on_epoch_end(episode)

        inner_lr, outer_lr, fine_tune_lr, lr_patience_counter, stop_patience_counter, stop_training = \
            reduce_lr_and_early_stop(
                episode, qwk, best_qwk, patience_lr, patience_stop,
                lr_patience_counter, stop_patience_counter,
                inner_lr, outer_lr, fine_tune_lr, min_lr, reduce_factor
            )

        meta_optimizer.learning_rate.assign(outer_lr)
        fine_tune_optimizer.learning_rate.assign(fine_tune_lr)

        # Log to TensorBoard
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar('support_loss', support_loss_value, step=episode)
            tf.summary.scalar('support_accuracy', support_accuracy, step=episode)
            tf.summary.scalar('query_loss', query_loss_value, step=episode)
            tf.summary.scalar('query_accuracy', query_accuracy, step=episode)
            tf.summary.scalar('qwk', qwk, step=episode)
            tf.summary.scalar('precision', precision, step=episode)
            tf.summary.scalar('recall', recall, step=episode)
            tf.summary.scalar('inner_lr', inner_lr, step=episode)
            tf.summary.scalar('outer_lr', outer_lr, step=episode)
            tf.summary.scalar('fine_tune_lr', fine_tune_lr, step=episode)

        print(f"\nEpisode {episode+1}/{n_episodes}:")
        print(f"  Support Loss: {support_loss_value:.4f}, Accuracy: {support_accuracy:.4f}")
        print(f"  Query Loss: {query_loss_value:.4f}, Accuracy: {query_accuracy:.4f}")
        print(f"  QWK (Ensemble): {qwk:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Save best weights
        if qwk > best_qwk:
            best_qwk = qwk
            try:
                meta_model.save_weights(weights_filepath, overwrite=True)
                print(f"Saved best weights at episode {episode+1} with QWK: {best_qwk:.4f}")
                cm = save_confusion_matrix(
                    valid_labels_classes,
                    valid_preds_classes,
                    episode + 1,
                    best_qwk,
                    feature_save_dir,
                    prefix='best_'
                )
                print(f"Confusion matrix for best QWK at Episode {episode+1}:\n{cm}")
            except Exception as e:
                logging.error(f"Error saving weights: {str(e)}")
                alt_weights_filepath = os.path.join(feature_save_dir, "meta_model_maml_fomaml_best_weights_alt.h5")
                meta_model.save_weights(alt_weights_filepath, overwrite=True)
                print(f"Saved weights (alternative) at: {alt_weights_filepath}")

        if stop_training:
            print(f"Early stopping triggered at episode {episode+1}")
            break
        gc.collect()

    # Save 2D features
    meta_features_2d = None
    if sample_ids is not None:
        logging.info("Saving 2D features of meta-learner...")
        meta_features_2d = save_meta_learner_features(
            feature_model, valid_features, valid_labels, sample_ids, feature_save_dir
        )

    # Fine-tune on validation set
    print("\nFine-tuning on validation set...")
    meta_classification_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    meta_classification_model.fit(
        valid_features, valid_labels,
        validation_data=(valid_features, valid_labels),
        epochs=150,
        batch_size=64,
        verbose=1,
        callbacks=[
            tensorboard_callback,
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            early_stopping
        ]
    )

    # Final evaluation
    valid_preds = meta_classification_model.predict(valid_features, batch_size=32)
    valid_preds = apply_temperature_scaling(valid_preds, temperature=2.0)
    valid_preds = laplace_smoothing(valid_preds, epsilon=1e-6)
    valid_probs = tf.reduce_max(valid_preds, axis=1)
    valid_probs = tf.clip_by_value(valid_probs, 0.0, 1.0)
    valid_probs_np = valid_probs.numpy()
    valid_preds_classes = tf.argmax(valid_preds, axis=1, output_type=tf.int32).numpy()
    valid_true_classes = tf.argmax(valid_labels, axis=1, output_type=tf.int32).numpy()
    valid_preds_calibrated = iso_reg.predict(valid_probs_np)

    qwk_final = cohen_kappa_score(valid_true_classes, valid_preds_classes,
                                  labels=list(range(NUM_CLASSES)), weights='quadratic')
    f1_final = f1_score(valid_true_classes, valid_preds_classes, average='weighted')
    recall_final = recall_score(valid_true_classes, valid_preds_classes, average='weighted')
    precision_final = precision_score(valid_true_classes, valid_preds_classes, average='weighted')

    print(f"\nFinal results on validation set:")
    print(f"Quadratic Weighted Kappa (QWK): {qwk_final:.4f}")
    print(f"Weighted F1 Score: {f1_final:.4f}")
    print(f"Weighted Recall: {recall_final:.4f}")
    print(f"Weighted Precision: {precision_final:.4f}")

    cm_final = save_confusion_matrix(
        valid_true_classes, valid_preds_classes, n_episodes, qwk_final, feature_save_dir, prefix='final_'
    )
    print(f"Final confusion matrix:\n{cm_final}")

    # Save model and metadata
    final_weights_filepath = os.path.join(feature_save_dir, "model.weights.h5")
    final_config_filepath = os.path.join(feature_save_dir, "config.json")
    final_metadata_filepath = os.path.join(feature_save_dir, "metadata.json")

    try:
        meta_model.save_weights(final_weights_filepath, overwrite=True)
        print(f"Saved meta-model weights at: {final_weights_filepath}")
    except Exception as e:
        logging.error(f"Error saving weights: {str(e)}")
        alt_final_weights_filepath = os.path.join(feature_save_dir, "model_final_weights_alt.h5")
        meta_model.save_weights(alt_final_weights_filepath, overwrite=True)
        print(f"Saved weights at: {alt_final_weights_filepath}")

    try:
        model_config = meta_model.to_json()
        with open(final_config_filepath, mode='w') as f:
            json.dump(json.loads(model_config), f, indent=4)
        print(f"Saved configuration at: {final_config_filepath}")
    except Exception as e:
        logging.error(f"Error saving meta-model configuration: {str(e)}")

    metadata = {
        "model_type": "meta_model_maml_fomaml",
        "n_episodes": n_episodes,
        "n_support": n_support,
        "n_query": n_query,
        "inner_lr": float(inner_lr),
        "outer_lr": float(outer_lr),
        "fine_tune_lr": float(fine_tune_lr),
        "qwk_final": float(qwk_final),
        "f1_final": float(f1_final),
        "recall_final": float(recall_final),
        "precision_final": float(precision_final),
        "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        with open(final_metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata at: {final_metadata_filepath}")
    except Exception as e:
        logging.error(f"Error saving metadata: {str(e)}")

    metrics_history = {
        'qwk': float(qwk_final),
        'f1_score': float(f1_final),
        'recall': float(recall_final),
        'precision': float(precision_final),
        'training_history': {
            'qwk': [float(q) for q in history['qwk']],
            'support_loss': [float(l) for l in history['support_loss']],
            'support_accuracy': [float(a) for a in history['support_accuracy']],
            'query_loss': [float(l) for l in history['query_loss']],
            'query_accuracy': [float(a) for a in history['query_accuracy']],
            'precision': [float(p) for p in history['precision']],
            'recall': [float(r) for r in history['recall']]
        }
    }
    metrics_filepath = os.path.join(feature_save_dir, 'final_metrics.json')
    try:
        with open(metrics_filepath, 'w') as f:
            json.dump(metrics_history, f, indent=4)
        print(f"Saved evaluation metrics at: {metrics_filepath}")
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")

    # Synchronize length with sample_ids
    max_len = len(sample_ids)

    # Check and truncate to avoid index out of bounds
    valid_features = valid_features[:max_len]
    valid_labels = valid_labels[:max_len]
    images = images[:max_len]

    # Compute and save Grad-CAM images
    if images is not None and features_4d_dict is not None:
        print("\nComputing and saving Grad-CAM images...")
        if sample_ids is None:
            logging.warning("sample_ids is None, skipping Grad-CAM computation.")
        else:
            for label in range(NUM_CLASSES):
                class_dir = os.path.join(gradcam_save_dir, f"class_label_{label}")
                os.makedirs(class_dir, exist_ok=True)

                indices = tf.where(tf.argmax(valid_labels, axis=1, output_type=tf.int32) == label)
                indices = tf.squeeze(indices, axis=-1)
                if tf.size(indices) == 0:
                    logging.warning(f"No samples found for label {label}. Skipping Grad-CAM.")
                    continue

                sample_idx = indices[0]
                img = images[sample_idx:sample_idx+1]
                feature_input = valid_features[sample_idx:sample_idx+1]
                image_id = sample_ids[sample_idx.numpy()]  # Use sample_ids instead of valid_sample_ids

                pred = meta_classification_model.predict(feature_input, batch_size=1, verbose=0)
                pred_class = tf.argmax(pred, axis=1, output_type=tf.int32)[0].numpy()

                # Load original (unprocessed) image
                original_unprocessed = load_original_image(image_id, extract_dir)
                if original_unprocessed is None:
                    logging.warning(f"Could not find unprocessed original image for ID {image_id}, skipping saving original image.")
                else:
                    original_unprocessed_path = os.path.join(class_dir, f"sample_label_{label}_original_unprocessed.png")
                    cv2.imwrite(original_unprocessed_path, cv2.cvtColor(original_unprocessed, cv2.COLOR_RGB2BGR))
                    print(f"Saved unprocessed original image for label {label} at {original_unprocessed_path}")

                for model_name in model_configs.keys():
                    config = model_configs[model_name]
                    # Load model with weights
                    model = load_model_from_config(
                        config['config_path'],
                        config['weights_path'],
                        config['base_model']
                    )
                    model.preprocess_input = config['preprocess']
                    features_4d = features_4d_dict[model_name.lower()][sample_idx:sample_idx+1]

                    heatmap = compute_gradcam_4d(
                        model=model,
                        img_array=img,
                        feature_4d=features_4d,
                        class_idx=pred_class,
                        layer_name=config['feature_layer_name'],
                        img_size=(config['img_size'], config['img_size'])
                    )

                    if heatmap is not None:
                        # Convert processed image to uint8 for saving
                        original_img = (img[0] * 255).astype(np.uint8)
                        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

                        original_path = os.path.join(class_dir, f"sample_label_{label}_processed.png")
                        heatmap_path = os.path.join(class_dir, f"sample_label_{label}_heatmap_{model_name}.png")
                        gradcam_path = os.path.join(class_dir, f"sample_label_{label}_gradcam_true_{label}_pred_{pred_class}_{model_name}.png")

                        cv2.imwrite(original_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(heatmap_path, heatmap)
                        cv2.imwrite(gradcam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

                        print(f"Saved processed image for label {label} at {original_path}")
                        print(f"Saved heatmap for label {label} at {heatmap_path}")
                        print(f"Saved Grad-CAM for label {label} ({model_name}) at: {gradcam_path}")
                    else:
                        print(f"Could not generate Grad-CAM for label: {label} ({model_name})")

                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()

        # Plot training history
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(history['qwk'], label='QWK')
        plt.title('Quadratic Weighted Kappa')
        plt.xlabel('Episode')
        plt.ylabel('QWK')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(history['support_loss'], label='Support Loss')
        plt.plot(history['query_loss'], label='Query Loss')
        plt.title('Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(history['support_accuracy'], label='Support Accuracy')
        plt.plot(history['query_accuracy'], label='Query Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(history['precision'], label='Precision')
        plt.plot(history['recall'], label='Recall')
        plt.title('Precision & Recall')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        history_plot_path = os.path.join(feature_save_dir, 'training_history.png')
        plt.savefig(history_plot_path)
        plt.close()
        print(f"\nSaved training history plot at: {history_plot_path}")

    return meta_model, meta_classification_model, feature_model, history

def validate_labels(labels, name="labels"):
    # Convert labels to tensor if not a NumPy array
    if not isinstance(labels, np.ndarray):
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        # Check for NaN/inf using TensorFlow
        if tf.reduce_any(tf.math.is_nan(labels)):
            raise ValueError(f"{name} contains NaN values.")
        if tf.reduce_any(tf.math.is_inf(labels)):
            raise ValueError(f"{name} contains inf values.")
        label_sums = tf.reduce_sum(labels, axis=1)
        if not tf.reduce_all(tf.logical_and(label_sums >= 0.9, label_sums <= 1.1)):
            raise ValueError(f"{name} is not in valid one-hot format.")
        if tf.reduce_any(labels < 0.0) or tf.reduce_any(labels > 1.0):
            raise ValueError(f"{name} contains values outside [0.0, 1.0].")
        # Convert to NumPy array for return
        labels_np = labels.numpy()
    else:
        labels_np = labels
        # Check for NaN/inf using NumPy
        if np.any(np.isnan(labels_np)):
            raise ValueError(f"{name} contains NaN values.")
        if np.any(np.isinf(labels_np)):
            raise ValueError(f"{name} contains inf values.")
        label_sums = np.sum(labels_np, axis=1)
        if not np.all(np.isclose(label_sums, 1.0, rtol=1e-5)):
            raise ValueError(f"{name} is not in valid one-hot format.")
        if np.any(labels_np < 0) or np.any(labels_np > 1):
            raise ValueError(f"{name} contains values outside [0, 1].")

    # Check label distribution
    label_indices = np.argmax(labels_np, axis=1)
    class_counts = np.bincount(label_indices, minlength=NUM_CLASSES)
    logging.info(f"{name} distribution: {dict(zip(range(NUM_CLASSES), class_counts))}")
    if np.all(class_counts == 0):
        raise ValueError(f"No samples found in {name} for any class.")
    return labels_np

# In the main section
if __name__ == "__main__":
    n_episodes = 20
    NUM_CLASSES = 5

    # Validate and normalize sample IDs
    logging.info("Validating sample IDs before combining features")
    if isinstance(train_x, (pd.Series, pd.DataFrame)):
        train_x = train_x.values.tolist()
    if isinstance(valid_x, (pd.Series, pd.DataFrame)):
        valid_x = valid_x.values.tolist()
    if isinstance(test_x, (pd.Series, pd.DataFrame)):
        logging.warning(f"Invalid type for test_x: {type(test_x)}. Converting to list.")
        test_x = test_x.values.tolist()

    # Check initial data lengths
    logging.info(f"train_x length: {len(train_x)}, balanced_train_y_multi shape: {balanced_train_y_multi.shape}")
    logging.info(f"valid_x length: {len(valid_x)}, valid_y_multi shape: {valid_y_multi.shape}")
    logging.info(f"test_x length: {len(test_x)}, test_y_multi shape: {test_y_multi.shape}")

    # Extract 2D features
    features_2d_dict_train = {}
    features_2d_dict_valid = {}
    features_2d_dict_test = {}
    for model_name, config in model_configs.items():
        img_size = config.get('img_size', 224)
        logging.info(f"Extracting 2D features for model: {model_name} with img_size={img_size}")

        train_generator = My_Generator(
            images=balanced_train_x,
            labels=balanced_train_y_multi,
            batch_size=64,
            is_train=True,
            model_type=config['model_type'],
            preprocess=config['preprocess'],
            size1=img_size,
            size2=img_size
        )
        valid_generator = My_Generator(
            images=resized_valid_x,
            labels=valid_y_multi,
            batch_size=64,
            is_train=False,
            model_type=config['model_type'],
            preprocess=config['preprocess'],
            size1=img_size,
            size2=img_size
        )
        test_generator = My_Generator(
            images=resized_test_x,
            labels=test_y_multi,
            batch_size=64,
            is_train=False,
            model_type=config['model_type'],
            preprocess=config['preprocess'],
            size1=img_size,
            size2=img_size
        )

        features_2d_dict_train[model_name] = extract_2d_features(
            model_name, config, train_generator, meta_save_dir, train_x
        )
        features_2d_dict_valid[model_name] = extract_2d_features(
            model_name, config, valid_generator, meta_save_dir, valid_x
        )
        features_2d_dict_test[model_name] = extract_2d_features(
            model_name, config, test_generator, meta_save_dir, test_x
        )

    # Load 4D features
    features_4d_dict_train = {}
    features_4d_dict_valid = {}
    features_4d_dict_test = {}
    for model_name in model_configs.keys():
        try:
            features_4d_dict_train[model_name.lower()] = load_4d_features(model_name.lower(), 'train')
            features_4d_dict_valid[model_name.lower()] = load_4d_features(model_name.lower(), 'valid')
            features_4d_dict_test[model_name.lower()] = load_4d_features(model_name.lower(), 'test')
        except FileNotFoundError:
            logging.error(f"4D feature file for {model_name.lower()} not found. Please generate 4D features first.")
            raise

    # Check for NaN/inf in 4D features
    logging.info("Checking for NaN/inf in 4D train features")
    for model_name in features_4d_dict_train:
        features_4d = features_4d_dict_train[model_name]
        if np.any(np.isnan(features_4d)) or np.any(np.isinf(features_4d)):
            logging.error(f"4D features for {model_name} contain NaN/inf.")
            raise ValueError(f"4D features for {model_name} contain NaN/inf.")

    # Synchronize number of samples
    n_samples_train = min(
        balanced_train_x.shape[0],
        balanced_train_y_multi.shape[0],
        len(train_x),
        *[features_2d_dict_train[model_name].shape[0] for model_name in features_2d_dict_train],
        *[features_4d_dict_train[model_name].shape[0] for model_name in features_4d_dict_train]
    )
    logging.info(f"Number of train samples after synchronization: {n_samples_train}")

    balanced_train_x = balanced_train_x[:n_samples_train]
    balanced_train_y_multi = balanced_train_y_multi[:n_samples_train]
    train_x = train_x[:n_samples_train]
    for model_name in features_2d_dict_train:
        features_2d_dict_train[model_name] = features_2d_dict_train[model_name][:n_samples_train]
    for model_name in features_4d_dict_train:
        features_4d_dict_train[model_name] = features_4d_dict_train[model_name][:n_samples_train]

    n_samples_valid = min(
        resized_valid_x.shape[0],
        valid_y_multi.shape[0],
        len(valid_x),
        *[features_2d_dict_valid[model_name].shape[0] for model_name in features_2d_dict_valid],
        *[features_4d_dict_valid[model_name].shape[0] for model_name in features_4d_dict_valid]
    )
    logging.info(f"Number of valid samples after synchronization: {n_samples_valid}")

    resized_valid_x = resized_valid_x[:n_samples_valid]
    valid_y_multi = valid_y_multi[:n_samples_valid]
    valid_x = valid_x[:n_samples_valid]
    for model_name in features_2d_dict_valid:
        features_2d_dict_valid[model_name] = features_2d_dict_valid[model_name][:n_samples_valid]
    for model_name in features_4d_dict_valid:
        features_4d_dict_valid[model_name] = features_4d_dict_valid[model_name][:n_samples_valid]

    n_samples_test = min(
        resized_test_x.shape[0],
        test_y_multi.shape[0],
        len(test_x),
        *[features_2d_dict_test[model_name].shape[0] for model_name in features_2d_dict_test],
        *[features_4d_dict_test[model_name].shape[0] for model_name in features_4d_dict_test]
    )
    logging.info(f"Number of test samples after synchronization: {n_samples_test}")

    resized_test_x = resized_test_x[:n_samples_test]
    test_y_multi = test_y_multi[:n_samples_test]
    test_x = test_x[:n_samples_test]
    for model_name in features_2d_dict_test:
        features_2d_dict_test[model_name] = features_2d_dict_test[model_name][:n_samples_test]
    for model_name in features_4d_dict_test:
        features_4d_dict_test[model_name] = features_4d_dict_test[model_name][:n_samples_test]

    # Check for NaN/inf in 2D features
    logging.info("Checking for NaN/inf in train features")
    for model_name in features_2d_dict_train:
        features_2d = features_2d_dict_train[model_name]
        if np.any(np.isnan(features_2d)) or np.any(np.isinf(features_2d)):
            logging.error(f"2D features for {model_name} contain NaN/inf.")
            raise ValueError(f"2D features for {model_name} contain NaN/inf.")

    # Validate and normalize labels
    logging.info("Checking for NaN/inf and one-hot format for labels")
    train_labels_np = validate_labels(balanced_train_y_multi, "balanced_train_y_multi")
    valid_labels_np = validate_labels(valid_y_multi, "valid_y_multi")
    test_labels_np = validate_labels(test_y_multi, "test_y_multi")

    # Combine and reduce features
    try:
        train_features, pca, train_valid_indices, input_dim, train_labels, train_sample_ids = combine_and_reduce_features(
            features_2d_dict_train, features_4d_dict_train, balanced_train_y_multi, train_x, meta_save_dir
        )
        valid_features, _, valid_valid_indices, _, valid_labels, valid_sample_ids = combine_and_reduce_features(
            features_2d_dict_valid, features_4d_dict_valid, valid_y_multi, valid_x, meta_save_dir
        )
        test_features, _, test_valid_indices, _, test_labels, test_sample_ids = combine_and_reduce_features(
            features_2d_dict_test, features_4d_dict_test, test_y_multi, test_x, meta_save_dir
        )
    except ValueError as e:
        logging.error(f"Error in combine_and_reduce_features: {str(e)}")
        raise

    # Validate labels after combine_and_reduce_features
    logging.info("Validating train_labels, valid_labels, and test_labels before training")
    train_labels_np = validate_labels(train_labels, "train_labels")
    valid_labels_np = validate_labels(valid_labels, "valid_labels")
    test_labels_np = validate_labels(test_labels, "test_labels")

    # Check data dimensions
    logging.info(f"Train: features={train_features.shape}, labels={train_labels.shape}, sample_ids={len(train_sample_ids)}")
    logging.info(f"Valid: features={valid_features.shape}, labels={valid_labels.shape}, sample_ids={len(valid_sample_ids)}")
    logging.info(f"Test: features={test_features.shape}, labels={test_labels.shape}, sample_ids={len(test_sample_ids)}")

    # Check alignment
    if train_features.shape[0] != train_labels.shape[0] or train_features.shape[0] != len(train_sample_ids):
        logging.error(f"Alignment mismatch: train_features ({train_features.shape[0]}), train_labels ({train_labels.shape[0]}), train_sample_ids ({len(train_sample_ids)})")
        raise ValueError("Alignment mismatch for train data")
    if valid_features.shape[0] != valid_labels.shape[0] or valid_features.shape[0] != len(valid_sample_ids):
        logging.error(f"Alignment mismatch: valid_features ({valid_features.shape[0]}), valid_labels ({valid_labels.shape[0]}), valid_sample_ids ({len(valid_sample_ids)})")
        raise ValueError("Alignment mismatch for valid data")
    if test_features.shape[0] != test_labels.shape[0] or test_features.shape[0] != len(test_sample_ids):
        logging.error(f"Alignment mismatch: test_features ({test_features.shape[0]}), test_labels ({test_labels.shape[0]}), test_sample_ids ({len(test_sample_ids)})")
        raise ValueError("Alignment mismatch for test data")

    # Print label distribution before training
    print("Shape features:", train_features.shape)
    print("Shape labels:", train_labels.shape)
    label_indices = np.argmax(train_labels_np, axis=1)
    class_counts = np.bincount(label_indices, minlength=NUM_CLASSES)
    print("Label distribution:")
    for cls, count in zip(range(NUM_CLASSES), class_counts):
        print(f"Class {cls}: {count} samples")

    # Final checks
    assert train_features.shape[0] == train_labels.shape[0], "Mismatch between number of train features and labels!"
    assert valid_features.shape[0] == valid_labels.shape[0], "Mismatch between number of valid features and labels!"
    assert test_features.shape[0] == test_labels.shape[0], "Mismatch between number of test features and labels!"

    train_features, train_labels_np = oversample_minority_classes(train_features, train_labels_np, min_count=100)

    # Call maml_fomaml_train_manual
    try:
        meta_model, meta_classification_model, feature_model, history = maml_fomaml_train_manual(
            features=train_features,
            labels=train_labels_np,
            valid_features=valid_features,
            valid_labels=valid_labels,
            input_dim=input_dim,
            n_episodes=n_episodes,
            n_support=20,
            n_query=20,
            inner_lr=0.001,
            outer_lr=0.001,
            fine_tune_lr=0.0002,
            sample_ids=valid_sample_ids,
            images=resized_test_x,
            features_4d_dict=features_4d_dict_test
        )
    except Exception as e:
        logging.error(f"Error in maml_fomaml_train_manual: {str(e)}")
        raise

    # Evaluate on test set
    try:
        test_preds = meta_classification_model.predict(test_features, batch_size=32)
        test_probs = np.max(test_preds, axis=1)
        test_probs = np.clip(test_probs, 0.0, 1.0)
        test_preds_classes = np.argmax(test_preds, axis=1)
        test_true_classes = np.argmax(test_labels, axis=1)
        test_qwk = cohen_kappa_score(test_true_classes, test_preds_classes,
                                     labels=list(range(NUM_CLASSES)), weights='quadratic')
        test_f1_score = f1_score(test_true_classes, test_preds_classes, average='weighted')
        test_recall_score = recall_score(test_true_classes, test_preds_classes, average='weighted')
        test_precision_score = precision_score(test_true_classes, test_preds_classes, average='weighted')

        print(f"\nResults on test set:")
        print(f"  Quadratic Weighted Kappa (QWK): {test_qwk:.4f}")
        print(f"  Weighted F1 Score: {test_f1_score:.4f}")
        print(f"  Weighted Recall: {test_recall_score:.4f}")
        print(f"  Weighted Precision: {test_precision_score:.4f}")

        cm_test = save_confusion_matrix(
            test_true_classes, test_preds_classes, n_episodes, test_qwk, feature_save_dir, prefix='test_'
        )
        print(f"Confusion matrix on test set:\n{cm_test}")

        test_metrics = {
            "test_qwk": float(test_qwk),
            "test_f1": float(test_f1_score),
            "test_recall": float(test_recall_score),
            "test_precision": float(test_precision_score),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        test_metrics_filepath = os.path.join(feature_save_dir, 'test_metrics.json')
        with open(test_metrics_filepath, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        logging.info(f"Saved test set metrics at: {test_metrics_filepath}")
    except Exception as e:
        logging.error(f"Error evaluating on test set: {str(e)}")
        raise
