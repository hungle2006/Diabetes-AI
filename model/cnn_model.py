import os
import cv2
import numpy as np
import pandas as pd
import glob
import zipfile
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3, DenseNet121, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                                    BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, recall_score, cohen_kappa_score, confusion_matrix, precision_score, accuracy_score
import albumentations as A
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
from datetime import datetime
import json
import shutil
from sklearn.utils import resample
import subprocess
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2
import albumentations as A
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, Xception, InceptionV3, ResNet50, DenseNet121
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet121_preprocess
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, confusion_matrix, precision_score, accuracy_score
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Layer, Dropout
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
import logging
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from google.colab import drive
import time
import subprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Path to data
drive_folder = "/content/drive/MyDrive/kaggle_data/aptos2019"
extract_root = "/content/extracted_zip_files"
os.makedirs(extract_root, exist_ok=True)

# Path to data and unzip the ZIP files if not already extracted (skip if already extracted)
zip_files = glob.glob(os.path.join(drive_folder, "*.zip"))
for zip_path in zip_files:
    zip_name = os.path.basename(zip_path).replace(".zip", "")
    extract_path = os.path.join(extract_root, zip_name)
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✅ Successfully extracted: {zip_path} → {extract_path}")

# 3. Read CSV file
df_train = pd.read_csv(os.path.join(drive_folder, "train.csv"))
df_test = pd.read_csv(os.path.join(drive_folder, "test.csv"))

# 4. Define image processing functions: crop, resize, and augment images
def crop_image_from_gray_to_color(img, tol=7):
    """
    Crop unnecessary areas (especially dark edges) of the image based on the grayscale image,
    then apply the crop to the original color image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    cropped_img = img[np.ix_(rows, cols)]
    return cropped_img

def load_ben_color(path, sigmaX=10, IMG_SIZE=244):
    """
    Load an image from the path, crop dark borders based on the grayscale image, resize, and augment with GaussianBlur.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image from path: {path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Crop image based on bright areas
    image = crop_image_from_gray_to_color(image, tol=7)
    # Resize image to desired size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    # Augment image with GaussianBlur and weighted addition
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

# 5. Process and save processed images to a temporary directory on Colab
train_img_folder = os.path.join(extract_root, "train_images")  # Directory containing original images
processed_folder = "/content/processed_train_images"           # Directory to save processed images
os.makedirs(processed_folder, exist_ok=True)

processed_ids = []  # Store IDs of successfully processed images

for idx, row in df_train.iterrows():
    img_filename = f"{row['id_code']}.png"
    img_path = os.path.join(train_img_folder, img_filename)

    try:
        proc_img = load_ben_color(img_path, sigmaX=10, IMG_SIZE=244)
        # cv2.imwrite saves images in BGR format, so convert from RGB to BGR
        proc_img_bgr = cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(processed_folder, img_filename)
        cv2.imwrite(save_path, proc_img_bgr)
        processed_ids.append(row['id_code'])
    except Exception as e:
        print(f"Error processing image {img_filename}: {e}")

print(f"Successfully processed {len(processed_ids)} images.")

# 6. Update DataFrame with only successfully processed images
df_train_processed = df_train[df_train['id_code'].isin(processed_ids)].copy()

# 7. Split data into train and validation sets based on the CSV file
x = df_train_processed['id_code']
y = df_train_processed['diagnosis']

# Shuffle data to ensure randomness
x, y = shuffle(x, y, random_state=42)

# Split into train+validation and test sets (80% - 20%)
x_temp, test_x, y_temp, test_y = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)

# Split into train and validation sets (85% train, 15% val from the 80% initial data)
train_x, valid_x, train_y, valid_y = train_test_split(x_temp, y_temp, test_size=0.15/0.80, stratify=y_temp, random_state=42)

# Print information for verification
print("Train X size:", len(train_x))
print("Train y size:", len(train_y))
print("Valid X size:", len(valid_x))
print("Valid y size:", len(valid_y))
print("Test X size:", len(test_x))
print("Test y size:", len(test_y))

# General configuration
WORKERS = 2
CHANNEL = 3
SIZE = 224
NUM_CLASSES = 5

# Convert labels to one-hot
if len(train_y.shape) == 1 or train_y.shape[1] != NUM_CLASSES:
    train_y_multi = to_categorical(train_y, num_classes=NUM_CLASSES)
    valid_y_multi = to_categorical(valid_y, num_classes=NUM_CLASSES)
    test_y_multi = to_categorical(test_y, num_classes=NUM_CLASSES)
else:
    train_y_multi = train_y
    valid_y_multi = valid_y
    test_y_multi = test_y

# Define My_Generator
class My_Generator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size, is_train=False,
                 mix=False, augment=False, size1=224, size2=299, model_type="default",
                 balance_classes=False):
        self.image_filenames = np.array(image_filenames)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        self.is_mix = mix
        self.model_type = str(model_type).lower()
        self.n_classes = self.labels.shape[1] if self.labels.ndim > 1 else int(max(self.labels) + 1)
        if "inceptionv3" in self.model_type or "xception" in self.model_type:
            self.target_size = (size2, size2)
        else:
            self.target_size = (size1, size1)
        self.base_path = "/content/processed_train_images/"
        if self.is_augment and self.is_train:
            self.augmenter = A.Compose([
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=1),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1),
                    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.1, p=1)
                ], p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.CropAndPad(percent=(-0.1, 0), p=0.5)
            ])
        self.rare_augmenter = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
        ])
        self.class_counts = self._compute_initial_class_counts()
        self.augmented_class_counts = self.class_counts.copy()
        self.class_weights = None
        if self.is_train and balance_classes:
            self.balance_class()
        if self.is_train:
            self.on_epoch_end()

    def _compute_initial_class_counts(self):
        labels = np.argmax(self.labels, axis=1) if self.labels.ndim > 1 else self.labels
        return np.bincount(labels, minlength=self.n_classes)

    def _compute_class_weights(self):
        total_samples = np.sum(self.augmented_class_counts)
        if total_samples == 0:
            return np.ones(self.n_classes)
        class_weights = total_samples / (self.n_classes * self.augmented_class_counts)
        class_weights = np.where(np.isinf(class_weights) | (self.augmented_class_counts == 0), 1.0, class_weights)
        return class_weights / np.min(class_weights[np.isfinite(class_weights)])

    def get_class_weights(self):
        """Return the current class weights for training."""
        return self.class_weights

    def balance_classes(self):
        class_counts = self._compute_initial_class_counts()
        max_count = class_counts[0]  # Use the number of samples in class 0 as the target

        print(f"Initial sample counts: {class_counts}")
        print(f"Target sample count per class (based on class 0): {max_count}")

        new_filenames = []
        new_labels = []
        for cls in range(self.n_classes):
            current_count = class_counts[cls]
            if current_count == 0:
                print(f"Class {cls} has no samples, skipping.")
                continue
            if cls == 3:
                target_count = int(max_count * 1.3)  # Class 3 is increased by 30%
            else:
                target_count = max_count
            if current_count < target_count:
                samples_to_add = target_count - current_count
                label_indices = np.argmax(self.labels, axis=1) if self.labels.ndim > 1 else self.labels
                class_indices = np.where(label_indices == cls)[0]
                for i in range(samples_to_add):
                    idx = np.random.choice(class_indices)
                    img_id = self.image_filenames[idx]
                    label = self.labels[idx]
                    img = self._load_image(img_id)
                    if img is None:
                        continue
                    aug_img = self.rare_augmenter(image=img)['image']
                    new_img_id = f"{img_id}_balance_aug_{i}"
                    save_path = os.path.join(self.base_path, f"{new_img_id}.png")
                    if cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)):
                        new_filenames.append(new_img_id)
                        new_labels.append(np.array(label, dtype=self.labels.dtype))
                        self.augmented_class_counts[cls] += 1
                    else:
                        print(f"Error saving augmented image {new_img_id}")

        if new_labels:
            new_labels_array = np.array(new_labels)
            if new_labels_array.ndim == 1:
                new_labels_array = new_labels_array[:, np.newaxis]
            self.image_filenames = np.concatenate([self.image_filenames, new_filenames])
            self.labels = np.concatenate([self.labels, new_labels_array])

        # Do not call on_epoch_end() here to avoid recalculating class weights immediately

        # Check sample counts after balancing
        updated_counts = self._compute_initial_class_counts()
        print(f"Sample counts after balancing and augmenting class 3: {updated_counts}")

    def augment_weak_classes(self, weak_classes, augment_factor=2):
        new_filenames = []
        new_labels = []
        for idx, label in enumerate(self.labels):
            label_class = np.argmax(label) if label.ndim > 1 else label
            if np.isscalar(label_class) and np.isin(label_class, weak_classes):
                img_id = self.image_filenames[idx]
                img = self._load_image(img_id)
                if img is None:
                    continue
                for i in range(augment_factor):
                    aug_img = self.rare_augmenter(image=img)['image']
                    new_img_id = f"{img_id}_weak_aug_{i}"
                    save_path = os.path.join(self.base_path, f"{new_img_id}.png")
                    if cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)):
                        new_filenames.append(new_img_id)
                        new_labels.append(np.array(label, dtype=self.labels.dtype))
                        self.augmented_class_counts[label_class] += 1
                    else:
                        print(f"Error saving augmented image {new_img_id}")
        if new_labels:
            new_labels_array = np.array(new_labels)
            if new_labels_array.ndim == 1:
                new_labels_array = new_labels_array[:, np.newaxis]
            self.image_filenames = np.concatenate([self.image_filenames, new_filenames])
            self.labels = np.concatenate([self.labels, new_labels_array])

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self._generate_batch(batch_x, batch_y, augment=self.is_train)

    def on_epoch_end(self):
        if self.is_train:
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
            # Calculate class weights at the end of each epoch
            self.class_weights = self._compute_class_weights()
            print(f"Class weights after epoch: {self.class_weights}")
            print(f"Augmented sample counts: {self.augmented_class_counts}")

    def _load_image(self, img_id):
        img_path = os.path.join(self.base_path, f"{img_id}.png")
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Image not found or corrupted: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
            return img
        except Exception as e:
            print(f"Error loading image {img_id}: {str(e)}")
            return None

    def _generate_batch(self, batch_x, batch_y, augment=False):
        batch_images = []
        valid_labels = []

        for img_id, label in zip(batch_x, batch_y):
            img = self._load_image(img_id)
            if img is None:
                continue
            if augment and self.is_augment:
                img = self.augmenter(image=img.astype(np.uint8))['image']
            img = img.astype(np.float32) / 255.0

            if "resnet50" in self.model_type:
                img = resnet50_preprocess(img)
            elif "efficientnetb0" in self.model_type:
                img = efficientnet_preprocess(img)
            elif "inceptionv3" in self.model_type:
                img = inception_preprocess(img)
            elif "densenet121" in self.model_type:
                img = densenet_preprocess(img)
            elif "xception" in self.model_type:
                img = xception_preprocess(img)

            batch_images.append(img)
            valid_labels.append(label)

        if not batch_images:
            return np.zeros((1, *self.target_size, 3), dtype=np.float32), np.zeros((1, *batch_y.shape[1:]), dtype=np.float32)

        batch_images = np.array(batch_images)
        valid_labels = np.array(valid_labels)

        if self.is_mix and len(batch_images) > 1:
            batch_images, valid_labels = self._mixup(batch_images, valid_labels)

        return batch_images, valid_labels

    def _mixup(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        index = np.random.permutation(len(x))
        mixed_x = np.zeros_like(x)
        mixed_y = np.zeros_like(y)
        for i in range(len(x)):
            if np.argmax(y[i]) == np.argmax(y[index[i]]):
                mixed_x[i] = lam * x[i] + (1 - lam) * x[index[i]]
                mixed_y[i] = y[i]
            else:
                mixed_x[i] = x[i]
                mixed_y[i] = y[i]
        return mixed_x, mixed_y

# Function to create model
def create_model(input_shape, n_out, model_type, weights_path=None, weights="imagenet"):
    input_tensor = Input(shape=input_shape)
    if model_type == "resnet50":
        base_model = ResNet50(include_top=False, weights=weights if not weights_path else None, input_tensor=input_tensor)
    elif model_type == "efficientnetb0":
        base_model = EfficientNetB0(include_top=False, weights=weights if not weights_path else None, input_tensor=input_tensor)
    elif model_type == "inceptionv3":
        base_model = InceptionV3(include_top=False, weights=weights if not weights_path else None, input_tensor=input_tensor)
    elif model_type == "densenet121":
        base_model = DenseNet121(include_top=False, weights=weights if not weights_path else None, input_tensor=input_tensor)
    elif model_type == "xception":
        base_model = Xception(include_top=False, weights=weights if not weights_path else None, input_tensor=input_tensor)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    if weights_path:
        try:
            base_model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights from {weights_path}: {e}")
            raise
    x = GlobalAveragePooling2D(name='global_avg_pool')(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    final_output = Dense(n_out, activation="softmax", name='final_output')(x)
    model = Model(input_tensor, final_output)
    return model

# Function to get the last convolutional layer
def get_last_conv_layer(model, model_type):
    """Identify the last convolutional layer based on the model type."""
    model_type = model_type.lower()
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower() or 'block' in layer.name.lower() or 'mixed' in layer.name.lower():
            if len(layer.output.shape) == 4:
                print(f"Last convolutional layer for {model_type}: {layer.name}")
                return layer.name
    raise ValueError(f"No 4D convolutional layer found for model {model_type}")

# Function to compute Grad-CAM
def compute_gradcam(model, img_array, last_conv_layer_name, class_index, model_type):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    target_size = (299, 299) if "inceptionv3" in model_type.lower() or "xception" in model_type.lower() else (224, 224)
    heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = img_array[0]
    if "resnet50" in model_type.lower():
        img = (img + 1) * 127.5
    elif "inceptionv3" in model_type.lower():
        img = (img + 1) * 127.5
    elif "xception" in model_type.lower():
        img = (img + 1) * 127.5
    elif "densenet121" in model_type.lower():
        img = (img + 1) * 127.5
    else:
        img = img * 255.0
    img = np.uint8(img)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0.0)
    return heatmap, superimposed_img

# Function to extract 2D and 4D features for meta-learning
def extract_features(model, generator, steps, model_type, save_path):
    """Extract 2D and 4D features from the model."""
    # Get the name of the last convolutional layer
    last_conv_layer_name = get_last_conv_layer(model, model_type)

    # Create feature model
    feature_model = Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,  # 4D features
            model.get_layer('global_avg_pool').output      # 2D features
        ]
    )

    features_4d = []
    features_2d = []
    labels = []

    # Extract features from generator
    for i in range(steps):
        batch_images, batch_labels = generator[i]
        batch_features_4d, batch_features_2d = feature_model.predict(batch_images, verbose=0)
        features_4d.append(batch_features_4d)
        features_2d.append(batch_features_2d)
        labels.append(batch_labels)

    # Convert to NumPy arrays
    features_4d = np.concatenate(features_4d, axis=0)
    features_2d = np.concatenate(features_2d, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Save features
    np.savez(save_path, features_4d=features_4d, features_2d=features_2d, labels=labels)
    print(f"Saved features at {save_path}: 4D shape {features_4d.shape}, 2D shape {features_2d.shape}")
    return features_4d, features_2d, labels

# Model configuration
model_configs = {
    # "xception": {
    #     "model_type": "xception",
    #     "weights": "imagenet",
    #     "save_path": "/content/drive/MyDrive/working/Xception_bestqwk_aptos.h5"
    # },
    # "resnet50": {
    #     "model_type": "resnet50",
    #     "weights_path": "/content/drive/MyDrive/keras_weights/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
    #     "save_path": "/content/drive/MyDrive/working/ResNet50_bestqwk_aptos.h5"
    # },
    # "efficientnetb0": {
    #     "model_type": "efficientnetb0",
    #     "weights": "imagenet",
    #     "save_path": "/content/drive/MyDrive/working/EfficientNetB0_bestqwk_aptos.h5"
    # },
    # "inceptionv3": {
    #     "model_type": "inceptionv3",
    #     "weights_path": "/content/drive/MyDrive/keras_weights/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
    #     "save_path": "/content/drive/MyDrive/working/InceptionV3_bestqwk_aptos.h5"
    # },
    "densenet121": {
        "model_type": "densenet121",
        "weights_path": "/content/drive/MyDrive/keras_weights/densenet121/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "save_path": "/content/drive/MyDrive/working/DenseNet121_bestqwk_aptos.h5"
    }
}

# Callback classes
class QWKEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=(), batch_size=32, interval=1, model_type=None, save_paths=None):
        super().__init__()
        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []
        self.model_type = model_type
        self.save_paths = save_paths if save_paths is not None else {}
        self.save_path = self.save_paths.get(model_type, None)
        self.best_qwk = -float('inf')
        self.best_y_true = None
        self.best_y_pred = None

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            steps = int(np.ceil(len(self.y_val) / self.batch_size))
            y_pred = self.model.predict(self.valid_generator, steps=steps, verbose=1)
            if len(self.y_val.shape) > 1 and self.y_val.shape[1] > 1:
                y_true = np.argmax(self.y_val, axis=1)
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                y_true = self.y_val.astype(int)
                y_pred_classes = np.argmax(y_pred, axis=1)
            score = cohen_kappa_score(y_true, y_pred_classes, labels=[0, 1, 2, 3, 4], weights='quadratic')
            print(f"\nEpoch {epoch+1} - QWK: {score:.4f}")
            f1 = f1_score(y_true, y_pred_classes, average=None, labels=[0, 1, 2, 3, 4])
            sensitivity = recall_score(y_true, y_pred_classes, average=None, labels=[0, 1, 2, 3, 4])
            print(f"F1-score per class: {f1}")
            print(f"Sensitivity per class: {sensitivity}")
            self.history.append(score)
            if score > self.best_qwk:
                self.best_qwk = score
                self.best_y_true = y_true
                self.best_y_pred = y_pred_classes
                print(f"New best QWK: {self.best_qwk:.4f} at Epoch {epoch+1}")
                if self.save_path:
                    keras_save_path = self.save_path.replace('.h5', '.keras')
                    save_dir = os.path.dirname(keras_save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    self.model.save(keras_save_path, overwrite=True)
                    print(f"Saved (overwritten) full model to {keras_save_path}")
                    save_dir = self.save_path.replace('.h5', '')
                    os.makedirs(save_dir, exist_ok=True)
                    model_json = self.model.to_json()
                    config_path = os.path.join(save_dir, "config.json")
                    with open(config_path, "w") as json_file:
                        json_file.write(model_json)
                    print(f"Saved model architecture to {config_path}")
                    weights_path = os.path.join(save_dir, "model.weights.h5")
                    self.model.save_weights(weights_path)
                    print(f"Saved model weights to {weights_path}")
                    metadata = {
                        "keras_version": tf.keras.__version__,
                        "save_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model_type": self.model_type
                    }
                    metadata_path = os.path.join(save_dir, "metadata.json")
                    with open(metadata_path, "w") as meta_file:
                        json.dump(metadata, meta_file)
                    print(f"Saved metadata to {metadata_path}")

class QWKReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, qwk_callback, factor=0.5, patience=3, min_lr=1e-6, verbose=1):
        super().__init__()
        self.qwk_callback = qwk_callback
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_qwk = -float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_qwk = self.qwk_callback.history[-1] if self.qwk_callback.history else -float('inf')
        if current_qwk > self.best_qwk:
            self.best_qwk = current_qwk
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.model.optimizer.learning_rate)
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch+1}: QWKReduceLROnPlateau reducing learning rate to {new_lr:.6f}.")
                    self.wait = 0

class DynamicRareClassAugmentationCallback(Callback):
    def __init__(self, train_generator, valid_generator, valid_labels, threshold=0.6, augment_factor=2):
        super().__init__()
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.valid_labels = valid_labels
        self.threshold = threshold
        self.augment_factor = augment_factor
        self.f1_history = []
        self.batch_size = self.train_generator.batch_size
        self.num_classes = self.train_generator.n_classes

    def on_epoch_end(self, epoch, logs=None):
        steps = int(np.ceil(len(self.valid_labels) / self.batch_size))
        y_pred = self.model.predict(self.valid_generator, steps=steps, verbose=1)
        y_true = np.argmax(self.valid_labels, axis=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1_scores = f1_score(y_true, y_pred_classes, average=None, labels=list(range(self.num_classes)))
        print(f"F1-scores at epoch {epoch+1}: {f1_scores}")
        self.f1_history.append(f1_scores)
        weak_classes = [i for i, f1 in enumerate(f1_scores) if f1 < self.threshold]
        print(f"Weak classes at epoch {epoch+1} (F1 < {self.threshold}): {weak_classes}")
        if weak_classes:
            self.train_generator.augment_weak_classes(weak_classes, augment_factor=self.augment_factor)
            print(f"Augmented {self.augment_factor} samples for weak classes: {weak_classes}")
            self.train_generator.on_epoch_end()
            print(f"Updated class weights: {self.train_generator.get_class_weights()}")
        if len(self.f1_history) > 1:
            prev_f1 = self.f1_history[-2]
            curr_f1 = self.f1_history[-1]
            print(f"F1-score comparison (epoch {epoch} vs {epoch+1}):")
            for i in range(self.num_classes):
                print(f"Class {i}: {prev_f1[i]:.4f} -> {curr_f1[i]:.4f} (Change: {curr_f1[i] - prev_f1[i]:.4f})")

class LossHistoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

    def plot_and_save_loss(self, model_type, save_dir="/content/drive/MyDrive/working/"):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Training Loss', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', marker='s')
        plt.title(f'Training and Validation Loss - {model_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'loss_plot_{model_type}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved loss plot to {save_path}")

# Training and processing
batch_size = 32
resized_train_x = train_x.values
resized_valid_x = valid_x.values
resized_test_x = test_x.values

early_stopping = EarlyStopping(monitor='accuracy', patience=7, restore_best_weights=True, verbose=1, mode='max')

# Check data
assert 'train_x' in globals() and 'valid_x' in globals() and 'test_x' in globals(), "train_x, valid_x, or test_x not defined"
assert 'train_y' in globals() and 'valid_y' in globals() and 'test_y' in globals(), "train_y, valid_y, or test_y not defined"
assert callable(create_model), "create_model is not a function"
assert 'My_Generator' in globals(), "My_Generator not defined"

print("train_y shape:", train_y.shape)
print("valid_y shape:", valid_y.shape)
print("test_y shape:", test_y.shape)
print("train_y_multi shape:", train_y_multi.shape)
print("valid_y_multi shape:", valid_y_multi.shape)
print("test_y_multi shape:", test_y_multi.shape)

# Training loop
meta_features = {}
meta_save_dir = "/content/drive/MyDrive/working/meta_features_aptos"
os.makedirs(meta_save_dir, exist_ok=True)

for model_name, config in model_configs.items():
    print(f"\n==> Training model {model_name} ...")
    if "inceptionv3" in config["model_type"].lower() or "xception" in config["model_type"].lower():
        model_input_shape = (299, 299, 3)
        img_size = 299
    else:
        model_input_shape = (SIZE, SIZE, 3)
        img_size = SIZE
    train_generator = My_Generator(
        resized_train_x, train_y_multi, batch_size,
        is_train=True, mix=False, augment=True,
        size1=SIZE, size2=299, model_type=config["model_type"],
        balance_classes=True
    )
    try:
        print(f"Sample counts: {train_generator.augmented_class_counts}")
    except AttributeError:
        print("Unable to access augmented_class_counts, continuing training...")
    valid_generator = My_Generator(
        resized_valid_x, valid_y_multi, batch_size,
        is_train=False, size1=SIZE, size2=299, model_type=config["model_type"]
    )
    test_generator = My_Generator(
        resized_test_x, test_y_multi, batch_size,
        is_train=False, size1=SIZE, size2=299, model_type=config["model_type"]
    )
    class_weights = train_generator.get_class_weights()
    if class_weights is None:
        class_weights = np.ones(NUM_CLASSES)
    class_weight = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"Initial class weights: {class_weight}")
    weights_path = config.get("weights_path", None)
    pretrained_weights = config.get("weights", "imagenet")
    model = create_model(
        input_shape=model_input_shape,
        n_out=NUM_CLASSES,
        model_type=config["model_type"],
        weights_path=weights_path,
        weights=pretrained_weights
    )
    qwk_callback = QWKEvaluation(
        validation_data=(valid_generator, valid_y_multi),
        batch_size=batch_size,
        interval=1,
        model_type=config["model_type"],
        save_paths={config["model_type"]: config["save_path"]}
    )
    qwk_reduce_lr = QWKReduceLROnPlateau(
        qwk_callback=qwk_callback,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    loss_history = LossHistoryCallback()
    for layer in model.layers[:50]:
        layer.trainable = False
    for layer in model.layers[50:]:
        layer.trainable = True
    loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(
        loss=loss_fn,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    augment_callback = DynamicRareClassAugmentationCallback(
        train_generator=train_generator,
        valid_generator=valid_generator,
        valid_labels=valid_y_multi,
        threshold=0.6,
        augment_factor=2
    )
    model.fit(
        train_generator,
        steps_per_epoch=int(np.ceil(len(train_generator.image_filenames) / batch_size)),
        epochs=5,
        validation_data=valid_generator,
        validation_steps=int(np.ceil(len(valid_x) / batch_size)),
        verbose=1,
        callbacks=[qwk_callback, qwk_reduce_lr, early_stopping, augment_callback, loss_history]
    )
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    train_mixup = My_Generator(
        resized_train_x, train_y_multi, batch_size,
        is_train=True, mix=True, augment=True,
        size1=SIZE, size2=299, model_type=config["model_type"],
        balance_classes=True
    )
    try:
        print(f"Sample counts (mixup): {train_mixup.augmented_class_counts}")
    except AttributeError:
        print("Unable to access augmented_class_counts (mixup), continuing training...")
    augment_callback = DynamicRareClassAugmentationCallback(
        train_generator=train_mixup,
        valid_generator=valid_generator,
        valid_labels=valid_y_multi,
        threshold=0.6,
        augment_factor=2
    )
    epochs = 30
    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch + 1} for model {model_name}")
        class_weights = train_mixup.get_class_weights()
        if class_weights is None:
            class_weights = np.ones(NUM_CLASSES)
        class_weight = {i: float(w) for i, w in enumerate(class_weights)}
        print(f"Class weights for epoch {epoch + 1}: {class_weight}")
        model.fit(
            train_mixup,
            steps_per_epoch=int(np.ceil(len(train_mixup.image_filenames) / batch_size)),
            epochs=1,
            validation_data=valid_generator,
            validation_steps=int(np.ceil(len(valid_x) / batch_size)),
            verbose=1,
            callbacks=[qwk_callback, qwk_reduce_lr, augment_callback, early_stopping, loss_history],
            class_weight=class_weight
        )
    final_save_path = config["save_path"].replace('.h5', '_final.keras')
    model.save(final_save_path, overwrite=True)
    print(f"Saved final model at {final_save_path}")
    loss_history.plot_and_save_loss(config["model_type"], save_dir="/content/drive/MyDrive/working/")
    if qwk_callback.best_y_true is not None and qwk_callback.best_y_pred is not None:
        cm = confusion_matrix(qwk_callback.best_y_true, qwk_callback.best_y_pred, labels=[0, 1, 2, 3, 4])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[0, 1, 2, 3, 4],
                    yticklabels=[0, 1, 2, 3, 4])
        plt.title(f'Best Confusion Matrix - QWK: {qwk_callback.best_qwk:.4f} ({config["model_type"]})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        save_cm_path = f"/content/drive/MyDrive/working/best_confusion_matrix_aptos_{config['model_type']}.png"
        plt.savefig(save_cm_path)
        plt.close()
        print(f"Saved best confusion matrix to {save_cm_path}")
    else:
        print(f"No best QWK recorded for model {config['model_type']}, no plot generated.")

    # Extract 2D and 4D features for meta-learning
    print(f"\n==> Extracting 2D and 4D features for model {model_name} ...")
    train_steps = int(np.ceil(len(train_x) / batch_size))
    train_save_path = os.path.join(meta_save_dir, f"{model_name}_train_features.npz")
    train_features_4d, train_features_2d, train_labels = extract_features(
        model, train_generator, train_steps, config["model_type"], train_save_path
    )
    meta_features[f"{model_name}_train_2d"] = train_features_2d
    meta_features[f"{model_name}_train_4d"] = train_features_4d
    print(f"Extracted train features: 2D shape {train_features_2d.shape}, 4D shape {train_features_4d.shape}")

    valid_steps = int(np.ceil(len(valid_x) / batch_size))
    valid_save_path = os.path.join(meta_save_dir, f"{model_name}_valid_features.npz")
    valid_features_4d, valid_features_2d, valid_labels = extract_features(
        model, valid_generator, valid_steps, config["model_type"], valid_save_path
    )
    meta_features[f"{model_name}_valid_2d"] = valid_features_2d
    meta_features[f"{model_name}_valid_4d"] = valid_features_4d
    print(f"Extracted valid features: 2D shape {valid_features_2d.shape}, 4D shape {valid_features_4d.shape}")

    test_steps = int(np.ceil(len(test_x) / batch_size))
    test_save_path = os.path.join(meta_save_dir, f"{model_name}_test_features.npz")
    test_features_4d, test_features_2d, test_labels = extract_features(
        model, test_generator, test_steps, config["model_type"], test_save_path
    )
    meta_features[f"{model_name}_test_2d"] = test_features_2d
    meta_features[f"{model_name}_test_4d"] = test_features_4d
    print(f"Extracted test features: 2D shape {test_features_2d.shape}, 4D shape {test_features_4d.shape}")

    # Compute and save Grad-CAM on the test set
    print(f"\n==> Computing Grad-CAM for model {model_name} on the test set ...")
    last_conv_layer_name = get_last_conv_layer(model, config["model_type"])
    gradcam_save_dir = f"/content/drive/MyDrive/working/gradcam_aptos_{config['model_type']}"
    os.makedirs(gradcam_save_dir, exist_ok=True)
    num_samples = 5
    test_indices = np.random.choice(len(test_x), num_samples, replace=False)
    test_images = test_x.values[test_indices]
    test_sample_generator = My_Generator(
        test_images, test_y_multi[test_indices], batch_size,
        is_train=False, size1=SIZE, size2=299, model_type=config["model_type"]
    )
    batch_images, batch_labels = test_sample_generator[0]
    for i in range(min(num_samples, len(batch_images))):
        img_array = np.expand_dims(batch_images[i], axis=0)
        true_class = np.argmax(batch_labels[i])
        pred_class = np.argmax(model.predict(img_array, verbose=0), axis=1)[0]
        heatmap, superimposed_img = compute_gradcam(
            model, img_array, last_conv_layer_name, pred_class, config["model_type"]
        )
        original_img = batch_images[i]
        if "resnet50" in config["model_type"].lower():
            original_img = (original_img + 1) * 127.5
        elif "inceptionv3" in config["model_type"].lower():
            original_img = (original_img + 1) * 127.5
        elif "xception" in config["model_type"].lower():
            original_img = (original_img + 1) * 127.5
        elif "densenet121" in config["model_type"].lower():
            original_img = (original_img + 1) * 127.5
        else:
            original_img = original_img * 255.0
        original_img = np.uint8(original_img)
        original_path = os.path.join(gradcam_save_dir, f"sample_{i}_original.png")
        cv2.imwrite(original_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        print(f"Saved original image at {original_path}")
        heatmap_path = os.path.join(gradcam_save_dir, f"sample_{i}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        print(f"Saved heatmap at {heatmap_path}")
        superimposed_path = os.path.join(gradcam_save_dir, f"sample_{i}_gradcam_true_{true_class}_pred_{pred_class}.png")
        cv2.imwrite(superimposed_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        print(f"Saved Grad-CAM image at {superimposed_path}")

    # Evaluate on test set
    print(f"\n==> Evaluating model {model_name} on test set ...")
    steps_test = int(np.ceil(len(test_x) / batch_size))
    y_pred_test = model.predict(test_generator, steps=steps_test, verbose=1)
    y_true_test = np.argmax(test_y_multi, axis=1)
    y_pred_classes_test = np.argmax(y_pred_test, axis=1)
    qwk_test = cohen_kappa_score(y_true_test, y_pred_classes_test, labels=[0, 1, 2, 3, 4], weights='quadratic')
    print(f"QWK on test set: {qwk_test:.4f}")
    accuracy_test = accuracy_score(y_true_test, y_pred_classes_test)
    print(f"Accuracy on test set: {accuracy_test:.4f}")
    f1_test = f1_score(y_true_test, y_pred_classes_test, average=None, labels=[0, 1, 2, 3, 4])
    sensitivity_test = recall_score(y_true_test, y_pred_classes_test, average=None, labels=[0, 1, 2, 3, 4])
    print(f"F1-score per class on test set: {[f'{f1:.4f}' for f1 in f1_test]}")
    print(f"Sensitivity per class on test set: {[f'{sens:.4f}' for sens in sensitivity_test]}")
    specificity_test = []
    cm = confusion_matrix(y_true_test, y_pred_classes_test, labels=[0, 1, 2, 3, 4])
    for cls in range(NUM_CLASSES):
        tn = np.sum(cm) - np.sum(cm[cls, :]) - np.sum(cm[:, cls]) + cm[cls, cls]
        fp = np.sum(cm[:, cls]) - cm[cls, cls]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_test.append(specificity)
    print(f"Specificity per class on test set: {[f'{spec:.4f}' for spec in specificity_test]}")
    precision_test = precision_score(y_true_test, y_pred_classes_test, average=None, labels=[0, 1, 2, 3, 4])
    print(f"Precision per class on test set: {[f'{prec:.4f}' for prec in precision_test]}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[0, 1, 2, 3, 4],
                yticklabels=[0, 1, 2, 3, 4])
    plt.title(f'Test Confusion Matrix - QWK: {qwk_test:.4f} ({config["model_type"]})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_cm_test_path = f"/content/drive/MyDrive/working/test_confusion_matrix_{config['model_type']}.png"
    plt.savefig(save_cm_test_path)
    plt.close()
    print(f"Saved test confusion matrix to {save_cm_test_path}")

# Save meta-learning features
for key, features in meta_features.items():
    np.save(os.path.join(meta_save_dir, f"{key}.npy"), features)
    print(f"Saved features {key} at {os.path.join(meta_save_dir, f'{key}.npy')}")

# Process 4D features and combine for meta-learning
def reduce_4d_to_2d(features_4d):
    """Reduce 4D features to 2D using global average pooling."""
    return np.mean(features_4d, axis=(1, 2))  # Average over spatial dimensions

model_names = ["xception", "resnet50", "efficientnetb0", "inceptionv3", "densenet121"]
combined_train_features_2d = []
combined_valid_features_2d = []
combined_test_features_2d = []

for model_name in model_names:
    # Load 2D features
    train_2d_path = os.path.join(meta_save_dir, f"{model_name}_train_2d.npy")
    valid_2d_path = os.path.join(meta_save_dir, f"{model_name}_valid_2d.npy")
    test_2d_path = os.path.join(meta_save_dir, f"{model_name}_test_2d.npy")

    if os.path.exists(train_2d_path):
        train_features_2d = np.load(train_2d_path)
        combined_train_features_2d.append(train_features_2d)
        print(f"Loaded 2D train features for {model_name}: shape {train_features_2d.shape}")

    if os.path.exists(valid_2d_path):
        valid_features_2d = np.load(valid_2d_path)
        combined_valid_features_2d.append(valid_features_2d)
        print(f"Loaded 2D valid features for {model_name}: shape {valid_features_2d.shape}")

    if os.path.exists(test_2d_path):
        test_features_2d = np.load(test_2d_path)
        combined_test_features_2d.append(test_features_2d)
        print(f"Loaded 2D test features for {model_name}: shape {test_features_2d.shape}")

    # Load 4D features and reduce dimensionality
    train_npz_path = os.path.join(meta_save_dir, f"{model_name}_train_features.npz")
    valid_npz_path = os.path.join(meta_save_dir, f"{model_name}_valid_features.npz")
    test_npz_path = os.path.join(meta_save_dir, f"{model_name}_test_features.npz")

    if os.path.exists(train_npz_path):
        train_data = np.load(train_npz_path)
        train_features_4d = train_data['features_4d']
        train_features_4d_reduced = reduce_4d_to_2d(train_features_4d)
        combined_train_features_2d.append(train_features_4d_reduced)
        print(f"Loaded and reduced 4D train features for {model_name}: shape {train_features_4d_reduced.shape}")

    if os.path.exists(valid_npz_path):
        valid_data = np.load(valid_npz_path)
        valid_features_4d = valid_data['features_4d']
        valid_features_4d_reduced = reduce_4d_to_2d(valid_features_4d)
        combined_valid_features_2d.append(valid_features_4d_reduced)
        print(f"Loaded and reduced 4D valid features for {model_name}: shape {valid_features_4d_reduced.shape}")

    if os.path.exists(test_npz_path):
        test_data = np.load(test_npz_path)
        test_features_4d = test_data['features_4d']
        test_features_4d_reduced = reduce_4d_to_2d(test_features_4d)
        combined_test_features_2d.append(test_features_4d_reduced)
        print(f"Loaded and reduced 4D test features for {model_name}: shape {test_features_4d_reduced.shape}")

# Combine 2D and 4D (after reduction) features
combined_train_features = np.concatenate(combined_train_features_2d, axis=1)
combined_valid_features = np.concatenate(combined_valid_features_2d, axis=1)
combined_test_features = np.concatenate(combined_test_features_2d, axis=1)

print(f"Shape of combined train features: {combined_train_features.shape}")
print(f"Shape of combined valid features: {combined_valid_features.shape}")
print(f"Shape of combined test features: {combined_test_features.shape}")

# Save combined features
np.save(os.path.join(meta_save_dir, "combined_train_features.npy"), combined_train_features)
np.save(os.path.join(meta_save_dir, "combined_valid_features.npy"), combined_valid_features)
np.save(os.path.join(meta_save_dir, "combined_test_features.npy"), combined_test_features)
print(f"Saved combined features at {meta_save_dir}")
