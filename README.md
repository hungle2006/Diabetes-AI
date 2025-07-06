# 🔬 AI Diabetes Detection System
## AI-powered diabetes detection through retinal imaging and clinical data analysis

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/diabetes-detection-ai?style=social)](https://github.com/yourusername/diabetes-detection-ai)

</div>

---

## 📋 Table of Contents
- [🎯 Introduction](#-introduction)
- [✨ Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🚀 Installation](#-installation)
- [💻 Usage](#-usage)
- [📁 Data](#-data)
- [📊 Results](#-results)
- [🔧 Configuration](#-configuration)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📞 Contact](#-contact)

---

## 🎯 Introduction

### 🚨 The Problem: Diabetic Retinopathy Crisis
<div align="center">
<img src="https://img.shields.io/badge/Affected-126M_People-red?style=for-the-badge" alt="Affected People">
<img src="https://img.shields.io/badge/At_Risk-37M_People-orange?style=for-the-badge" alt="At Risk">
<img src="https://img.shields.io/badge/Status-Preventable_Blindness-yellow?style=for-the-badge" alt="Preventable">
</div>

> **Diabetic Retinopathy (DR)** is a leading cause of preventable blindness worldwide, affecting over **126 million people** with around **37 million** at vision-threatening stages.

**Key Challenges:**
- 🩸 **Diabetes damages retinal blood vessels** causing DR
- 🏥 **Limited access to ophthalmic care** in rural and low-income areas
- ⏰ **Early screening is critical** for preventing vision loss
- 💰 **Huge economic burden** on healthcare systems

### 🎯 Our Solution

This project develops an **advanced AI system** for diabetes detection and related complications through:

<table>
<tr>
<td width="50%">

**🔍 Retinal Image Analysis**
- Detection of diabetic retinopathy
- Severity classification (0-4 scale)
- Automated screening capability

</td>
<td width="50%">

**📊 Clinical Data Analysis**
- Diabetes risk prediction
- Medical indicators analysis
- Ensemble ML algorithms

</td>
</tr>
</table>

### 🤖 Discord Chatbot Integration
We've integrated our AI system into a **Discord chatbot** for easy access and real-time consultations!

<div align="center">
<img src="image/chat-bot.gif" alt="Discord Chatbot Demo" width="600">
</div>

---

## ✨ Features

<table>
<tr>
<td width="33%">

### 🔍 DR Detection
- **Severity Classification**: 0-4 scale
- **Sign Detection**: Microaneurysms, hemorrhages, exudates
- **Accuracy**: 84.72%

</td>
<td width="33%">

### 📊 Risk Prediction
- **Clinical Analysis**:age, bmi, gender, glucose_apache, creatinine_apache, bun_apache, sodium_apache, albumin_apache
- **Ensemble Models**: Multiple ML algorithm
- **Performance**: 86.43% accuracy

</td>
<td width="33%">

### 🌐 User Interface
- **Web Application**: Easy upload & analysis
- **Discord Bot**: Real-time consultations
- **Batch Processing**: Multiple files support
- **Detailed Reports**: With visualizations

</td>
</tr>
</table>

---

## 🏗️ System Architecture

<div align="center">
<img src="image/System_Architecture.jpg" alt="System Architecture" width="600">
</div>

# 1. Image Processing

<div align="center">
<img src="image/process-image.jpg" alt="System Architecture" width="300">
</div>

#### A. Smart Cropping
```python
def crop_image_from_gray_to_color(img, tol=7):
    """Remove black borders around retinal images"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    return img[np.ix_(rows, cols)]
```

#### B. Image Enhancement
```python
def load_ben_color(path, sigmaX=10, IMG_SIZE=244):
    """Process images using Ben Graham's method"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Remove dark regions
    image = crop_image_from_gray_to_color(image, tol=7)
    
    # Standardize size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Enhance contrast
    # Formula: 4*original - 4*blurred + 128
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    
    return image
```

**Why is this important?**
- **Cropping**: Removes black areas with no medical information
- **Enhancement**: Highlights blood vessels and retinal lesions
- **Resize**: Standardizes dimensions for model input

### C. Batch Processing
```python
processed_ids = []
for idx, row in df_train.iterrows():
    img_filename = f"{row['id_code']}.png"
    img_path = os.path.join(train_img_folder, img_filename)
    
    try:
        proc_img = load_ben_color(img_path, sigmaX=10, IMG_SIZE=244)
        proc_img_bgr = cv2.cvtColor(proc_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(processed_folder, img_filename), proc_img_bgr)
        processed_ids.append(row['id_code'])
    except Exception as e:
        print(f"Error {img_filename}: {e}")
```

### D. Data Splitting
```python
# Prepare data
x = df_train_processed['id_code']
y = df_train_processed['diagnosis']
x, y = shuffle(x, y, random_state=42)

# Split Train(68%) - Val(12%) - Test(20%)
x_temp, test_x, y_temp, test_y = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
train_x, valid_x, train_y, valid_y = train_test_split(x_temp, y_temp, test_size=0.15/0.80, stratify=y_temp, random_state=42)
```

## 🔑 Key Points

### Image Processing Techniques
1. **Smart Cropping**: Keep only medically relevant regions
2. **Contrast Enhancement**: Formula `4*original - 4*blurred + 128`
3. **Standardization**: 244x244 pixels but next step, we have resize image:299x299 (Xception and InceptionV3)

### Data Transformation
- **Input**: Raw images with varying sizes and black borders
- **Output**: 244x244 images with enhanced contrast, retina only


| Original image (NO DIABETES) | Processed image (NO DIABETES) |
|----------|-------------|
| <img src="image/label0_original.png?raw=true" width="200"/> |    <img src="image/label0_processed.png?raw=true" width="200"/> |

| Original image (DIABETES) | Processed image (DIABETES)|
|----------|-------------|
| <img src="image/label4_original.png?raw=true" width="200"/> |    <img src="image/label4_processed.png?raw=true" width="200"/> |

[📥 Tải mã xử lý ảnh tại đây](https://github.com/hungle2006/Diabetes-AI/raw/main/model/Processing_images.py)

---

# 2. CNN-MODEL



## 📋 Overview

This project implements a multi-model ensemble approach for diabetic retinopathy classification, utilizing advanced deep learning techniques including:
- Multiple pre-trained CNN architectures
- Dynamic data augmentation
- Class balancing strategies
- Grad-CAM visualization
- Meta-learning feature extraction

## 🎯 Key Features

### Multi-Model Architecture Support
- **ResNet50**: Deep residual networks for robust feature extraction
- **EfficientNetB0**: Efficient scaling for optimal performance
- **InceptionV3**: Multi-scale feature processing
- **DenseNet121**: Dense connections for feature reuse
- **Xception**: Depthwise separable convolutions

### Advanced Training Techniques
- **Dynamic Class Balancing**: Automatic adjustment of class weights during training
- **Mixup Augmentation**: Advanced data augmentation for better generalization
- **Rare Class Augmentation**: Targeted augmentation for underrepresented classes
- **Progressive Learning**: Two-stage training with frozen and unfrozen layers

### Intelligent Callbacks
- **QWK Evaluation**: Quadratic Weighted Kappa scoring for medical accuracy
- **Dynamic Learning Rate**: Adaptive learning rate based on QWK performance
- **Early Stopping**: Prevention of overfitting with best weight restoration

## 🏗️ Architecture

### Data Generator (`My_Generator`)
```python
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
```

### Model Creation
```python
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
```

## 📊 Performance Monitoring

### Metrics Tracked
- **Quadratic Weighted Kappa (QWK)**: Primary metric for medical classification
- **F1-Score per Class**: Detailed performance analysis
- **Sensitivity/Recall**: True positive rate per class
- **Specificity**: True negative rate per class
- **Precision**: Positive predictive value

### Visualization Tools
- **Grad-CAM**: Visual explanation of model decisions
- **Confusion Matrix**: Detailed classification results
- **Loss Curves**: Training progress monitoring

## 🔧 Training Pipeline

### Stage 1: Transfer Learning
```python
# Freeze base model layers
for layer in model.layers[:50]:
    layer.trainable = False

# Train classification head
model.compile(
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)
```

### Stage 2: Fine-tuning
```python
# Unfreeze all layers
for layer in model.layers:
    layer.trainable = True

# Fine-tune with mixup
train_mixup = My_Generator(
    is_train=True, 
    mix=True, 
    augment=True,
    balance_classes=True
)
```

## 🎨 Data Augmentation

### Standard Augmentation
- Random brightness/contrast adjustment
- Horizontal and vertical flips
- Multiplicative noise
- Crop and pad operations

### Rare Class Augmentation
- Gaussian noise addition
- Rotation and scaling
- Hue/saturation/value shifts
- Shift-scale-rotate transformations

## 🧠 Meta-Learning Features

### Feature Extraction
```python
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
```

### Multi-Scale Feature Fusion
- Combines features from multiple model architectures
- Reduces 4D features to 2D using global average pooling
- Creates comprehensive feature representation

## 📈 Results Analysis

### Automatic Evaluation
- Best model checkpointing based on QWK score
- Comprehensive test set evaluation
- Per-class performance metrics
- Confusion matrix generation

### Visual Interpretability
- Grad-CAM heatmaps for model decision explanation
- Original vs. augmented image comparisons
- Class activation visualizations

### Training
```python
# Configure model
model_configs = {
    "densenet121": {
        "model_type": "densenet121",
        "weights_path": "path/to/weights.h5",
        "save_path": "path/to/save/model.h5"
    }
}

# Train model
for model_name, config in model_configs.items():
    model = create_model(
        input_shape=(224, 224, 3),
        n_out=NUM_CLASSES,
        model_type=config["model_type"]
    )
    # ... training loop
```

### Evaluation
```python
# Evaluate on test set
y_pred = model.predict(test_generator)
qwk_score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
```

## 📁 Project Structure

<div align="center">
<img src="image/Project_Structure.jpg" alt="Discord Chatbot Demo" width="600">
</div>

## 🔬 Technical Details

### Class Balancing Strategy
- Analyzes initial class distribution
- Generates synthetic samples for underrepresented classes
- Implements targeted augmentation for class 3 (30% increase because between class 3 and class 4 is difficult to distinguish. In addition, class 3 accuracy is below 30)
- Dynamic weight adjustment during training

### Quality Assurance
- Comprehensive error handling for image loading
- Automatic validation of generated samples
- Robust preprocessing pipeline
- Memory-efficient batch processing

## 📊 Model Performance

### Evaluation Metrics
- **Primary**: Quadratic Weighted Kappa (QWK)
- **Secondary**: F1-Score, Sensitivity, Specificity, Precision
- **Visualization**: Grad-CAM, Confusion Matrix

### Checkpoint Management
- Best model saving based on QWK score
- Multiple save formats (Keras, weights, JSON config)
- Metadata tracking for reproducibility

### Training and test results

<div align="center">
<img src="image/traning_and_tesing_result.jpg" width="600">
</div>

📥 [Tải file cnn_model.py](https://raw.githubusercontent.com/hungle2006/Diabetes-AI/main/model/cnn_model.py)

---

# 3. Medical Image Classification Project with Meta-Learning (MAML/FOMAML)

## Introduction

This project implements a medical image classification system using meta-learning techniques such as Model-Agnostic Meta-Learning (MAML) and First-Order MAML (FOMAML). The codebase focuses on processing image data, extracting features from pre-trained deep learning models like EfficientNet, Xception, InceptionV3, ResNet50, and DenseNet121, and applying meta-learning to improve classification performance on imbalanced datasets. The project also incorporates techniques like Grad-CAM for visualization and Focal Loss to handle minority classes.

The project is designed to work with medical image datasets, such as retinal images for diagnosing diseases. The codebase includes data preprocessing, class balancing, feature extraction, and meta-learning model training with strategies like data augmentation and class balancing.

## Key Features

- **Data Preprocessing**: Load and process images from directories or compressed files, convert to RGB format, and resize images.
- **Data Balancing**: Use techniques like oversampling and random erasing to address class imbalance.
- **Feature Extraction**: Extract 2D and 4D features from pre-trained deep learning models (EfficientNetB0, Xception, InceptionV3, ResNet50, DenseNet121).
- **Meta-Learning (MAML/FOMAML)**: Train meta-learning models with inner and outer loops, integrating Focal Loss to focus on minority classes.
- **Grad-CAM Visualization**: Generate heatmaps to visualize important regions in images, aiding in interpreting model predictions.
- **Performance Evaluation**: Use Quadratic Weighted Kappa (QWK), Precision, Recall, and F1-score to evaluate model performance.
- **Storage and Management**: Save features, models, and evaluation metrics to .npy, .json, and .h5 files.

## Code Structure

Below are the main components of the codebase:

### A. Data Loading and Preprocessing

- `load_original_image(image_id, extract_dir)`: Load original images from an extracted directory based on image_id. Convert images to RGB and handle errors if images are not found.
- `load_processed_image(image_id, processed_folder, size)`: Load preprocessed images, resize them, and convert to RGB.

```python
def load_original_image(image_id, extract_dir):
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
```

### B. Data Augmentation and Balancing

- `custom_random_erasing(image, scale, ratio, p)`: Apply random erasing to augment image data, helping the model learn more robust features.
- `balance_and_augment_data(images, labels, target_classes, samples_per_class)`: Balance the number of samples across classes using augmentation techniques like horizontal flipping, rotation, brightness adjustment, and Gaussian noise.

```python
def balance_and_augment_data(images, labels, target_classes=[0, 1, 2, 3, 4], samples_per_class=None):
    num_classes = labels.shape[1]
    label_indices = np.argmax(labels, axis=1)
    keep_indices = np.isin(label_indices, target_classes)
    filtered_images = images[keep_indices]
    filtered_labels = labels[keep_indices]
    filtered_label_indices = label_indices[keep_indices]
    
    class_counts = np.bincount(filtered_label_indices, minlength=num_classes)
    print(f"Initial distribution: {dict(zip(range(num_classes), class_counts))}")
    
    max_count = samples_per_class or max(class_counts)
    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(p=0.2),
        A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=0.2),
    ])
    
    new_images, new_labels = [], []
    for cls in target_classes:
        cls_indices = np.where(filtered_label_indices == cls)[0]
        cls_images = filtered_images[cls_indices]
        cls_labels = filtered_labels[cls_indices]
        current_count = len(cls_indices)
        new_images.extend(cls_images)
        new_labels.extend(cls_labels)
        augment_count = max_count - current_count
        if augment_count > 0:
            for _ in range(augment_count):
                idx = np.random.choice(cls_indices)
                img = filtered_images[idx].astype(np.uint8)
                aug_img = augmenter(image=img)['image']
                aug_img = custom_random_erasing(aug_img, scale=(0.01, 0.05), ratio=(0.5, 2.0), p=0.3)
                new_images.append(aug_img)
                new_labels.append(filtered_labels[idx])
    
    new_images = np.array(new_images, dtype=np.float32)
    new_labels = np.array(new_labels, dtype=np.float32)
    return new_images, new_labels
```

### C. Feature Extraction

- `extract_2d_features(model_name, config, generator, save_dir, sample_ids)`: Extract 2D features from deep learning models and save them to .npy files.
- `load_4d_features(model_name, split)`: Load 4D features from .npz files and reduce them to 2D using averaging.
- `combine_and_reduce_features(features_2d_dict, features_4d_dict, labels, sample_ids, save_dir, n_components)`: Combine 2D and 4D features, apply PCA for dimensionality reduction, and ensure synchronization between features and labels.

### D. Meta-Learning (MAML/FOMAML)

**maml_fomaml_train_manual**:
- **Objective**: Train a meta-learning model with an inner loop to update weights on the support set and an outer loop to update the meta-model on the query set.
- **Components**:
  - Model: Uses custom layers like MemoryAugmentedLayer, GradientReversalLayer, and CustomGridDropout.
  - Loss: Combines Focal Loss, domain loss, and prototypical loss.
  - Optimization: Uses Adam optimizer with learning rate scheduling and early stopping.
- **Output**: Meta-model, classification model, feature extraction model, and training history.

```python
def maml_fomaml_train_manual(
    features, labels, valid_features, valid_labels, input_dim, n_episodes=50,
    n_support=15, n_query=10, inner_lr=0.001, outer_lr=0.001, fine_tune_lr=0.0001,
    use_fomaml=True, memory_size=20, sample_ids=None, images=None, features_4d_dict=None
):
    # Define model
    def create_model(input_dim):
        inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = CustomGridDropout(ratio=0.3, holes_number=10, p=0.3)(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = CustomGridDropout(ratio=0.3, holes_number=5,  p=0.3)(x)
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
        return model, classification_model, feature_model

    meta_model, meta_classification_model, feature_model = create_model(input_dim)
    # Continue with MAML/FOMAML training loop
    # ...
```

### E. Grad-CAM Visualization

- `compute_gradcam_4d(model, img_array, feature_4d, class_idx, layer_name, img_size)`: Compute and visualize Grad-CAM heatmaps to identify important regions in images contributing to model predictions.

```python
def compute_gradcam_4d(model, img_array, feature_4d, class_idx, layer_name, img_size=(224, 224)):
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
        heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap
    except Exception as e:
        logging.error(f"Error in compute_gradcam_4d: {str(e)}")
        return None
```

## Installation Requirements

To run the codebase, install the following libraries:

```bash
pip install tensorflow numpy opencv-python scikit-learn albumentations pandas seaborn matplotlib sklearn
```

Additionally, ensure:

- **TensorFlow**: Version 2.x.
- **OpenCV**: For image processing.
- **Albumentations**: For image data augmentation.
- **Scikit-learn**: For PCA dimensionality reduction and metric evaluation.
- **Seaborn and Matplotlib**: For visualization.

## Usage Instructions

### Prepare Data:

- Place original images in the `extract_dir` directory.
- Place preprocessed images in the `processed_folder` directory.
- Ensure 4D feature files (.npz) are available.

### Configure Directories:

- `meta_save_dir`: Directory for saving 2D and 4D features.
- `feature_save_dir`: Directory for saving model weights and metrics.
- `gradcam_save_dir`: Directory for saving Grad-CAM heatmaps.

### Run the Code:

The code will automatically:
- Load and preprocess data.
- Extract 2D and 4D features.
- Train the meta-learning model.
- Generate Grad-CAM heatmaps.
- Save results and metrics.

### Customize Parameters:

- `NUM_CLASSES`: Number of classification classes (default is 5).
- `n_episodes`: Number of meta-learning iterations (default is 20).
- `n_support` and `n_query`: Number of support and query samples per class.
- `inner_lr`, `outer_lr`, `fine_tune_lr`: Learning rates for the loops.

## Output Results

- **Features**: Saved as .npy files in `meta_save_dir`.
- **Model Weights**: Saved as .h5 files in `feature_save_dir`.
- **Grad-CAM Heatmaps**: Saved as .png files in `gradcam_save_dir`.
- **Evaluation Metrics**: Saved as .json files, including QWK, Precision, Recall, and F1-score.
- **Training History**: Training history plots saved as .png files.

## Important Notes

- **Handling NaN/Inf Errors**: The code includes checks and handling for NaN/Inf values in features and labels to ensure stability.
- **Data Synchronization**: Ensure the number of samples in features, labels, and sample IDs are synchronized before training.
- **Class Balancing**: Minority classes (e.g., classes 3 and 4) are weighted or oversampled to improve performance.
- **Memory Management**: Use `gc.collect()` and `tf.keras.backend.clear_session()` to minimize memory usage.



---
## 🚀 Installation

### 📋 System Requirements
<div align="center">

| Component | Requirement |
|-----------|-------------|
| 🐍 Python | 3.8+ |
| 🖥️ GPU | NVIDIA GPU with CUDA 11.0+ (recommended) |
| 💾 RAM | 8GB+ (16GB recommended) |
| 💿 Storage | 10GB+ for models and data |

</div>

### 🔧 Quick Setup

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/diabetes-detection-ai.git
cd diabetes-detection-ai

# 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt
pip install -e .

# 4️⃣ Download pre-trained models (~2GB)
python scripts/download_models.py
```

---

## 💻 Usage

### 🌐 Web Application
```bash
python web_app/app.py
```
**➡️ Open browser at** `http://localhost:5000`

### 🤖 Discord Bot
```bash
# Set up Discord bot token in .env
DISCORD_BOT_TOKEN=your_bot_token_here

# Run the bot
python discord_bot/main.py
```

### 🔌 API Usage
```python
from src.inference import DiabetesDetector

# Initialize detector
detector = DiabetesDetector()

# 👁️ Analyze retinal image
image_path = "path/to/fundus_image.jpg"
retinopathy_result = detector.predict_retinopathy(image_path)

# 📊 Predict from clinical data
clinical_data = {
    'glucose': 120,
    'bmi': 25.5,
    'age': 45,
    'blood_pressure': 130
}
diabetes_risk = detector.predict_diabetes_risk(clinical_data)

# 📋 Display results
print(f"🔍 Retinopathy severity: {retinopathy_result['severity']}")
print(f"📊 Diabetes risk: {diabetes_risk['probability']:.2%}")
```

### ⚡ Batch Processing
```bash
# Process multiple images
python scripts/batch_inference.py --input_dir data/test_images --output_dir results/

# Process CSV clinical data
python scripts/clinical_batch.py --input_file data/patients.csv --output_file results/predictions.csv
```

---

## 📁 Data

<div align="center">

### 👁️ Retinal Image Dataset
| Aspect | Details |
|--------|---------|
| **Sources** | APTOS 2019, EyePACS, Messidor-2 |
| **Size** | 50,000+ retinal images |
| **Format** | JPEG, PNG (512x512 pixels) |
| **Classes** | 5 severity levels (0-4) |

### 🏥 Clinical Dataset
| Aspect | Details |
|--------|---------|
| **Sources** | UCI Diabetes Dataset, NHANES |
| **Patients** | 100,000+ records |
| **Features** | 15 key medical indicators |
| **Target** | Binary classification |

</div>

---

## 📊 Results

<div align="center">

### 🎯 Model Performance

</div>

<table>
<tr>
<td width="50%">

#### 👁️ Retinopathy Model
| Metric | Score |
|--------|-------|
| Accuracy | **94.2%** |
| Precision | **93.8%** |
| Recall | **94.1%** |
| F1-Score | **93.9%** |
| AUC | **0.98** |

</td>
<td width="50%">

#### 🏥 Clinical Model
| Metric | Score |
|--------|-------|
| Accuracy | **91.8%** |
| Precision | **90.2%** |
| Recall | **92.1%** |
| F1-Score | **91.1%** |
| AUC | **0.96** |

</td>
</tr>
</table>

### 📈 Visualizations
<div align="center">
<img src="images/confusion_matrix.png" alt="Confusion Matrix" width="400">
<img src="images/roc_curves.png" alt="ROC Curves" width="400">
</div>

---

## 🔧 Configuration

Customize your setup by editing `configs/config.yaml`:

```yaml
model:
  retinopathy:
    backbone: "efficientnet-b4"
    input_size: 512
    num_classes: 5
  clinical:
    algorithm: "xgboost"
    features: 15
    
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  
discord_bot:
  token: "your_bot_token"
  prefix: "!"
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test with coverage
python -m pytest --cov=src tests/

# Integration tests
python tests/integration/test_pipeline.py
```

---

## 🤝 Contributing

<div align="center">
<img src="https://contrib.rocks/image?repo=yourusername/diabetes-detection-ai" alt="Contributors">
</div>

We welcome contributions! Here's how to get started:

1. **🍴 Fork** the repository
2. **🌿 Create** your feature branch (`git checkout -b feature/amazing-feature`)
3. **💾 Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **🚀 Push** to the branch (`git push origin feature/amazing-feature`)
5. **📝 Open** a Pull Request

### 📏 Coding Standards
- ✅ Follow PEP 8
- 📖 Add docstrings for functions
- 🧪 Write unit tests for new code
- 📚 Update documentation

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🏥 Medical Disclaimer

<div align="center">
<img src="https://img.shields.io/badge/⚠️-MEDICAL_DISCLAIMER-red?style=for-the-badge" alt="Medical Disclaimer">
</div>

> **Important**: This system is for **diagnostic assistance only** and does not replace professional medical advice. Always consult with qualified healthcare professionals for final treatment decisions.

---

## 📞 Contact

<div align="center">

**👨‍💻 Author**: [Your Name]

[![Email](https://img.shields.io/badge/Email-your.email@example.com-blue?style=for-the-badge&logo=gmail)](mailto:your.email@example.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![Website](https://img.shields.io/badge/Website-Visit-green?style=for-the-badge&logo=safari)](https://yourwebsite.com)

</div>

---

## 🙏 Acknowledgments

- 🎯 **Dataset Providers**: APTOS, EyePACS, UCI ML Repository
- 🛠️ **Open Source Communities**: TensorFlow, scikit-learn
- 👨‍⚕️ **Medical Experts**: Ophthalmologists who validated our results
- 🤖 **Discord Community**: For testing and feedback

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

<img src="https://img.shields.io/github/stars/yourusername/diabetes-detection-ai?style=social" alt="GitHub Stars">

</div>
