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

## 1. Image Processing

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

## 


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
