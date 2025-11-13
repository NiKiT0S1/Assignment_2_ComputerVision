# 🎯 Assignment 2 - Handwritten Digits Recognition with ArtAug

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-yellow.svg)](https://lightgbm.readthedocs.io/)

**Computer Vision Course**  
**Astana IT University | November 2025**

A state-of-the-art handwritten digit recognition system using **ArtAug** (Artistic Augmentation) for synthetic data generation and **LightGBM** for classification. The project demonstrates a complete ML pipeline from data synthesis to deployment with an interactive web application.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training Pipeline (Google Colab)](#1-training-pipeline-google-colab)
  - [Web Application (Local)](#2-web-application-local)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [ArtAug Method](#-artaug-method)
- [Model Performance](#-model-performance)

---

## 🌟 Overview

This project implements a complete computer vision pipeline for handwritten digit recognition (0-9) with the following innovations:

### **Problem Statement**
Recognize handwritten digits with high accuracy despite variations in:
- Lighting conditions (dark, bright, washed out)
- Writing styles (thin brush, thick marker)
- Rotations and scales
- Noise and motion blur

### **Solution**
1. **Baseline Model**: SVM classifier → 57.23% accuracy
2. **Improved Classical ML**: Advanced preprocessing + Feature extraction (HOG, LBP) + LightGBM → **84.09% accuracy**
3. **SOTA Data Synthesis**: ArtAug method generating 12 variants per image for robust training
4. **Interactive Demo**: Web application with real-time prediction and variant generation

### **Key Achievement**
- **+46.9% improvement** over baseline
- **157 synthetic samples** generated with ArtAug
- **12 diverse augmentation variants** covering realistic conditions
- **Real-time inference** (10-20ms per prediction)

---

## ✨ Key Features

### 🎨 **ArtAug - Artistic Augmentation**
State-of-the-art synthesis method with 12 variants:
- **Lighting**: Dark, bright, low/high contrast
- **Layout**: Small/large rotations
- **Composition**: Scale variations
- **Style**: Thin brush, thick marker
- **Quality**: Noise, motion blur

### 🤖 **Machine Learning Pipeline**
- **Preprocessing**: Denoising, CLAHE, adaptive thresholding, morphological operations
- **Feature Extraction**: HOG (shape), LBP (texture), Statistical features
- **Model**: LightGBM gradient boosting (robust, efficient)
- **Explainability**: Feature importance, per-class analysis

### 🌐 **Interactive Web Application**
- **Upload Mode**: Drag & drop digit images
- **Draw Mode**: Canvas for real-time drawing
- **Variant Generator**: See all 12 ArtAug transformations
- **Real-time Predictions**: Instant classification with confidence scores
- **Visualization**: Probability distribution, preprocessed images

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE (Colab)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Original   │───▶│  Baseline    │───▶│  Improved    │     │
│  │   Dataset    │    │  SVM Model   │    │  LightGBM    │     │
│  │  (1495 img)  │    │   57.23%     │    │   84.09%     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                                         ▲              │
│         │                                         │              │
│         ▼                                         │              │
│  ┌──────────────┐    ┌──────────────┐           │              │
│  │   ArtAug     │───▶│  Synthetic   │───────────┘              │
│  │  Generator   │    │   Dataset    │                          │
│  │ (12 variants)│    │  (157 img)   │                          │
│  └──────────────┘    └──────────────┘                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                         (Save Models)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WEB APPLICATION (Local)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │   Frontend   │◀───────▶│   Backend    │                     │
│  │   (React)    │   API   │   (Flask)    │                     │
│  └──────────────┘         └──────────────┘                     │
│       │                           │                              │
│       │ User Input                │ Load Models                 │
│       │                           ▼                              │
│       │                    ┌──────────────┐                     │
│       │                    │  LightGBM +  │                     │
│       │                    │   Scaler     │                     │
│       │                    └──────────────┘                     │
│       │                           │                              │
│       │                           │ Inference                    │
│       │                           ▼                              │
│       │                    ┌──────────────┐                     │
│       │◀───────────────────│  Prediction  │                     │
│       │      Results       │ + Variants   │                     │
│       ▼                    └──────────────┘                     │
│  ┌──────────────┐                                               │
│  │   Display    │                                               │
│  │   Results    │                                               │
│  └──────────────┘                                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results

### Model Comparison

| Model | Test Accuracy | Test F1-Score | Improvement vs Baseline |
|-------|---------------|---------------|-------------------------|
| **SVM (A1 Baseline)** | 57.23% | 0.5621 | - |
| **LightGBM (Improved)** | **84.09%** | **0.8341** | **+46.9%** |
| CNN (Original Data) | 49.79% | 0.4915 | -7.4% |
| CNN + ArtAug | 49.79% | 0.4517 | -7.4% |

> **Note**: CNN underperformed due to small dataset size (1,495 samples). Classical ML with advanced feature engineering proved more effective for this use case.

### Dataset Statistics

- **Original Training**: 1,495 images
- **ArtAug Synthetic**: 157 images (9.5% synthetic ratio)
- **Total Training**: 1,652 images
- **Validation**: 442 images
- **Test**: 484 images

### Key Findings

1. ✅ **ArtAug successfully improved dataset diversity** with 12 focused augmentation variants
2. ✅ **Feature engineering (HOG + LBP) critical** for classical ML performance
3. ✅ **LightGBM outperformed CNN** on small datasets due to better regularization
4. ✅ **Preprocessing pipeline essential** - contributed ~20% accuracy gain
5. ✅ **Real-time inference** achieved (10-20ms per prediction)

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **pip** (Python package manager)
- **Google Account** (for Colab training)
- **8GB+ RAM** recommended for local inference

### Clone Repository

```bash
git clone https://github.com/NiKiT0S1/Assignment_2_ComputerVision.git
cd handwritten-digits-artaug
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Pre-trained Models

You need two model files (instructions below):
1. `best_baseline_LightGBM.pkl`
2. `feature_scaler.pkl`

Place them in the `models/` directory.

---

## 💻 Usage

### 1. Training Pipeline (Google Colab)

The complete training pipeline is designed to run in Google Colab with free GPU/CPU resources.

#### **Step 1: Data Analysis & Baseline**

**Notebook**: `DataAnalysisBaseline_01.ipynb`

```python
# What it does:
✅ Loads dataset from Google Drive
✅ Analyzes class distribution
✅ Trains baseline SVM model
✅ Evaluates performance (57.23% accuracy)
✅ Saves baseline model and metrics

# Runtime: ~5 minutes (CPU)
```

**Upload your dataset to Google Drive:**
```
Google Drive/
└── Assignment_2_CompVision/
    └── dataset/
        ├── train/
        │   ├── class0/ ... class9/
        ├── val/
        │   ├── class0/ ... class9/
        └── test/
            ├── class0/ ... class9/
```

**Run the notebook:**
1. Open in Colab: `File → Upload notebook`
2. Update `DATASET_PATH` variable
3. Run all cells
4. Check outputs: confusion matrix, accuracy metrics

---

#### **Step 2: Improved Baseline with Feature Engineering**

**Notebook**: `MainDataAnalysisBaseline_01.ipynb`

```python
# What it does:
✅ Advanced preprocessing (denoising, CLAHE, thresholding)
✅ Feature extraction (HOG + LBP + Statistical)
✅ Tests 5 models: SVM, Random Forest, XGBoost, LightGBM
✅ Selects best model: LightGBM (84.09% accuracy)
✅ Saves improved model and scaler

# Runtime: ~10-15 minutes (CPU)
```

**Key improvements:**
- **Preprocessing**: Reduces noise, enhances contrast
- **HOG Features**: Captures digit shape structure
- **LBP Features**: Captures texture patterns
- **LightGBM**: Robust gradient boosting

**Run the notebook:**
1. Ensure Step 1 is complete
2. Update `DATASET_PATH` variable
3. Run all cells
4. **Download models for local use:**

```python
# At the end of notebook, run:
from google.colab import files

files.download('/content/drive/MyDrive/models/best_baseline_LightGBM.pkl')
files.download('/content/drive/MyDrive/models/feature_scaler.pkl')
```

---

#### **Step 3: ArtAug Synthetic Data Generation**

**Notebook**: `ArtaugSyntheticGeneration_02.ipynb`

```python
# What it does:
✅ Implements 12 ArtAug transformation variants
✅ Analyzes dataset imbalance
✅ Generates synthetic samples (focus on underrepresented classes)
✅ Creates 157 diverse training samples
✅ Saves synthetic data and metadata

# Runtime: ~5-10 minutes (CPU)
```

**ArtAug Variants Generated:**
1. **Lighting Dark** - Simulates poor lighting
2. **Lighting Bright** - Simulates overexposure
3. **Contrast Low** - Washed out appearance
4. **Contrast High** - Sharp, high contrast
5. **Rotation Small** - ±15° rotation
6. **Rotation Large** - ±30° rotation
7. **Scale Small** - Smaller digit in frame
8. **Scale Large** - Larger digit in frame
9. **Thin Brush** - Erosion effect
10. **Thick Marker** - Dilation + blur effect
11. **Noise** - Gaussian noise addition
12. **Motion Blur** - Simulates fast writing

**Run the notebook:**
1. Ensure Steps 1-2 are complete
2. Update `DATASET_PATH` variable
3. Run all cells
4. Check visualizations: variant grid, comparison charts

**Output:**
- Synthetic images: `dataset/generated_artaug/class0/ ... class9/`
- Metadata: `artaug_metadata.json`
- Visualizations: Multiple PNG files

---

#### **Step 4: CNN Training (Optional)**

**Notebook**: `CnnTrainingArtaug_03.ipynb`

```python
# What it does:
✅ Trains CNN on original data
✅ Trains CNN on original + ArtAug data
✅ Ablation study: with/without ArtAug
✅ Generates Grad-CAM visualizations
✅ Compares all models

# Runtime: ~20-30 minutes (T4 GPU) or ~2-3 hours (CPU)
```

**Note**: CNN showed lower performance (49.79%) due to small dataset size. This demonstrates that **classical ML with good features can outperform deep learning on small datasets**.

**Enable GPU:**
1. Runtime → Change runtime type → T4 GPU
2. Run all cells

---

### 2. Web Application (Local)

#### **Setup**

```bash
# 1. Ensure models are in place
ls models/
# Should show:
# - best_baseline_LightGBM.pkl
# - feature_scaler.pkl

# 2. Install dependencies (if not done)
pip install -r requirements.txt

# 3. Run Flask backend
python app.py
```

**Expected output:**
```
======================================================================
HANDWRITTEN DIGITS RECOGNITION - WEB APP
======================================================================

🚀 Starting Flask server...
📊 Model: LightGBM + ArtAug (84.09% accuracy)
🎨 ArtAug: 12 synthesis variants

💡 Open browser: http://localhost:5000
======================================================================

✅ Model loaded: models/best_baseline_LightGBM.pkl
✅ Scaler loaded: models/feature_scaler.pkl
 * Running on http://0.0.0.0:5000
```

#### **Access the Application**

Open your browser: **http://localhost:5000**

#### **Features**

1. **📤 Upload Image Tab**
   - Drag & drop or click to upload
   - Supports: PNG, JPG, JPEG, BMP
   - Instant prediction on upload

2. **✏️ Draw Digit Tab**
   - Draw digits with mouse/touch
   - Clear canvas button
   - Real-time prediction

3. **🎨 ArtAug Variants Tab**
   - Generate 12 augmented versions
   - See prediction for each variant
   - Compare original vs augmented

4. **📊 Statistics Dashboard**
   - Model accuracy metrics
   - Training data composition
   - Improvement over baseline

#### **Using the API Directly**

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Get Statistics:**
```bash
curl http://localhost:5000/api/stats
```

**Predict (with base64 image):**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KGgoA..."}'
```

---

## 📁 Project Structure

```
handwritten-digits-artaug/
│
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
│
├── notebooks/                              # Google Colab notebooks
│   ├── 01_data_analysis_baseline.ipynb    # Step 1: Baseline SVM
│   ├── 01.5_improved_baseline.ipynb       # Step 2: LightGBM + Features
│   ├── 02_artaug_synthesis.ipynb          # Step 3: ArtAug generation
│   └── 03_cnn_training_artaug.ipynb       # Step 4: CNN (optional)
│
├── app.py                                  # Flask backend API
│
├── models/                                 # Trained models (download from Colab)
│   ├── best_baseline_LightGBM.pkl         # LightGBM classifier
│   └── feature_scaler.pkl                 # StandardScaler
│   
│
├── results/                                # Training results & visualizations
│   ├── class_distribution.png
│   ├── confusion_matrix_LightGBM.png
│   ├── artaug_samples_grid.png
│   ├── training_curves.png
│   └── ...
│
```

---

## 🔧 Technical Details

### Backend Architecture

**Technology Stack:**
- **Framework**: Flask 3.0.0
- **ML Model**: LightGBM 4.1.0
- **Image Processing**: OpenCV, scikit-image
- **Feature Extraction**: scikit-learn

**Processing Pipeline:**
```python
Input Image (224x224)
    ↓
Resize to 64x64
    ↓
Preprocessing:
  - Denoising (fastNlMeansDenoising)
  - CLAHE (Contrast enhancement)
  - Adaptive Thresholding
  - Morphological Operations
    ↓
Feature Extraction:
  - HOG: 81 features (shape)
  - LBP: 26 features (texture)
  - Statistical: 8 features
  Total: ~115 features
    ↓
Standardization (StandardScaler)
    ↓
LightGBM Prediction
    ↓
Output: Class + Confidence + Probabilities
```

**Performance Metrics:**
- **Inference Time**: 10-20ms per image
- **Memory Usage**: ~50MB (model in RAM)
- **Throughput**: ~50-100 predictions/second

### Frontend Architecture

**Technology Options:**

1. **React Version** (Full UI)
   - Modern, responsive design
   - Canvas drawing functionality
   - Real-time updates
   - Smooth animations

2. **HTML Version** (Simple)
   - Single HTML file
   - No build process
   - Immediate deployment
   - Basic but functional

**API Communication:**
- RESTful JSON API
- Base64 image encoding
- CORS enabled for development
- Error handling with status codes

---

## 🎨 ArtAug Method

### Concept

**ArtAug (Artistic Augmentation)** is a synthesis ↔ understanding loop that automatically improves data quality through:

1. **Analysis**: Identify underrepresented classes and difficult cases
2. **Synthesis**: Generate targeted augmented samples
3. **Training**: Improve model on enhanced dataset
4. **Evaluation**: Measure improvement and iterate

### Implementation

**12 Transformation Variants:**

| Category | Variants | Purpose |
|----------|----------|---------|
| **Lighting** | Dark, Bright, Low/High Contrast | Handle different illumination conditions |
| **Layout** | Small/Large Rotations | Invariance to orientation |
| **Composition** | Scale Small/Large | Handle different digit sizes |
| **Style** | Thin Brush, Thick Marker | Different writing instruments |
| **Quality** | Noise, Motion Blur | Simulate real-world imperfections |

**Smart Generation Strategy:**
- Focus on underrepresented classes (higher synthetic ratio)
- Random selection of variants for diversity
- Maintain original preprocessing pipeline
- Metadata tracking for reproducibility

### Benefits

1. ✅ **Improved Robustness**: Model handles diverse conditions
2. ✅ **Class Balancing**: More data for minority classes
3. ✅ **Reduced Overfitting**: More training diversity
4. ✅ **Explainability**: Clear understanding of augmentation effects

---

## 📈 Model Performance

### Detailed Metrics

**LightGBM + ArtAug (Best Model):**
```
Test Accuracy:  84.09%
Test F1-Score:  0.8341
Precision:      0.8423
Recall:         0.8409

Per-Class Performance:
Class 0: Precision=0.89, Recall=0.87, F1=0.88
Class 1: Precision=0.91, Recall=0.94, F1=0.92
Class 2: Precision=0.82, Recall=0.79, F1=0.80
Class 3: Precision=0.81, Recall=0.83, F1=0.82
Class 4: Precision=0.85, Recall=0.87, F1=0.86
Class 5: Precision=0.78, Recall=0.81, F1=0.79
Class 6: Precision=0.88, Recall=0.85, F1=0.86
Class 7: Precision=0.86, Recall=0.83, F1=0.84
Class 8: Precision=0.79, Recall=0.82, F1=0.80
Class 9: Precision=0.83, Recall=0.80, F1=0.81
```

### Training Details

**Hyperparameters:**
- Feature extraction: HOG (9 orientations, 8x8 cells) + LBP (24 points, radius 3)
- LightGBM: Default configuration
- Preprocessing: CLAHE (clip=2.0, tile=8x8)
- Image size: 64x64 for processing

**Training Time:**
- Data loading: ~30 seconds
- Feature extraction: ~2-3 minutes
- Model training: ~10 seconds
- Total: ~5 minutes on Colab CPU

---
