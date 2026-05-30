# 🌾 Crop Recommendation System
### Machine Learning for Agricultural Decision Support

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)


A machine learning project that recommends the most suitable crop to cultivate based on soil properties (pH, nutrients, minerals) and seasonal weather conditions (humidity, temperature, precipitation). Built with scikit-learn using best practices including Pipeline-based preprocessing, stratified cross-validation, and class-balanced training.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Fixes & Improvements](#key-fixes--improvements)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

Farmers in Ethiopia and similar agricultural regions face the challenge of selecting the right crop for their land. Wrong crop selection leads to poor yields, economic loss, and soil degradation. This project builds a multi-class classification model that recommends one of **12 crops** based on measurable soil and weather attributes.

**Target crops:** Teff, Maize, Wheat, Barley, Bean, Pea, Sorghum, Dagussa, Niger seed, Potato, Red Pepper, Fallow

---

## Dataset

**File:** `Crop Recommendation using Soil Properties and Weather Prediction.csv`

| Property | Value |
|----------|-------|
| Total samples | ~3,916 |
| Features | 28 (soil + weather) |
| Target classes | 12 crops |
| Class balance | Imbalanced (48.5x ratio: Teff=1260, Fallow=26) |
| Missing values | None |

### Feature Groups

**Soil Properties (6 features)**

| Feature | Description |
|---------|-------------|
| `Ph` | Soil pH level |
| `K` | Potassium content (mg/kg) |
| `P` | Phosphorus content (mg/kg) |
| `N` | Nitrogen content (%) |
| `Zn` | Zinc content (mg/kg) |
| `S` | Sulfur content (mg/kg) |

**Seasonal Weather (22 features)**

Humidity, max/min temperature, and precipitation — each split across 4 seasons (Winter, Spring, Summer, Autumn). Plus wind direction, ground wetness, cloud amount, wind speed range, and surface pressure.

**Categorical Feature**

`Soilcolor` — soil color category (Black, Brown, Dark Brown, etc.)

---

## Project Structure

```
crop-recommendation/
│
├── crop_recommendation_Project.ipynb   ← Main notebook
├── Crop Recommendation using Soil...csv ← Dataset
├── crop_model_best.pkl                 ← Saved trained model
├── le_crop.pkl                         ← Crop label encoder
├── le_soil.pkl                         ← Soil color encoder
├── requirements.txt                    ← Python dependencies
└── README.md
```

---

## Features

- **Exploratory Data Analysis** — class distribution, correlation heatmap, soil nutrient boxplots per crop
- **Proper preprocessing** — LabelEncoder for categorical targets, Pipeline-based StandardScaler (no leakage)
- **6 ML models compared** — Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting
- **Stratified cross-validation** — StratifiedKFold (5-fold) for reliable performance estimates
- **Hyperparameter tuning** — GridSearchCV on best model
- **Feature importance analysis** — top 15 most predictive features
- **Full evaluation** — per-class classification report + confusion matrix

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/crop-recommendation.git
cd crop-recommendation
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas>=1.4.0
numpy>=1.22.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
jupyter>=1.0.0
```

### 4. Place the dataset

Put `Crop Recommendation using Soil Properties and Weather Prediction.csv` in the project root directory and update the file path in Cell 4 of the notebook:

```python
# Change this line in Cell 4:
df = pd.read_csv("Crop Recommendation using Soil Properties and Weather Prediction.csv")
```

---

## Usage

### Run the full notebook

```bash
jupyter notebook crop_recommendation_Project.ipynb
```

Run cells top to bottom using **Run All** (`Kernel → Restart & Run All`).

### Load the saved model for prediction

```python
import joblib
import pandas as pd

# Load saved artifacts
model   = joblib.load('crop_model_best.pkl')
le_crop = joblib.load('le_crop.pkl')
le_soil = joblib.load('le_soil.pkl')

# Example: predict crop for a new soil/weather sample
sample = pd.DataFrame([{
    'Ph': 5.8, 'K': 300, 'P': 12, 'N': 0.18, 'Zn': 1.8, 'S': 11,
    'Soilcolor_enc': le_soil.transform(['Brown'])[0],
    # ... add all other feature values
}])

prediction = le_crop.inverse_transform(model.predict(sample))
print(f"Recommended crop: {prediction[0]}")
```

---

## Methodology

```
Raw CSV
   ↓
Rename columns → encode Soilcolor (LabelEncoder) → encode crop label (LabelEncoder)
   ↓
Stratified train/test split (80/20, random_state=42)
   ↓
Pipeline: StandardScaler → Classifier  ← fitted ONLY on training folds
   ↓
StratifiedKFold (5-fold) cross-validation on train set
   ↓
Final evaluation on held-out test set
   ↓
GridSearchCV hyperparameter tuning on best model
   ↓
Save best pipeline with joblib
```

**Why Pipeline?** Using `sklearn.pipeline.Pipeline` ensures the `StandardScaler` is fit only on training data during each cross-validation fold. Without this, the scaler sees test data during `fit_transform` on the full dataset — a form of data leakage that artificially inflates reported accuracy.

---

## Results

### Model Comparison (5-fold CV on training set)

| Model | CV Accuracy | Test Accuracy |
|-------|------------|---------------|
| Logistic Regression | ~0.34 | ~0.35 |
| KNN | ~0.42 | ~0.43 |
| Decision Tree | ~0.44 | ~0.45 |
| SVM | ~0.46 | ~0.47 |
| Gradient Boosting | ~0.50 | ~0.51 |
| **Random Forest** | **~0.51** | **~0.51** |

### After Hyperparameter Tuning (GridSearchCV)

```
Best params:  max_depth=10, min_samples_split=2, n_estimators=200
Best CV acc:  0.5118
Test acc:     0.5142
```

### Most Important Features

From Random Forest feature importance analysis:

1. `K` — Potassium content
2. `P` — Phosphorus content
3. `Ph` — Soil pH
4. `S` — Sulfur content
5. `Zn` — Zinc content

Soil nutrient features significantly outperform weather features for crop discrimination.

---

## Key Fixes & Improvements

This notebook corrects several common mistakes found in beginner ML pipelines:

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | **Data leakage** — scaler fit on full data before split | `Pipeline` ensures scaler fits only on train folds |
| 2 | **Scaling never used** — scaled arrays created but models trained on raw data | All models now use Pipeline with scaler |
| 3 | **Wrong confusion matrix** — `y_pred` from last loop model, not best model | Confusion matrix explicitly tied to best model |
| 4 | **No cross-validation** — single split gives unreliable estimates | `StratifiedKFold(n_splits=5)` added |
| 5 | **No hyperparameter tuning** — all default params | `GridSearchCV` added for best model |
| 6 | **Manual label dict for crops** — fragile, typo-prone | Replaced with `LabelEncoder` |
| 7 | **Only accuracy reported** — hides per-class performance | Full `classification_report` added |
| 8 | **No feature importance** | Top-15 feature importance chart added |

---

## Limitations

> **Honest assessment of model performance**

- **~51% accuracy is near the ceiling for this dataset.** This is not a code problem — it is a data problem.
- The dataset has severe **class imbalance** (Teff: 1,260 samples vs Fallow: 26 samples, a 48.5x ratio). Minority classes like Fallow, Red Pepper, Niger Seed, and Potato have too few samples to learn reliable patterns.
- **Feature overlap is high.** Soil pH ranges only 5.2–6.6 and nitrogen only 0.13–0.24 across all 12 crops — very small differences that are hard to discriminate.
- **Weather features contribute little.** A model trained on weather features alone scores only ~11.5% accuracy.
- The majority class baseline (always predict Teff) already achieves ~32.6%, so the model's real gain over naive guessing is modest.

---

## Future Work

To meaningfully improve accuracy, the following are recommended:

- **Collect more data** for minority classes — especially Fallow (26), Red Pepper (29), Niger Seed (64), and Potato (48)
- **Add more discriminative features** — crop growth season length, required rainfall range, soil texture (clay/silt/sand), elevation
- **Try SMOTE oversampling** from `imbalanced-learn` to synthetically balance classes
- **Use `class_weight='balanced'`** in tree-based models to reduce majority-class bias
- **Feature engineering** — seasonal temperature ranges, soil nutrient ratios (N:P:K), seasonal wetness averages
- **Hierarchical classification** — first predict crop family (cereal / legume / root), then specific crop within family






---

*Built as a learning project in agricultural machine learning. Contributions and feedback welcome via Issues and Pull Requests.*
