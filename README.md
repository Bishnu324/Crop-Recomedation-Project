# Crop-Recomedation-Project
This project builds a multi-class crop classification system using agronomic and meteorological features. Given soil nutrient levels (N, P, K, pH, Zn, S), soil color, and seasonal weather variables (temperature, humidity, precipitation, wind, pressure), the model predicts the most suitable crop to grow.



## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Bugs Fixed from Original](#bugs-fixed-from-original)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Importance](#feature-importance)
- [Contributing](#contributing)

---

## Overview

This project builds a multi-class crop classification system using agronomic and meteorological features. Given soil nutrient levels (N, P, K, pH, Zn, S), soil color, and seasonal weather variables (temperature, humidity, precipitation, wind, pressure), the model predicts the most suitable crop to grow.

**Key improvements over the baseline:** proper train/test isolation via sklearn Pipelines, stratified cross-validation, hyperparameter tuning with GridSearchCV, and full evaluation metrics beyond simple accuracy.

---

## Dataset

**File:** `Crop Recommendation using Soil Properties and Weather Prediction.csv`

| Feature Group | Columns |
|---|---|
| Soil nutrients | `N`, `P`, `K`, `Ph`, `Zn`, `S` |
| Soil type | `Soilcolor` |
| Seasonal humidity | `QV2M-W`, `QV2M-Sp`, `QV2M-Su`, `QV2M-Au` |
| Seasonal temperature (max/min) | `T2M_MAX-*`, `T2M_MIN-*` (W/Sp/Su/Au) |
| Seasonal precipitation | `PRECTOTCORR-*` (W/Sp/Su/Au) |
| Atmospheric | `WD10M`, `GWETTOP`, `CLOUD_AMT`, `WS2M_RANGE`, `PS` |
| Target | `label` (crop type) |

Weather variable naming follows [NASA POWER](https://power.larc.nasa.gov/) conventions. Columns are renamed for readability during preprocessing.

---

## Bugs Fixed from Original

| # | Issue | Status |
|---|---|---|
| 1 | **Data leakage** — scaler fit on full dataset before split | ✅ Fixed via Pipeline |
| 2 | **Scaling never applied** — `normalized_data` created but unused | ✅ Scaling now inside Pipeline |
| 3 | **Wrong confusion matrix** — used `y_pred` from last loop iteration | ✅ Now uses best model's predictions |
| 4 | **No cross-validation** — single split gives unreliable estimates | ✅ 5-fold StratifiedKFold CV added |
| 5 | **No hyperparameter tuning** — all models used default params | ✅ GridSearchCV for Random Forest |
| 6 | **Manual label encoding** — hand-coded dict prone to typos | ✅ Replaced with `LabelEncoder` |
| 7 | **No classification report** — only accuracy shown | ✅ Full per-class precision/recall/F1 |
| 8 | **Redundant code** — `pd.DataFrame()` on an already-DataFrame | ✅ Cleaned up |
| 9 | **No feature importance analysis** | ✅ Added (Random Forest importances) |
| 10 | **No model saving** | ✅ Best model saved with `joblib` |

---

## Project Structure

```
crop-recommendation/
│
├── crop_recommendation_improved.ipynb   # Main notebook (this project)
├── Crop Recommendation using Soil Properties and Weather Prediction.csv
├── best_model.pkl                       # Saved best model (generated after run)
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/crop-recommendation.git
cd crop-recommendation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

**Python version:** 3.8+

---

## Usage

1. Place the dataset CSV in the project root directory.
2. Open the notebook:

```bash
jupyter notebook crop_recommendation_improved.ipynb
```

3. Run all cells sequentially. The notebook will:
   - Load and preprocess the data
   - Run 6 classifiers with 5-fold cross-validation
   - Display a model comparison chart
   - Print the full classification report for the best model
   - Show the confusion matrix and feature importances
   - Run GridSearchCV hyperparameter tuning on Random Forest
   - Save the best pipeline to `best_model.pkl`

**To predict on new data:**

```python
import joblib
import pandas as pd

model = joblib.load('best_model.pkl')

# Example input — match your dataset's feature columns
new_sample = pd.DataFrame([{
    'N': 90, 'P': 42, 'K': 43, 'Ph': 6.5, 'Zn': 1.2, 'S': 15,
    'Soilcolor_types': 2,
    'humidity_winter': 0.015, 'humidity_spring': 0.018,
    # ... add all features
}])

prediction = model.predict(new_sample)
print(f'Recommended crop: {prediction[0]}')
```

---

## Models & Results

Six classifiers are evaluated inside sklearn Pipelines (StandardScaler + model):

| Model | CV Mean Accuracy | CV Std | Test Accuracy |
|---|---|---|---|
| Logistic Regression | — | — | — |
| SVM | — | — | — |
| KNN | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |

> Results populate after running the notebook. The best model by CV accuracy is selected for detailed evaluation.

---

## Pipeline Architecture

```
Raw Data
   │
   ├─ LabelEncoder (crop_label → int)
   ├─ LabelEncoder (Soilcolor → int)
   │
   └─ Train/Test Split (80/20, stratified)
         │
         ├─ Pipeline per model:
         │     StandardScaler (fit on train fold only)
         │           │
         │     Classifier
         │
         ├─ StratifiedKFold CV (5 folds) on X_train
         └─ Final evaluation on X_test
```

Using `sklearn.pipeline.Pipeline` ensures the scaler is **never fit on test data**, eliminating data leakage.

---

## Feature Importance

Top predictive features are extracted from the Random Forest model post-training. The top 15 features are visualized as a horizontal bar chart. Typically, soil nutrient levels (N, P, K, pH) and seasonal precipitation variables rank highest.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improved feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

[MIT](LICENSE)
