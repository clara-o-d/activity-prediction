# Activity Prediction - Electrolyte Pitzer Coefficients

Machine learning pipeline for predicting Pitzer coefficients from electrolyte properties.

## 📁 Project Structure

```
activity-prediction/
├── data/                          # Data files
│   ├── Activity Database.xlsx - Electrolyte Properties.csv
│   ├── Activity Database.xlsx - Electrolyte Activity.csv
│   ├── clean_activity_dataset.csv
│   ├── ml_ready_dataset.csv
│   ├── ml_ready_dataset_X.csv
│   └── ml_ready_dataset_y.csv
├── data-processing/               # Data processing scripts
│   ├── process_activity_data.py
│   └── prepare_ml_data.py
├── model/                         # Model training and prediction
│   ├── baseline/                  # Baseline model folder
│   │   ├── baseline.py            # Baseline implementation
│   │   ├── baseline_model.pkl     # Trained baseline model
│   │   ├── BASELINE_README.md     # Baseline documentation
│   │   └── empirical_correlations.png
│   ├── train_model.py
│   ├── predict_new.py
│   ├── compare_models.py          # Compare baseline vs ML model
│   ├── best_pitzer_model.pkl
│   └── prediction_plots.png
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 1. Process Raw Data

Convert raw electrolyte data to clean dataset:

```bash
python data-processing/process_activity_data.py
```

This will:
- Read electrolyte properties and Pitzer coefficients
- Filter electrolytes with complete data
- Match electrolytes between datasets
- Output: `data/clean_activity_dataset.csv`

### 2. Prepare ML-Ready Data

Convert clean data to ML-ready format:

```bash
python data-processing/prepare_ml_data.py
```

This will:
- Remove non-numeric columns
- Remove sparse columns (<70% filled)
- Rename columns to concise names
- Handle missing values
- Output: `data/ml_ready_dataset.csv` and separate X/y files

### 3. Train Baseline Model (Optional)

Train the empirical baseline model based on ionic charge and radius correlations:

```bash
python model/baseline/baseline.py
```

This will:
- Train a physics-based empirical model using correlations from literature
- Uses only ionic charge and radius features
- Beta(0) = a * Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2 + b
- Beta(1) follows a polynomial relationship with charge/radius features
- Output: `model/baseline_model.pkl` and prediction plots

**Performance:**
- Overall R²: 0.83
- Beta(0) R²: 0.31
- Beta(1) R²: 0.81

### 4. Train ML Models

Train and evaluate multiple ML models:

```bash
python model/train_model.py
```

This will:
- Train 5 different models (RF, GB, Ridge, Lasso, Elastic Net)
- Evaluate on test set
- Perform cross-validation
- Generate prediction plots
- Save best model: `model/best_pitzer_model.pkl`

**Performance:**
- Overall R²: 0.90
- Beta(0) R²: 0.76
- Beta(1) R²: 0.88

### 5. Compare Baseline vs ML Model

Compare the performance of both models:

```bash
python model/compare_models.py
```

This will:
- Load both baseline and ML models
- Evaluate on their respective test sets
- Generate comparison metrics and plots
- Show improvement achieved by ML approach

**Key Improvements (Baseline → ML):**
- Beta(0): R² 0.31 → 0.76 (145% improvement)
- Beta(1): R² 0.81 → 0.88 (9% improvement)
- Overall RMSE: 26% reduction

### 6. Make Predictions

Use trained model for predictions:

```bash
python model/predict_new.py
```