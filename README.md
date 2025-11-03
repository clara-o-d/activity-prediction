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
│   ├── train_model.py
│   ├── predict_new.py
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

### 3. Train Models

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

### 4. Make Predictions

Use trained model for predictions:

```bash
python model/predict_new.py
```

## 📊 Current Results

**Dataset**: 36 electrolytes with 27 features  
**Best Model**: Random Forest  
**Average Test R²**: 0.55  
**Cross-Validation R²**: 0.55 (± 0.27)

### Performance by Target

- **beta_0**: Good predictions (R² ≈ 0.75)
- **beta_1**: Moderate predictions (R² ≈ 0.25)
- **beta_2**: Perfect (all zeros in data)
- **c_mx**: Challenging (R² ≈ 0.18)

### Top 5 Important Features

1. Molecular ionic species count (22.8%)
2. Cation charge (18.9%)
3. Cation hydration enthalpy (14.1%)
4. Anion charge (9.3%)
5. Cation valence electrons (8.5%)

## 🔧 Custom Usage

### Load and use the trained model

```python
import pickle
import pandas as pd
import os

# Load model
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, 'model', 'best_pitzer_model.pkl')

with open(model_path, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

# Load features for new electrolyte
data_dir = os.path.join(project_root, 'data')
X = pd.read_csv(os.path.join(data_dir, 'ml_ready_dataset_X.csv'))

# Make predictions
X_scaled = scaler.transform(X.iloc[:, 1:])  # Skip electrolyte_name
predictions = model.predict(X_scaled)

# predictions = [beta_0, beta_1, beta_2, c_mx]
```

### Predict for custom electrolyte

```python
# Prepare your electrolyte features (must match training data exactly)
X_new = pd.DataFrame({
    'mol_molar_mass': [58.44],
    'mol_is_organic': [0],
    'mol_is_acid': [0],
    'mol_ionic_species': [2],
    'mol_solubility': [35.9],
    # ... all 27 features in order
})

# Scale and predict
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

## 📈 Model Details

### Features (27 total)
- **Molecular** (5): mass, organic flag, acid flag, ionic species count, solubility
- **Cation** (15): mass, charge, valence, radii, hydration properties, periodic table, etc.
- **Anion** (7): charge, valence, radii, hydration properties, etc.

### Targets (4)
- `beta_0`: First Pitzer parameter
- `beta_1`: Second Pitzer parameter
- `beta_2`: Third Pitzer parameter (all zeros in current data)
- `c_mx`: Pitzer mixing parameter

### Models Compared
1. Random Forest (Best) - R² = 0.55
2. Gradient Boosting - R² = 0.53
3. Lasso - R² = 0.37
4. Elastic Net - R² = 0.30
5. Ridge - R² = 0.12

## 📝 Notes

- **Limited data**: Only 36 samples; model performance improves with more data
- **Best for beta_0**: Most reliable predictions for first Pitzer parameter
- **Use for screening**: Good for candidate ranking, not high-precision calculations
- **Physical features**: Model identifies meaningful physical properties as important

## 🎯 Next Steps

1. Collect more electrolyte data
2. Try separate models for beta_1 and c_mx
3. Feature engineering (polynomial, interactions)
4. Physics-informed machine learning
5. Ensemble methods

## 📚 Files

- `data-processing/process_activity_data.py` - Convert raw → clean data
- `data-processing/prepare_ml_data.py` - Convert clean → ML-ready data
- `model/train_model.py` - Train and evaluate models
- `model/predict_new.py` - Make predictions with trained model

