# Baseline Model Folder

This folder contains all baseline model files organized in one place.

## 📁 Folder Contents

```
model/baseline/
├── baseline.py                      # Main baseline model implementation
├── baseline_model.pkl               # Trained baseline model
├── BASELINE_README.md               # Detailed documentation
├── baseline_predictions.png         # Prediction vs truth plots
├── empirical_correlations.png       # Empirical correlation plots (Figures 3 & 5 style)
├── baseline_comparison.png          # Baseline model comparison
└── ml_comparison.png                # ML model comparison
```

## 🚀 Quick Start

### Train the baseline model:
```bash
python model/baseline/baseline.py
```

This will:
- Train the empirical baseline model
- Generate all plots
- Save the trained model

### Compare baseline vs ML model:
```bash
python model/compare_models.py
```

## 📊 Generated Plots

### 1. `empirical_correlations.png`
Shows the empirical relationships between Pitzer parameters and ionic properties (matching your reference Figures 3 and 5):

**Left plot**: Beta(0) vs Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2
- Different electrolyte types shown with distinct markers
- Linear regression fit
- R² = 0.31

**Right plot**: Beta(1) * Z_X^-0.4 vs Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)^2
- Polynomial (quadratic) regression fit
- R² = 0.78
- Shows stronger correlation than Beta(0)

### 2. `baseline_predictions.png`
Standard prediction vs truth plots for both Beta(0) and Beta(1)

### 3. `baseline_comparison.png` & `ml_comparison.png`
Side-by-side comparison of baseline and ML model performance

## 🔍 Key Results

| Parameter | R² | RMSE | MAE |
|-----------|-----|------|-----|
| Beta(0) | 0.31 | 0.133 | 0.102 |
| Beta(1) | 0.81 | 0.375 | 0.221 |
| **Overall** | **0.83** | **0.281** | **0.161** |

## 📖 Documentation

See `BASELINE_README.md` for complete documentation including:
- Theoretical background
- Implementation details
- Performance analysis
- Comparison with ML model
- Usage examples

## 🔗 Import from Other Scripts

```python
import sys
sys.path.append('model/baseline')
from baseline import PitzerBaselineModel

# Use the model
model = PitzerBaselineModel()
model.fit(X, y)
predictions = model.predict(X_test)
```

## ✨ Features

- Physics-based empirical correlations
- Interpretable relationships
- Fast computation
- Serves as baseline for ML model comparison
- Matches literature correlation styles

