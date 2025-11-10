# Baseline Model for Pitzer Parameter Prediction

## Overview

This baseline model implements empirical correlations from the literature that relate Pitzer parameters (Beta(0) and Beta(1)) to fundamental ionic properties: charge and ionic radius.

## Theoretical Background

The baseline model is based on the observation that Pitzer parameters follow systematic trends with ionic charge (Z) and ionic radius (r). These correlations were identified through analysis of experimental data and are presented in the reference figures.

### Beta(0) Correlation

Linear relationship:
```
Beta(0) = a * Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2 + b
```

Where:
- Z_M = cation charge (absolute value)
- Z_X = anion charge (absolute value)
- r_M = cation ionic radius (pm)
- r_X = anion ionic radius (pm)

### Beta(1) Correlation

Polynomial relationship:
```
Beta(1) * Z_X^-0.4 = a * X^2 + b * X + c

where X = Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)^2
```

Therefore:
```
Beta(1) = Z_X^0.4 * (a * X^2 + b * X + c)
```

## Implementation

The model is implemented as a scikit-learn compatible class `PitzerBaselineModel` with standard `fit()` and `predict()` methods.

### Key Features

1. **Physics-based**: Uses fundamental ionic properties
2. **Interpretable**: Clear relationship between inputs and outputs
3. **Lightweight**: Only 2 features per ion (charge and radius)
4. **Fast**: Simple polynomial/linear calculations

### Training

The model fits the empirical correlations to the available data using linear regression:
- Beta(0): Linear regression on the computed feature
- Beta(1): Polynomial regression (quadratic) on the computed feature

## Performance

Evaluated on 60 electrolyte samples with complete ionic radius data:

| Metric | Beta(0) | Beta(1) | Overall |
|--------|---------|---------|---------|
| R²     | 0.3149  | 0.8104  | 0.8318  |
| RMSE   | 0.1328  | 0.3746  | 0.2810  |
| MAE    | 0.1016  | 0.2209  | 0.1613  |

### Strengths

- **Beta(1)**: Very good performance (R² = 0.81), capturing most of the variance
- **Simple**: Easy to understand and interpret
- **Physical basis**: Grounded in ionic theory

### Limitations

- **Beta(0)**: Moderate performance (R² = 0.31), misses some complexity
- **Limited features**: Only uses charge and radius, ignoring other important properties
- **Missing data**: Requires ionic radius data, which is not available for all electrolytes

## Comparison with ML Model

The ML model (gradient boosting) significantly outperforms the baseline:

| Metric | Baseline | ML Model | Improvement |
|--------|----------|----------|-------------|
| Overall R² | 0.8318 | 0.9016 | +0.0698 |
| Beta(0) R² | 0.3149 | 0.7596 | +0.4447 |
| Beta(1) R² | 0.8104 | 0.8813 | +0.0709 |
| Overall RMSE | 0.2810 | 0.2085 | -25.8% |

### Key Insights

1. **Beta(0)** shows the largest improvement (R² 0.31 → 0.76), suggesting that charge and radius alone are insufficient
2. **Beta(1)** shows modest improvement (R² 0.81 → 0.88), indicating the empirical correlation captures most of the physics
3. The ML model benefits from using additional ionic properties (polarizability, hydration enthalpy, electronegativity, etc.)

## Usage

### Training

```python
import sys
sys.path.append('model/baseline')
from baseline import PitzerBaselineModel
import pandas as pd

# Load data
data = pd.read_csv('data/clean_activity_dataset.csv')

# Prepare features and targets
X = data[['Cation_Charge (formal)', 'Anion_Charge',
          'Cation_Ionic radius (pm) (molecular radius if polyatomic)',
          'Anion_Ionic radius (pm) (molecular radius if polyatomic).1']]
y = data[['Beta(0)', 'Beta(1)']]

# Train model
model = PitzerBaselineModel()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

### Quick Start

```bash
# Train fitted baseline model (fits equations to your data)
python model/baseline/baseline.py

# Evaluate literature empirical model (uses exact literature coefficients)
python model/baseline/baseline_empirical.py

# Compare with ML model
python model/compare_models.py
```

## Two Baseline Versions

1. **`baseline.py`** - Fitted baseline: Fits the empirical equation form to YOUR dataset
2. **`baseline_empirical.py`** - Literature baseline: Uses exact coefficients from literature (no fitting)

## Files

All baseline files are organized in `model/baseline/`:

- `baseline.py` - Fitted baseline implementation
- `baseline_empirical.py` - Literature empirical baseline (no fitting)
- `baseline_model.pkl` - Saved fitted model
- `empirical_model.pkl` - Saved literature empirical model
- `baseline_predictions.png` - Fitted model predictions
- `empirical_predictions.png` - Literature model predictions
- `empirical_correlations.png` - Empirical correlations (Figures 3 & 5 style)
- `empirical_literature_fit.png` - Literature equations applied to data

## Empirical Correlation Plots

The `empirical_correlations.png` file shows two key relationships:

1. **Left plot**: Beta(0) vs the computed feature Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2
   - Shows different electrolyte types with distinct markers (1-1, 2-1, 3-1, 4-1, 2-2)
   - Linear regression fit with R² and equation displayed
   - Emulates Figure 3 from the literature

2. **Right plot**: Beta(1) * Z_X^-0.4 vs the computed feature Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)^2
   - Shows the polynomial (quadratic) relationship
   - Different electrolyte types color-coded
   - Emulates Figure 5 from the literature

These plots demonstrate how well the empirical correlations capture the underlying physics of ion-ion interactions in our dataset.

## References

The empirical correlations are based on systematic analysis of Pitzer parameter data across different electrolyte types, as shown in the provided reference figures (Figure 3 and Figure 5).

## Conclusion

The baseline model provides a solid foundation for comparison and demonstrates that:

1. Physics-based correlations can capture significant variance (R² = 0.83)
2. Simple charge/radius features work well for Beta(1) but not Beta(0)
3. ML models can significantly improve predictions by leveraging additional ionic properties
4. The 26% RMSE reduction achieved by the ML model justifies the added complexity

This baseline establishes that our ML approach provides meaningful improvements over traditional empirical methods.

