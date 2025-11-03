# ML-Ready Dataset Documentation

## Overview
This dataset has been processed and cleaned for immediate use in machine learning models to predict Pitzer coefficients from electrolyte properties.

## Files Generated

### Main Files
- **`ml_ready_dataset.csv`**: Complete dataset (30 electrolytes × 35 columns)
- **`ml_ready_dataset_X.csv`**: Feature matrix only (30 features)
- **`ml_ready_dataset_y.csv`**: Target matrix only (4 Pitzer coefficients)

### Processing Script
- **`prepare_ml_data.py`**: Script that processes the clean activity dataset

## Dataset Specifications

### Size
- **30 electrolytes** (observations)
- **30 features** (X variables)
- **4 targets** (y variables - Pitzer coefficients)
- **100% complete** - no missing values

### Data Quality
- ✅ All columns are numeric (except electrolyte_name identifier)
- ✅ Sparse columns removed (kept only columns with ≥70% data)
- ✅ Non-numeric columns removed (formulas, SMILES, text descriptions)
- ✅ Missing values imputed using column means
- ✅ Concise, ML-friendly column names

## Feature Breakdown

### Molecular Features (5 features)
1. `mol_molar_mass` - Molecular molar mass (g/mol)
2. `mol_is_organic` - Binary: 1 if organic, 0 if not
3. `mol_is_acid` - Binary: 1 if acid, 0 if not
4. `mol_ionic_species` - Number of ionic species
5. `mol_solubility` - Saltwater solubility limit (g/100mL water, 20°C)

### Cation Features (16 features)
1. `cat_molar_mass` - Cation molar mass (g/mol)
2. `cat_charge` - Formal charge
3. `cat_valence_e` - Number of valence electrons
4. `cat_hydrated_r` - Hydrated radius (nm)
5. `cat_ionic_r` - Ionic radius (pm)
6. `cat_hydration_num` - Hydration number
7. `cat_hydration_h` - Hydration enthalpy (kJ/mol)
8. `cat_hsab` - HSAB type (1=hard, 0=soft)
9. `cat_atomic_num` - Atomic number
10. `cat_pt_group` - Periodic table group
11. `cat_pt_period` - Periodic table period
12. `cat_is_polyatomic` - Binary: 1 if polyatomic, 0 if not
13. `cat_electroneg` - Electronegativity
14. `cat_electron_aff` - Electron affinity (kJ/mol)
15. `cat_ionization_e` - Ionization energy (eV)
16. `cat_molar_mass_2` - Secondary molar mass value

### Anion Features (9 features)
1. `an_charge` - Anion charge
2. `an_valence_e_neutral` - Valence electrons (neutral state)
3. `an_hydrated_r` - Hydrated radius (nm)
4. `an_ionic_r` - Ionic radius (pm)
5. `an_hydration_num` - Hydration number
6. `an_hsab` - HSAB type (1=hard, 0=soft, 0.5=borderline)
7. `an_is_polyatomic` - Binary: 1 if polyatomic, 0 if not
8. `an_electron_aff` - Electron affinity (kJ/mol)
9. `an_ionization_e` - Ionization energy (eV)

### Target Variables (4 features)
1. `beta_0` - First Pitzer parameter
2. `beta_1` - Second Pitzer parameter
3. `beta_2` - Third Pitzer parameter
4. `c_mx` - Pitzer mixing parameter

## Usage Example

### Python with pandas and scikit-learn

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Option 1: Load complete dataset
df = pd.read_csv('ml_ready_dataset.csv')

# Separate features and targets
feature_cols = [col for col in df.columns if col not in ['electrolyte_name', 'beta_0', 'beta_1', 'beta_2', 'c_mx']]
target_cols = ['beta_0', 'beta_1', 'beta_2', 'c_mx']

X = df[feature_cols]
y = df[target_cols]

# Option 2: Load pre-separated files
X = pd.read_csv('ml_ready_dataset_X.csv').drop('electrolyte_name', axis=1)
y = pd.read_csv('ml_ready_dataset_y.csv').drop('electrolyte_name', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (example: Random Forest for multi-output regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error
for i, target in enumerate(target_cols):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    rmse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i], squared=False)
    print(f"{target}: R² = {r2:.3f}, RMSE = {rmse:.4f}")
```

## Processing Details

### Removed Items
**Sparse Columns** (< 70% data filled):
- Cation polarizability
- Anion polarizability
- Various property measurements with insufficient data

**Non-Numeric Columns**:
- `Molecular_Formula` (text)
- `Molecular_SMILES representation` (text)
- `Cation_Name` (text)
- `Anion_Name` (text)
- `Dipole moment` columns (text with "non-zero" entries)

### Missing Value Handling
- Strategy: Mean imputation
- Applied to: Numeric columns with missing values
- Result: 100% complete dataset (39 missing values → 0)

## Electrolytes Included

All 30 electrolytes from the clean dataset:
NaF, NaCl, NaBr, NaI, NaClO4, NaNO3, HCl, HBr, HI, HClO4, HNO3, LiOH, LiCl, LiBr, LiI, LiClO4, LiNO3, AgNO3, NaOH, NaClO3, KF, CsNO2, KNO2, NaNO2, LiNO2, ZnCl2, CdClO4, NiCl2, CoCl2, and one more.

## Model Recommendations

### Suitable Algorithms
Given the small dataset size (30 samples), consider:

1. **Ensemble Methods**
   - Random Forest (handles non-linearity, robust to overfitting)
   - Gradient Boosting (XGBoost, LightGBM)
   
2. **Linear Methods with Regularization**
   - Ridge Regression (L2 regularization)
   - Lasso (L1 regularization, feature selection)
   - Elastic Net (combination of L1 and L2)

3. **Support Vector Machines**
   - SVR with RBF kernel (good for small datasets)

4. **Neural Networks** (with caution)
   - Small networks with dropout
   - Consider transfer learning if similar data available

### Best Practices
1. **Cross-validation**: Use k-fold cross-validation (k=5 or 10) instead of simple train/test split
2. **Feature scaling**: Standardize features (zero mean, unit variance)
3. **Feature selection**: Consider PCA or feature importance analysis
4. **Regularization**: Essential given small sample size
5. **Multi-output vs Single-output**: Can train separate models for each Pitzer coefficient or use multi-output regression

### Avoiding Overfitting
- Use regularization (L1/L2)
- Limit model complexity
- Cross-validation for hyperparameter tuning
- Consider domain knowledge for feature engineering

## Notes

- **Small dataset**: 30 samples may limit model generalization. Consider data augmentation or transfer learning if possible.
- **Balanced features**: 5 molecular + 16 cation + 9 anion features provide good coverage of electrolyte properties.
- **Target correlation**: Pitzer coefficients may be correlated; consider this in modeling approach.

## Citation

If you use this processed dataset, please cite the original activity database sources and acknowledge the data processing pipeline.

