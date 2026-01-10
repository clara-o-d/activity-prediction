# Activity Prediction - Electrolyte Pitzer Coefficients

This project predicts Pitzer coefficients (Beta(0) and Beta(1)) for electrolytes using machine learning. The baseline empirical model uses literature correlations based on ionic charges and radii, while ML models leverage additional ion properties to improve predictions.

## Project Structure

```
activity-prediction/
├── data/                                    # Data files
│   └── baseline_with_ion_properties.csv    # Main dataset with merged baseline and ion properties
├── data-processing/                        # Data processing scripts
│   └── merge_baseline_with_ion_properties.py
├── model/                                  # Model training and evaluation
│   ├── train_baseline_ion_model.py         # Main training script
│   ├── neuralnet.py                        # Neural network template
│   ├── best_baseline_ion_model.pkl         # Trained model
│   ├── baseline_empirical_correlations.png
│   ├── baseline_ion_enriched_predictions.png
│   └── best_model_test_predictions.png
├── analysis/                               # Analysis scripts
│   ├── pca_analysis.py                     # PCA on full dataset
│   ├── pca_analysis_baseline.py            # PCA on baseline features only
│   └── pca_results/                         # PCA visualization outputs
└── misc./                                  # Miscellaneous files
```

## Setup

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow
```

## Workflow

### 1. Merge baseline data with ion properties

The baseline dataset gets enriched with additional ion properties from external sources:

```bash
python data-processing/merge_baseline_with_ion_properties.py
```

This merges the baseline electrolyte data with ion property data, matching on molecule names. The output is saved to `data/baseline_with_ion_properties.csv`.

### 2. Train models

Train multiple models and compare their performance:

```bash
python model/train_baseline_ion_model.py
```

The script trains several models:
- **Baseline Empirical**: Uses literature equations based on ionic charges and radii. No training required - just applies the empirical formulas.
- **Random Forest**: Tree-based ensemble model
- **Gradient Boosting**: Boosting ensemble
- **Ridge, Lasso, Elastic Net**: Regularized linear models

Models are evaluated on validation and test sets. The best model (selected based on validation performance) is saved to `model/best_baseline_ion_model.pkl`.

The baseline empirical model uses these equations:
- Beta(0) = 0.04850 × Z_M^1.62 × Z_X^-1.35 × |r_M - 1.5×r_X|^1.2 + 0.03898
- Beta(1) = Z_X^0.4 × (0.00738 × X² + 0.16800 × X - 0.09320), where X = Z_M² × Z_X^0.6 × (1 + |r_M - 1.2×r_X|^0.2)

### 3. PCA analysis

Run principal component analysis to explore the data:

```bash
python analysis/pca_analysis.py              # Full dataset
python analysis/pca_analysis_baseline.py     # Baseline features only
```

These generate visualizations showing data structure, component loadings, and variance explained.

## Outputs

After training, you'll get:
- `model/best_baseline_ion_model.pkl`: The trained model and scaler (pickled)
- `model/baseline_empirical_correlations.png`: Shows how well the empirical equations fit the data
- `model/baseline_ion_enriched_predictions.png`: Prediction plots for all models on validation set
- `model/best_model_test_predictions.png`: Comparison of best model vs baseline on test set

The training script prints detailed metrics including R², RMSE, and MAE for each model and target variable.