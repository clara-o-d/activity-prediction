#!/usr/bin/env python3
"""
Train Machine Learning Models to Predict Pitzer Coefficients
Using Baseline Data Enriched with Ion Properties

This script demonstrates a complete ML workflow:
1. Load the baseline+ion properties dataset
2. Split data into train/test sets
3. Scale features
4. Train multiple models including baseline empirical model
5. Evaluate and compare performance
6. Save the best model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore')


class PitzerEmpiricalBaselineModel(BaseEstimator, RegressorMixin):
    """
    Baseline model using empirical equations from literature.
    
    This uses fixed literature coefficients (no training):
    - Beta(0) = 0.04850 * Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2 + 0.03898
    - Beta(1) = Z_X^0.4 * (0.00738 * X^2 + 0.16800 * X - 0.09320)
      where X = Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)
    
    These equations predict the ORIGINAL Pitzer parameters.
    """
    
    def __init__(self):
        # Beta(0) coefficients from Figure 3 (literature values)
        self.beta0_slope = 0.04850
        self.beta0_intercept = 0.03898
        
        # Beta(1) coefficients from Figure 5 (literature values)
        self.beta1_a2 = 0.00738   # coefficient of X^2
        self.beta1_a1 = 0.16800   # coefficient of X
        self.beta1_a0 = -0.09320  # constant term
        
    def compute_beta0_feature(self, Z_M, Z_X, r_M, r_X):
        """
        Compute the feature for Beta(0) prediction.
        Feature = Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2
        """
        Z_M = np.abs(Z_M)
        Z_X = np.abs(Z_X)
        feature = (Z_M**1.62) * (Z_X**(-1.35)) * (np.abs(r_M - 1.5*r_X)**1.2)
        return feature
    
    def compute_beta1_feature(self, Z_M, Z_X, r_M, r_X):
        """
        Compute the feature for Beta(1) prediction.
        Feature = Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)
        """
        Z_M = np.abs(Z_M)
        Z_X = np.abs(Z_X)
        feature = (Z_M**2) * (Z_X**0.6) * (1 + np.abs(r_M - 1.2*r_X)**0.2)
        return feature
    
    def fit(self, X, y):
        """
        Dummy fit method for compatibility with scikit-learn interface.
        Does not actually train - uses fixed literature coefficients.
        """
        print("  Note: Using fixed literature coefficients (no training)")
        return self
    
    def predict(self, X):
        """
        Predict Pitzer coefficients using empirical equations.
        
        Parameters:
        -----------
        X : DataFrame
            Must contain columns: electrolyte_type, r_M_angstrom, r_X_angstrom
        
        Returns:
        --------
        predictions : ndarray
            2D array with columns [B_MX_0_simplified, B_MX_1_simplified]
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            # Assume last columns are the required features
            X_df = pd.DataFrame(X)
        else:
            X_df = X
            
        # Extract charges from electrolyte_type_numeric
        # Format: 11 (1-1), 12 (1-2), 21 (2-1), 22 (2-2), 31 (3-1), 41 (4-1)
        if 'electrolyte_type_numeric' in X_df.columns:
            type_num = X_df['electrolyte_type_numeric'].values
            Z_M = type_num // 10  # First digit(s) = cation charge
            Z_X = type_num % 10   # Last digit = anion charge
        else:
            # Default to 1-1 if not available
            Z_M = np.ones(len(X_df))
            Z_X = np.ones(len(X_df))
        
        # Get radii (already in Angstroms in baseline data)
        r_M = X_df['r_M_angstrom'].values
        r_X = X_df['r_X_angstrom'].values
        
        # Predict Beta(0)
        feature_beta0 = self.compute_beta0_feature(Z_M, Z_X, r_M, r_X)
        beta0_pred = self.beta0_slope * feature_beta0 + self.beta0_intercept
        
        # Predict Beta(1)
        feature_beta1 = self.compute_beta1_feature(Z_M, Z_X, r_M, r_X)
        y_beta1_transformed = (self.beta1_a2 * feature_beta1**2 + 
                              self.beta1_a1 * feature_beta1 + 
                              self.beta1_a0)
        Z_X_abs = np.abs(Z_X)
        beta1_pred = y_beta1_transformed * (Z_X_abs**0.4)
        
        return np.column_stack([beta0_pred, beta1_pred])


def load_data(filepath='data/baseline_with_ion_properties.csv'):
    """Load the ML-ready baseline+ion dataset and separate features and targets."""
    print("=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Define target columns (Pitzer coefficients - using ORIGINAL for empirical equations)
    target_cols = ['B_MX_0_original', 'B_MX_1_original']
    
    # ID columns
    id_cols = ['electrolyte', 'electrolyte_type', 'cation', 'anion', 'molecule_formula']
    
    # Columns to exclude (reference info, type indicators, other coefficients)
    exclude_cols = [
        'ref_simplified', 'ref_original',  # Reference numbers
        'cation_type', 'anion_type',  # String type indicators
        'B_MX_0_simplified', 'B_MX_1_simplified',  # Simplified coefficients (not used)
        'B_MX_2_original', 'C_MX_phi_original'  # Additional original coefficients (not predicted by empirical equations)
    ]
    
    # Get all feature columns (for baseline empirical model)
    all_feature_cols = [col for col in df.columns 
                       if col not in id_cols + target_cols + exclude_cols]
    
    # ML models will use only the top features from importance analysis
    ml_feature_cols = [
        'electrolyte_type_numeric',
        'molecule_radius_vdw',
        'molecule_molecular_weight',
        'cation_1_molecular_weight',
        'r_X_angstrom',
        'molecule_n_atoms',
        'anion_1_radius_hydrated',
        'cation_1_radius_vdw',
        'anion_1_molecular_weight',
        'anion_1_n_atoms',
        'r_M_angstrom',
        'cation_type_numeric',
        'cation_1_radius_hydrated',
        'anion_1_radius_vdw',
        'anion_type_numeric'
    ]
    
    # Remove duplicates and verify all ML features exist in the dataset
    ml_feature_cols = list(dict.fromkeys(ml_feature_cols))  # Preserves order while removing duplicates
    missing_ml_features = [col for col in ml_feature_cols if col not in df.columns]
    if missing_ml_features:
        print(f"⚠️  Warning: Missing ML features: {missing_ml_features}")
        ml_feature_cols = [col for col in ml_feature_cols if col in df.columns]
    
    # Baseline empirical model needs: electrolyte_type_numeric, r_M_angstrom, r_X_angstrom
    # ML models will use a specific subset of features
    X_all = df[all_feature_cols]  # For baseline empirical model
    X_ml = df[ml_feature_cols]    # For ML models (subset of features)
    y = df[target_cols]
    molecules = df['electrolyte']
    
    print(f"✓ All features (for baseline): {X_all.shape[1]} columns")
    print(f"✓ ML features (subset): {X_ml.shape[1]} columns")
    print(f"  ML features: {', '.join(ml_feature_cols)}")
    print(f"✓ Targets: {y.shape[1]} columns ({', '.join(target_cols)})")
    print(f"✓ Missing values: {X_all.isna().sum().sum()} in all features, {X_ml.isna().sum().sum()} in ML features, {y.isna().sum().sum()} in targets")
    
    return X_all, X_ml, y, molecules, all_feature_cols, ml_feature_cols, target_cols


def explore_data(X_all, X_ml, y, all_feature_cols, ml_feature_cols, target_cols):
    """Display basic statistics about the dataset."""
    print("\n" + "=" * 80)
    print("Data Exploration")
    print("=" * 80)
    
    print("\nAll Features Statistics (first 10):")
    print(X_all.describe().T[['mean', 'std', 'min', 'max']].head(10))
    print("...")
    
    print("\nML Features Statistics (subset used for ML models):")
    print(X_ml.describe().T[['mean', 'std', 'min', 'max']])
    
    print("\nTarget Statistics:")
    print(y.describe().T[['mean', 'std', 'min', 'max']])
    
    print("\nFeature Groups (All Features):")
    baseline_features = [c for c in all_feature_cols if c in ['r_M_angstrom', 'r_X_angstrom', 
                                                          'std_dev_simplified', 'std_dev_original',
                                                          'max_molality_simplified', 'max_molality_original']]
    mol_features = [c for c in all_feature_cols if c.startswith('molecule_')]
    cat_features = [c for c in all_feature_cols if c.startswith('cation_')]
    an_features = [c for c in all_feature_cols if c.startswith('anion_')]
    type_features = [c for c in all_feature_cols if 'type_numeric' in c or 'electrolyte_type_numeric' in c]
    
    print(f"  Baseline properties: {len(baseline_features)}")
    print(f"  Molecular features: {len(mol_features)}")
    print(f"  Cation features: {len(cat_features)}")
    print(f"  Anion features: {len(an_features)}")
    print(f"  Type features: {len(type_features)}")
    
    print(f"\nML Models will use {len(ml_feature_cols)} features (subset)")


def split_and_scale_data(X_all, X_ml, y, train_size=0.75, val_size=0.18, test_size=0.07, random_state=42):
    """Split data into train/validation/test and scale features."""
    print("\n" + "=" * 80)
    print("Data Preparation (Train/Validation/Test Split)")
    print("=" * 80)
    
    # Use X_ml for splitting (same indices for both feature sets)
    # First split: separate training from temp (val + test)
    X_train_ml, X_temp_ml, y_train, y_temp = train_test_split(
        X_ml, y, test_size=(val_size + test_size), random_state=random_state
    )
    
    # Second split: separate validation from test
    val_prop = val_size / (val_size + test_size)
    X_val_ml, X_test_ml, y_val, y_test = train_test_split(
        X_temp_ml, y_temp, test_size=(1 - val_prop), random_state=random_state
    )
    
    # Get corresponding splits for X_all (baseline features) using same indices
    train_indices = X_train_ml.index
    val_indices = X_val_ml.index
    test_indices = X_test_ml.index
    
    X_train_all = X_all.loc[train_indices]
    X_val_all = X_all.loc[val_indices]
    X_test_all = X_all.loc[test_indices]
    
    print(f"✓ Train set: {len(X_train_ml)} samples ({train_size*100:.0f}%)")
    print(f"✓ Validation set: {len(X_val_ml)} samples ({val_size*100:.0f}%)")
    print(f"✓ Test set: {len(X_test_ml)} samples ({test_size*100:.0f}%)")
    print(f"✓ Total: {len(X_ml)} samples")
    
    # Scale ML features using only training data statistics
    scaler_ml = StandardScaler()
    X_train_ml_scaled = scaler_ml.fit_transform(X_train_ml)
    X_val_ml_scaled = scaler_ml.transform(X_val_ml)
    X_test_ml_scaled = scaler_ml.transform(X_test_ml)
    
    # Baseline features are NOT scaled (empirical model uses raw values)
    print(f"✓ ML features scaled (mean=0, std=1) using training data only")
    print(f"✓ Baseline features kept unscaled (for empirical model)")
    
    return (X_train_all, X_val_all, X_test_all, 
            X_train_ml, X_val_ml, X_test_ml,
            X_train_ml_scaled, X_val_ml_scaled, X_test_ml_scaled,
            y_train, y_val, y_test, scaler_ml)


def train_models(X_train_all, X_train_ml, X_train_ml_scaled, y_train, target_cols):
    """Train multiple models and return them."""
    print("\n" + "=" * 80)
    print("Training Models")
    print("=" * 80)
    
    models = {}
    
    # 1. Baseline Empirical Model (uses unscaled baseline features)
    print("\n[1/6] Creating Baseline Empirical Model...")
    empirical = PitzerEmpiricalBaselineModel()
    empirical.fit(X_train_all, y_train)  # Dummy fit
    
    # Print sample predictions for debugging
    print("\n  Sample predictions from Baseline Empirical Model:")
    sample_pred = empirical.predict(X_train_all[:5])
    for i in range(min(5, len(sample_pred))):
        print(f"    Sample {i+1}:")
        print(f"      Predicted Beta(0): {sample_pred[i, 0]:.4f}, Actual: {y_train.iloc[i, 0]:.4f}")
        print(f"      Predicted Beta(1): {sample_pred[i, 1]:.4f}, Actual: {y_train.iloc[i, 1]:.4f}")
    
    # Calculate and print prediction for AlCl3
    print("\n  Detailed calculation for AlCl3:")
    print("  " + "="*70)
    # Load original baseline_data.csv to find AlCl3 (it may not be in merged dataset)
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_path = os.path.join(project_root, 'data', 'baseline_data.csv')
    df_baseline = pd.read_csv(baseline_path, engine='python', na_values=['-', ''], 
                              keep_default_na=True, on_bad_lines='warn')
    
    # Find AlCl3
    alcl3_idx = df_baseline[df_baseline['electrolyte'] == 'AlCl3'].index
    if len(alcl3_idx) > 0:
        idx = alcl3_idx[0]
        # Extract electrolyte type from string like '3-1'
        electrolyte_type = df_baseline.loc[idx, 'electrolyte_type']
        parts = electrolyte_type.split('-')
        Z_M = int(parts[0])
        Z_X = int(parts[1])
        r_M = df_baseline.loc[idx, 'r_M_angstrom']
        r_X = df_baseline.loc[idx, 'r_X_angstrom']
        beta0_actual = df_baseline.loc[idx, 'B_MX_0_original']
        beta1_actual = df_baseline.loc[idx, 'B_MX_1_original']
        
        print(f"    AlCl3 Properties:")
        print(f"      Electrolyte type: {electrolyte_type} (Z_M={Z_M}, Z_X={Z_X})")
        print(f"      r_M (Al³⁺): {r_M:.2f} Angstroms")
        print(f"      r_X (Cl⁻): {r_X:.2f} Angstroms")
        print(f"      Actual B_MX_0_original: {beta0_actual:.4f}")
        print(f"      Actual B_MX_1_original: {beta1_actual:.4f}")
        print()
        
        # Calculate Beta(0)
        feat_beta0 = (Z_M**1.62) * (Z_X**(-1.35)) * (abs(r_M - 1.5*r_X)**1.2)
        beta0_pred = 0.04850 * feat_beta0 + 0.03898
        
        print(f"    Beta(0) Calculation:")
        print(f"      feature = Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2")
        print(f"              = {Z_M}^1.62 * {Z_X}^-1.35 * |{r_M:.2f} - 1.5*{r_X:.2f}|^1.2")
        print(f"              = {Z_M**1.62:.4f} * {Z_X**(-1.35):.4f} * {abs(r_M - 1.5*r_X):.4f}^1.2")
        print(f"              = {feat_beta0:.6f}")
        print(f"      Beta(0) = 0.04850 * {feat_beta0:.6f} + 0.03898")
        print(f"              = {beta0_pred:.4f}")
        print(f"      Actual Beta(0): {beta0_actual:.4f}")
        print(f"      Error: {abs(beta0_pred - beta0_actual):.4f} ({abs(beta0_pred - beta0_actual)/abs(beta0_actual)*100:.1f}%)")
        print()
        
        # Calculate Beta(1)
        feat_beta1 = (Z_M**2) * (Z_X**0.6) * (1 + abs(r_M - 1.2*r_X)**0.2)
        y_trans = 0.00738 * feat_beta1**2 + 0.16800 * feat_beta1 - 0.09320
        beta1_pred = y_trans * (Z_X**0.4)
        
        print(f"    Beta(1) Calculation:")
        print(f"      feature = Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)")
        print(f"              = {Z_M}^2 * {Z_X}^0.6 * (1 + |{r_M:.2f} - 1.2*{r_X:.2f}|^0.2)")
        print(f"              = {Z_M**2:.1f} * {Z_X**0.6:.4f} * (1 + {abs(r_M - 1.2*r_X):.4f}^0.2)")
        print(f"              = {feat_beta1:.6f}")
        print(f"      y_transformed = 0.00738 * {feat_beta1:.6f}^2 + 0.16800 * {feat_beta1:.6f} - 0.09320")
        print(f"                    = {y_trans:.6f}")
        print(f"      Beta(1) = {y_trans:.6f} * Z_X^0.4")
        print(f"              = {y_trans:.6f} * {Z_X**0.4:.4f}")
        print(f"              = {beta1_pred:.4f}")
        print(f"      Actual Beta(1): {beta1_actual:.4f}")
        print(f"      Error: {abs(beta1_pred - beta1_actual):.4f} ({abs(beta1_pred - beta1_actual)/abs(beta1_actual)*100:.1f}%)")
        print()
        print(f"    Summary:")
        print(f"      Beta(0): Predicted is {beta0_pred/beta0_actual*100:.1f}% of actual")
        print(f"      Beta(1): Predicted is {beta1_pred/beta1_actual*100:.1f}% of actual")
    else:
        print("    AlCl3 not found in dataset (may have been excluded during merge)")
    
    models['Baseline Empirical'] = {'model': empirical, 'use_scaled': False, 'use_all_features': True}
    print("  ✓ Complete (no training - using literature coefficients)")
    
    # 2. Random Forest (uses ML feature subset)
    print("\n[2/6] Training Random Forest...")
    print(f"  Using {X_train_ml_scaled.shape[1]} ML features")
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )
    rf.fit(X_train_ml_scaled, y_train)
    models['Random Forest'] = {'model': rf, 'use_scaled': True, 'use_all_features': False}
    print("  ✓ Complete")
    
    # 3. Gradient Boosting (uses ML feature subset)
    print("\n[3/6] Training Gradient Boosting...")
    print(f"  Using {X_train_ml_scaled.shape[1]} ML features")
    gb = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_multi = MultiOutputRegressor(gb)
    gb_multi.fit(X_train_ml_scaled, y_train)
    models['Gradient Boosting'] = {'model': gb_multi, 'use_scaled': True, 'use_all_features': False}
    print("  ✓ Complete")
    
    # 4. Ridge Regression (uses ML feature subset)
    print("\n[4/6] Training Ridge Regression...")
    print(f"  Using {X_train_ml_scaled.shape[1]} ML features")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_ml_scaled, y_train)
    models['Ridge'] = {'model': ridge, 'use_scaled': True, 'use_all_features': False}
    print("  ✓ Complete")
    
    # 5. Lasso Regression (uses ML feature subset)
    print("\n[5/6] Training Lasso Regression...")
    print(f"  Using {X_train_ml_scaled.shape[1]} ML features")
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
    lasso.fit(X_train_ml_scaled, y_train)
    models['Lasso'] = {'model': lasso, 'use_scaled': True, 'use_all_features': False}
    print("  ✓ Complete")
    
    # 6. Elastic Net (uses ML feature subset)
    print("\n[6/6] Training Elastic Net...")
    print(f"  Using {X_train_ml_scaled.shape[1]} ML features")
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
    elastic.fit(X_train_ml_scaled, y_train)
    models['Elastic Net'] = {'model': elastic, 'use_scaled': True, 'use_all_features': False}
    print("  ✓ Complete")
    
    return models


def plot_baseline_empirical_correlations(empirical_model, X_train_all, y_train, output_file='model/baseline_empirical_correlations.png'):
    """
    Plot the baseline empirical model correlations showing feature vs actual values
    with the empirical equation line, similar to Figures 3 and 5 in the literature.
    """
    print("\n" + "=" * 80)
    print("Generating Baseline Empirical Correlation Plots")
    print("=" * 80)
    
    # Extract charges and radii from X_train_all
    type_num = X_train_all['electrolyte_type_numeric'].values
    Z_M = type_num // 10
    Z_X = type_num % 10
    r_M = X_train_all['r_M_angstrom'].values
    r_X = X_train_all['r_X_angstrom'].values
    
    # Compute features
    features_beta0 = []
    features_beta1 = []
    for i in range(len(X_train_all)):
        feat0 = empirical_model.compute_beta0_feature(Z_M[i], Z_X[i], r_M[i], r_X[i])
        feat1 = empirical_model.compute_beta1_feature(Z_M[i], Z_X[i], r_M[i], r_X[i])
        features_beta0.append(feat0)
        features_beta1.append(feat1)
    
    features_beta0 = np.array(features_beta0)
    features_beta1 = np.array(features_beta1)
    
    # Get actual values
    beta0_true = y_train.iloc[:, 0].values  # B_MX_0_original
    beta1_true = y_train.iloc[:, 1].values  # B_MX_1_original
    
    # Classify electrolytes by type for coloring
    electrolyte_types = []
    for zm, zx in zip(Z_M, Z_X):
        if zm == 1 and zx == 1:
            electrolyte_types.append('1-1')
        elif zm == 2 and zx == 1:
            electrolyte_types.append('2-1')
        elif zm == 3 and zx == 1:
            electrolyte_types.append('3-1')
        elif zm == 4 and zx == 1:
            electrolyte_types.append('4-1')
        elif zm == 1 and zx == 2:
            electrolyte_types.append('1-2')
        elif zm == 2 and zx == 2:
            electrolyte_types.append('2-2')
        else:
            electrolyte_types.append('other')
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== Plot 1: Beta(0) correlation =====
    ax = axes[0]
    
    # Plot points by type
    unique_types = sorted(list(set(electrolyte_types)))
    marker_map = {'1-1': 'o', '2-1': '+', '3-1': '^', '4-1': 'x', '1-2': 's', '2-2': 'd', 'other': '*'}
    color_map = {'1-1': 'blue', '2-1': 'red', '3-1': 'green', '4-1': 'orange', 
                 '1-2': 'purple', '2-2': 'brown', 'other': 'gray'}
    
    for etype in unique_types:
        mask = np.array([t == etype for t in electrolyte_types])
        if mask.sum() > 0:
            ax.scatter(features_beta0[mask], beta0_true[mask], 
                      marker=marker_map.get(etype, 'o'), s=100, alpha=0.7,
                      label=f'{etype} electrolytes', color=color_map.get(etype, 'black'))
    
    # Literature empirical line
    x_line = np.linspace(features_beta0.min(), features_beta0.max(), 100)
    y_line = empirical_model.beta0_slope * x_line + empirical_model.beta0_intercept
    ax.plot(x_line, y_line, 'k--', lw=2.5, label='Literature equation')
    
    # Calculate R² with literature equation
    beta0_pred = empirical_model.beta0_slope * features_beta0 + empirical_model.beta0_intercept
    r2 = r2_score(beta0_true, beta0_pred)
    
    ax.set_xlabel(r'$Z_M^{1.62} Z_X^{-1.35} |r_M - 1.5 r_X|^{1.2}$', fontsize=13)
    ax.set_ylabel(r'$B_{MX}^{(0)}$', fontsize=14)
    ax.set_title(f'Beta(0) - Literature Empirical Equation\n' +
                f'y = {empirical_model.beta0_slope:.5f}x + {empirical_model.beta0_intercept:.5f}\n' +
                f'R² = {r2:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 2: Beta(1) correlation =====
    ax = axes[1]
    
    # Transform Beta(1) by dividing by Z_X^0.4
    y_beta1_transformed = beta1_true / (Z_X**0.4)
    
    # Plot points by type
    for etype in unique_types:
        mask = np.array([t == etype for t in electrolyte_types])
        if mask.sum() > 0:
            ax.scatter(features_beta1[mask], y_beta1_transformed[mask], 
                      marker=marker_map.get(etype, 'o'), s=100, alpha=0.7,
                      label=f'{etype} electrolytes', color=color_map.get(etype, 'black'))
    
    # Literature empirical curve (polynomial)
    x_line = np.linspace(features_beta1.min(), features_beta1.max(), 100)
    y_line = (empirical_model.beta1_a2 * x_line**2 + 
              empirical_model.beta1_a1 * x_line + 
              empirical_model.beta1_a0)
    ax.plot(x_line, y_line, 'k--', lw=2.5, label='Literature equation')
    
    # Calculate R²
    y_pred_transformed = (empirical_model.beta1_a2 * features_beta1**2 + 
                         empirical_model.beta1_a1 * features_beta1 + 
                         empirical_model.beta1_a0)
    r2 = r2_score(y_beta1_transformed, y_pred_transformed)
    
    ax.set_xlabel(r'$Z_M^2 Z_X^{0.6} (1 + |r_M - 1.2 r_X|^{0.2})$', fontsize=13)
    ax.set_ylabel(r'$B_{MX}^{(1)} \times Z_X^{-0.4}$', fontsize=14)
    ax.set_title(f'Beta(1) - Literature Empirical Equation\n' +
                f'y = {empirical_model.beta1_a2:.5f}x² + {empirical_model.beta1_a1:.5f}x + {empirical_model.beta1_a0:.5f}\n' +
                f'R² = {r2:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved baseline empirical correlation plots to '{output_file}'")
    plt.close()
    
    return fig


def evaluate_models(models, X_train_all, X_val_all, X_train_ml, X_val_ml,
                   X_train_ml_scaled, X_val_ml_scaled, 
                   y_train, y_val, target_cols, dataset_name="Validation"):
    """Evaluate all models on validation set (used for model selection)."""
    print("\n" + "=" * 80)
    print(f"Model Evaluation ({dataset_name} Set - For Model Selection)")
    print("=" * 80)
    
    results = []
    
    for model_name, model_info in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        model = model_info['model']
        use_scaled = model_info['use_scaled']
        use_all_features = model_info.get('use_all_features', False)
        
        # Baseline uses all features (unscaled), ML models use ML features (scaled)
        if use_all_features:
            X_train_input = X_train_all
            X_val_input = X_val_all
        else:
            X_train_input = X_train_ml_scaled if use_scaled else X_train_ml
            X_val_input = X_val_ml_scaled if use_scaled else X_val_ml
        
        # Predictions
        y_train_pred = model.predict(X_train_input)
        y_val_pred = model.predict(X_val_input)
        
        # Metrics for each target
        model_results = {'model': model_name}
        
        for i, target in enumerate(target_cols):
            # Train metrics
            train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
            train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i]))
            
            # Validation metrics
            val_r2 = r2_score(y_val.iloc[:, i], y_val_pred[:, i])
            val_rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], y_val_pred[:, i]))
            val_mae = mean_absolute_error(y_val.iloc[:, i], y_val_pred[:, i])
            
            print(f"  {target}:")
            print(f"    Train R² = {train_r2:.4f}, RMSE = {train_rmse:.6f}")
            print(f"    Val   R² = {val_r2:.4f}, RMSE = {val_rmse:.6f}, MAE = {val_mae:.6f}")
            
            model_results[f'{target}_val_r2'] = val_r2
            model_results[f'{target}_val_rmse'] = val_rmse
            model_results[f'{target}_val_mae'] = val_mae
        
        # Average performance
        avg_r2 = np.mean([model_results[f'{t}_val_r2'] for t in target_cols])
        avg_rmse = np.mean([model_results[f'{t}_val_rmse'] for t in target_cols])
        model_results['avg_val_r2'] = avg_r2
        model_results['avg_val_rmse'] = avg_rmse
        
        print(f"  Average Val R² = {avg_r2:.4f}, RMSE = {avg_rmse:.6f}")
        
        results.append(model_results)
    
    return pd.DataFrame(results)


def evaluate_final_model(model, X_test_all, X_test_ml, X_test_ml_scaled, y_test, target_cols, 
                        model_name="Final Model", use_scaled=True, use_all_features=False):
    """Evaluate the final selected model on test set (ONLY USED ONCE)."""
    print("\n" + "=" * 80)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 80)
    
    # Select appropriate feature set
    if use_all_features:
        X_test_input = X_test_all
    else:
        X_test_input = X_test_ml_scaled if use_scaled else X_test_ml
    
    y_test_pred = model.predict(X_test_input)
    
    results = {}
    print(f"\n{model_name} - Test Set Performance:")
    print("-" * 60)
    
    for i, target in enumerate(target_cols):
        test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
        test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i]))
        test_mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
        
        print(f"  {target}:")
        print(f"    Test R² = {test_r2:.4f}, RMSE = {test_rmse:.6f}, MAE = {test_mae:.6f}")
        
        results[f'{target}_test_r2'] = test_r2
        results[f'{target}_test_rmse'] = test_rmse
        results[f'{target}_test_mae'] = test_mae
    
    # Average performance
    avg_r2 = np.mean([results[f'{t}_test_r2'] for t in target_cols])
    avg_rmse = np.mean([results[f'{t}_test_rmse'] for t in target_cols])
    results['avg_test_r2'] = avg_r2
    results['avg_test_rmse'] = avg_rmse
    
    print(f"\n  ⭐ Average Test R² = {avg_r2:.4f}, RMSE = {avg_rmse:.6f}")
    print("\n" + "=" * 80)
    
    return results, y_test_pred


def cross_validate_best_model(best_model, X_train_all, X_train_ml, X_train_ml_scaled, 
                              y_train, use_scaled, use_all_features, cv=5):
    """Perform cross-validation on the best model."""
    print("\n" + "=" * 80)
    print("Cross-Validation (Best Model)")
    print("=" * 80)
    
    # Select appropriate feature set
    if use_all_features:
        X_input = X_train_all
    else:
        X_input = X_train_ml_scaled if use_scaled else X_train_ml
    
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_input, y_train, cv=kfold, scoring='r2', n_jobs=1)
    
    print(f"\n{cv}-Fold Cross-Validation R² Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print(f"\nMean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores


def plot_predictions(models, X_val_all, X_val_ml, X_val_ml_scaled, y_val, target_cols, 
                    output_file='model/baseline_ion_enriched_predictions.png', dataset_name="Validation"):
    """Plot predicted vs actual values for each model and target on validation set."""
    print("\n" + "=" * 80)
    print(f"Generating Prediction Plots ({dataset_name} Set)")
    print("=" * 80)
    
    n_models = len(models)
    n_targets = len(target_cols)
    
    fig, axes = plt.subplots(n_models, n_targets, figsize=(7*n_targets, 5*n_models))
    fig.suptitle(f'Baseline + Ion Properties: Predicted vs Actual Pitzer Coefficients ({dataset_name} Set)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_targets == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (model_name, model_info) in enumerate(models.items()):
        model = model_info['model']
        use_scaled = model_info['use_scaled']
        use_all_features = model_info.get('use_all_features', False)
        
        # Select appropriate feature set
        if use_all_features:
            X_input = X_val_all
        else:
            X_input = X_val_ml_scaled if use_scaled else X_val_ml
        
        y_pred = model.predict(X_input)
        
        for j, target in enumerate(target_cols):
            ax = axes[i, j]
            y_true = y_val.iloc[:, j]
            y_p = y_pred[:, j]
            
            # Scatter plot
            ax.scatter(y_true, y_p, alpha=0.6, s=80, edgecolors='k', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_p.min())
            max_val = max(y_true.max(), y_p.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
            
            # Metrics
            r2 = r2_score(y_true, y_p)
            rmse = np.sqrt(mean_squared_error(y_true, y_p))
            
            ax.set_xlabel('Actual', fontsize=11)
            ax.set_ylabel('Predicted', fontsize=11)
            title = f'{model_name} - {target}\nR² = {r2:.4f}, RMSE = {rmse:.4f}'
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to '{output_file}'")
    
    return fig


def plot_best_model_test_predictions(best_model, best_model_name, baseline_model, X_test_all, X_test_ml, X_test_ml_scaled,
                                     y_test, target_cols, use_scaled=True, use_all_features=False,
                                     output_file='model/best_model_test_predictions.png'):
    """Plot predicted vs actual values for the best model and baseline on test set."""
    print("\n" + "=" * 80)
    print("Generating Best Model and Baseline Test Set Prediction Plot")
    print("=" * 80)
    
    n_targets = len(target_cols)
    
    # Create subplots: 2 rows (best model, baseline) x n_targets columns
    fig, axes = plt.subplots(2, n_targets, figsize=(7*n_targets, 10))
    fig.suptitle('Test Set Predictions: Best Model vs Baseline Empirical', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if n_targets == 1:
        axes = axes.reshape(-1, 1)
    
    # Select appropriate feature set for best model
    if use_all_features:
        X_input_best = X_test_all
    else:
        X_input_best = X_test_ml_scaled if use_scaled else X_test_ml
    
    # Baseline uses all features (unscaled)
    X_input_baseline = X_test_all
    
    y_pred_best = best_model.predict(X_input_best)
    y_pred_baseline = baseline_model.predict(X_input_baseline)
    
    for j, target in enumerate(target_cols):
        y_true = y_test.iloc[:, j]
        
        # Top row: Best model
        ax = axes[0, j]
        y_p_best = y_pred_best[:, j]
        
        # Scatter plot
        ax.scatter(y_true, y_p_best, alpha=0.7, s=100, edgecolors='k', linewidth=0.5, color='steelblue', label=best_model_name)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_p_best.min(), y_pred_baseline[:, j].min())
        max_val = max(y_true.max(), y_p_best.max(), y_pred_baseline[:, j].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')
        
        # Metrics
        r2 = r2_score(y_true, y_p_best)
        rmse = np.sqrt(mean_squared_error(y_true, y_p_best))
        mae = mean_absolute_error(y_true, y_p_best)
        
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        title = f'{best_model_name} - {target}\nR² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Baseline empirical
        ax = axes[1, j]
        y_p_baseline = y_pred_baseline[:, j]
        
        # Scatter plot
        ax.scatter(y_true, y_p_baseline, alpha=0.7, s=100, edgecolors='k', linewidth=0.5, color='orange', label='Baseline Empirical')
        
        # Perfect prediction line
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')
        
        # Metrics
        r2_baseline = r2_score(y_true, y_p_baseline)
        rmse_baseline = np.sqrt(mean_squared_error(y_true, y_p_baseline))
        mae_baseline = mean_absolute_error(y_true, y_p_baseline)
        
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        title = f'Baseline Empirical - {target}\nR² = {r2_baseline:.4f}, RMSE = {rmse_baseline:.4f}, MAE = {mae_baseline:.4f}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to '{output_file}'")
    
    return fig


def analyze_feature_importance(model, feature_cols, model_name, target_cols, top_n=None):
    """Analyze and display feature importance."""
    print("\n" + "=" * 80)
    print(f"Feature Importance Analysis ({model_name})")
    print("=" * 80)
    
    importances = None
    
    # Extract feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        print(f"\nUsing built-in feature importances")
        
    elif hasattr(model, 'estimators_'):
        print(f"\nAveraging feature importances across {len(target_cols)} target models")
        importances_list = []
        for estimator in model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances_list.append(estimator.feature_importances_)
        if importances_list:
            importances = np.mean(importances_list, axis=0)
            
    elif hasattr(model, 'coef_'):
        print(f"\nUsing absolute coefficient values")
        if model.coef_.ndim == 1:
            importances = np.abs(model.coef_)
        else:
            importances = np.mean(np.abs(model.coef_), axis=0)
    
    if importances is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Show all features if top_n is None, otherwise show top_n
        if top_n is None:
            top_n = len(feature_importance)
            print(f"\nALL FEATURES RANKED BY IMPORTANCE ({len(feature_importance)} features):")
        else:
            print(f"\nTOP {top_n} MOST IMPORTANT FEATURES:")
        print("-" * 80)
        for rank, (idx, row) in enumerate(feature_importance.head(top_n).iterrows(), 1):
            print(f"  {rank:2d}. {row['feature']:45s} : {row['importance']:.6f}")
        
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE BY CATEGORY")
        print("=" * 80)
        
        # Categorize features
        baseline_imp = feature_importance[feature_importance['feature'].str.contains('r_|std_dev|max_molality')]['importance'].sum()
        mol_imp = feature_importance[feature_importance['feature'].str.startswith('molecule_')]['importance'].sum()
        cat_imp = feature_importance[feature_importance['feature'].str.startswith('cation_')]['importance'].sum()
        an_imp = feature_importance[feature_importance['feature'].str.startswith('anion_')]['importance'].sum()
        type_imp = feature_importance[feature_importance['feature'].str.contains('type_numeric')]['importance'].sum()
        
        total_imp = baseline_imp + mol_imp + cat_imp + an_imp + type_imp
        if total_imp > 0:
            print(f"  Baseline properties : {baseline_imp:.4f} ({100*baseline_imp/total_imp:.1f}%)")
            print(f"  Type indicators     : {type_imp:.4f} ({100*type_imp/total_imp:.1f}%)")
            print(f"  Molecular features  : {mol_imp:.4f} ({100*mol_imp/total_imp:.1f}%)")
            print(f"  Cation features     : {cat_imp:.4f} ({100*cat_imp/total_imp:.1f}%)")
            print(f"  Anion features      : {an_imp:.4f} ({100*an_imp/total_imp:.1f}%)")
        
        return feature_importance
    else:
        print(f"\n⚠️ Feature importance not available for this model type")
        return None


def save_model(model, scaler, filepath='model/best_baseline_ion_model.pkl'):
    """Save the trained model and scaler."""
    import pickle
    
    print("\n" + "=" * 80)
    print("Saving Model")
    print("=" * 80)
    
    model_data = {
        'model': model,
        'scaler': scaler
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model and scaler saved to '{filepath}'")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("PITZER COEFFICIENTS PREDICTION")
    print("Baseline Data Enriched with Ion Properties")
    print("=" * 80)
    
    # Set up paths
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'model')
    
    # 1. Load data
    X_all, X_ml, y, molecules, all_feature_cols, ml_feature_cols, target_cols = load_data(
        os.path.join(data_dir, 'baseline_with_ion_properties.csv')
    )
    
    # 2. Explore data
    explore_data(X_all, X_ml, y, all_feature_cols, ml_feature_cols, target_cols)
    
    # 3. Split into train/validation/test and scale
    (X_train_all, X_val_all, X_test_all,
     X_train_ml, X_val_ml, X_test_ml,
     X_train_ml_scaled, X_val_ml_scaled, X_test_ml_scaled,
     y_train, y_val, y_test, scaler_ml) = \
        split_and_scale_data(X_all, X_ml, y, train_size=0.75, val_size=0.10, test_size=0.15, random_state=42)
    
    # 4. Train models
    models = train_models(X_train_all, X_train_ml, X_train_ml_scaled, y_train, target_cols)
    
    # 4.5. Plot baseline empirical correlations
    if 'Baseline Empirical' in models:
        empirical_model = models['Baseline Empirical']['model']
        plot_output = os.path.join(model_dir, 'baseline_empirical_correlations.png')
        plot_baseline_empirical_correlations(empirical_model, X_train_all, y_train, plot_output)
    
    # 5. Evaluate models on VALIDATION set (used for model selection)
    results_df = evaluate_models(models, X_train_all, X_val_all, X_train_ml, X_val_ml,
                                 X_train_ml_scaled, X_val_ml_scaled,
                                 y_train, y_val, target_cols, dataset_name="Validation")
    
    # 6. Compare models based on VALIDATION performance
    print("\n" + "=" * 80)
    print("Model Comparison (Based on Validation Set)")
    print("=" * 80)
    print("\nAverage Validation R² by Model:")
    comparison = results_df[['model', 'avg_val_r2', 'avg_val_rmse']].sort_values('avg_val_r2', ascending=False)
    print(comparison.to_string(index=False))
    
    best_model_name = comparison.iloc[0]['model']
    best_model_info = models[best_model_name]
    best_model = best_model_info['model']
    best_use_scaled = best_model_info['use_scaled']
    best_use_all_features = best_model_info.get('use_all_features', False)
    
    print(f"\n🏆 Best Model (Selected on Validation): {best_model_name}")
    print(f"   Validation R²: {comparison.iloc[0]['avg_val_r2']:.4f}")
    
    # 7. Cross-validation on training set only
    if best_model_name != 'Baseline Empirical':
        cv_scores = cross_validate_best_model(
            best_model, X_train_all, X_train_ml, X_train_ml_scaled, 
            y_train, best_use_scaled, best_use_all_features
        )
    
    # 8. FINAL EVALUATION: Evaluate selected model ONCE on test set
    final_test_results, y_test_pred = evaluate_final_model(
        best_model, X_test_all, X_test_ml, X_test_ml_scaled, y_test, target_cols, 
        best_model_name, best_use_scaled, best_use_all_features
    )
    
    # 8.5. Evaluate baseline empirical model on test set
    baseline_model = models['Baseline Empirical']['model']
    baseline_test_results, y_test_pred_baseline = evaluate_final_model(
        baseline_model, X_test_all, X_test_ml, X_test_ml_scaled, y_test, target_cols,
        'Baseline Empirical', use_scaled=False, use_all_features=True
    )
    
    # 9. Plot predictions on validation set (all models)
    plot_output = os.path.join(model_dir, 'baseline_ion_enriched_predictions.png')
    plot_predictions(models, X_val_all, X_val_ml, X_val_ml_scaled, y_val, target_cols, 
                    plot_output, dataset_name="Validation")
    
    # 9.5. Plot best model and baseline predictions on test set
    best_model_plot_output = os.path.join(model_dir, 'best_model_test_predictions.png')
    plot_best_model_test_predictions(
        best_model, best_model_name, baseline_model, X_test_all, X_test_ml, X_test_ml_scaled, y_test, 
        target_cols, best_use_scaled, best_use_all_features, best_model_plot_output
    )
    
    # 10. Feature importance
    if best_model_name == 'Baseline Empirical':
        # Baseline Empirical won, but also show best ML model's feature importance
        print("\n" + "=" * 80)
        print("Note: Baseline Empirical model uses fixed equations")
        print("=" * 80)
        print("\nThe empirical model uses:")
        print("  - Electrolyte type (charges)")
        print("  - Ionic radii (r_M_angstrom, r_X_angstrom)")
        print("\nNo other features are used in the literature equations.")
        
        # Find and analyze best ML model
        ml_comparison = comparison[comparison['model'] != 'Baseline Empirical']
        if len(ml_comparison) > 0:
            best_ml_model_name = ml_comparison.iloc[0]['model']
            best_ml_model = models[best_ml_model_name]['model']
            
            print("\n" + "=" * 80)
            print(f"Best ML Model Feature Importance ({best_ml_model_name})")
            print(f"Validation R²: {ml_comparison.iloc[0]['avg_val_r2']:.4f}")
            print("=" * 80)
            
            feature_importance = analyze_feature_importance(
                best_ml_model, ml_feature_cols, best_ml_model_name, target_cols
            )
    else:
        # Best model is already an ML model
        feature_importance = analyze_feature_importance(
            best_model, ml_feature_cols, best_model_name, target_cols
        )
    
    # 11. Save best model
    save_model(best_model, scaler_ml, os.path.join(model_dir, 'best_baseline_ion_model.pkl'))
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nDataset: Baseline + Ion Properties")
    print(f"  Samples: {len(X_ml)} ({len(X_train_ml)} train, {len(X_val_ml)} validation, {len(X_test_ml)} test)")
    print(f"  All features (baseline): {len(all_feature_cols)}")
    print(f"  ML features (subset): {len(ml_feature_cols)}")
    print(f"  Targets: {len(target_cols)} ({', '.join(target_cols)})")
    
    print(f"\nBest Model: {best_model_name}")
    print(f"  Selected based on Validation R²: {comparison.iloc[0]['avg_val_r2']:.4f}")
    print(f"  Final Test R² (unbiased): {final_test_results['avg_test_r2']:.4f}")
    print(f"  Final Test RMSE: {final_test_results['avg_test_rmse']:.6f}")
    
    print(f"\nBaseline Empirical Model (Test Set):")
    print(f"  Test R²: {baseline_test_results['avg_test_r2']:.4f}")
    print(f"  Test RMSE: {baseline_test_results['avg_test_rmse']:.6f}")
    
    print("\nGenerated Files:")
    print("  - best_baseline_ion_model.pkl (trained model + scaler)")
    print("  - baseline_ion_enriched_predictions.png (all models on validation set)")
    print("  - best_model_test_predictions.png (best model vs baseline on test set)")
    print("  - baseline_empirical_correlations.png (baseline empirical model correlations)")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("\n1. Baseline Empirical Model:")
    print("   - Uses literature equations (no training)")
    print("   - Only requires charges and radii")
    print("   - Good baseline for comparison")
    
    print("\n2. ML Models:")
    print("   - Leverage additional ion properties")
    print("   - Can capture non-linear relationships")
    print("   - May generalize better to new electrolytes")
    
    empirical_val_r2 = results_df[results_df['model'] == 'Baseline Empirical']['avg_val_r2'].values[0]
    best_ml_val_r2 = comparison[comparison['model'] != 'Baseline Empirical'].iloc[0]['avg_val_r2']
    improvement = ((best_ml_val_r2 - empirical_val_r2) / abs(empirical_val_r2)) * 100 if empirical_val_r2 != 0 else 0
    
    print(f"\n3. Performance Comparison (Validation Set):")
    print(f"   - Empirical baseline: R² = {empirical_val_r2:.4f}")
    print(f"   - Best ML model: R² = {best_ml_val_r2:.4f}")
    print(f"   - Improvement: {improvement:+.1f}%")
    
    print(f"\n4. Final Test Set Performance:")
    print(f"   - Best model test R²: {final_test_results['avg_test_r2']:.4f}")
    print(f"   - Best model test RMSE: {final_test_results['avg_test_rmse']:.6f}")
    
    print("\nData Split:")
    print(f"  - Training: {len(X_train_ml)} samples (75%)")
    print(f"  - Validation: {len(X_val_ml)} samples (10%) - used for model selection")
    print(f"  - Test: {len(X_test_ml)} samples (15%) - used ONCE for final evaluation")


if __name__ == "__main__":
    main()

