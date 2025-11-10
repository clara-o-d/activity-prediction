"""
Baseline model for Pitzer parameter prediction based on empirical correlations.

This implements the correlations from Figures 3 and 5 that relate Beta(0) and Beta(1) 
to ionic charges (Z_M, Z_X) and ionic radii (r_M, r_X).

Reference equations:
- Beta(0) = a0 * Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2 + b0
- Beta(1) = Z_X^0.4 * (a1 * X^2 + b1 * X + c1)
  where X = Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)^2
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle


class PitzerBaselineModel:
    """
    Baseline model for predicting Pitzer parameters from ionic properties.
    Uses empirical correlations based on charge and ionic radius.
    """
    
    def __init__(self):
        # Coefficients for Beta(0) prediction (will be fit)
        self.beta0_model = None
        
        # Coefficients for Beta(1) prediction (will be fit) 
        self.beta1_model = None
        
        # Store fitted coefficients
        self.beta0_coef_ = None
        self.beta0_intercept_ = None
        self.beta1_coef_ = None
        
    def compute_beta0_feature(self, Z_M, Z_X, r_M, r_X):
        """
        Compute the feature for Beta(0) prediction.
        
        Feature = Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2
        
        Parameters:
        -----------
        Z_M : array-like
            Cation charge (absolute value)
        Z_X : array-like
            Anion charge (absolute value)
        r_M : array-like
            Cation ionic radius (pm)
        r_X : array-like
            Anion ionic radius (pm)
        
        Returns:
        --------
        feature : array-like
            Computed feature for Beta(0) prediction
        """
        Z_M = np.abs(Z_M)
        Z_X = np.abs(Z_X)
        
        feature = (Z_M**1.62) * (Z_X**(-1.35)) * (np.abs(r_M - 1.5*r_X)**1.2)
        return feature
    
    def compute_beta1_feature(self, Z_M, Z_X, r_M, r_X):
        """
        Compute the feature for Beta(1) prediction.
        
        Feature = Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)^2
        
        Parameters:
        -----------
        Z_M : array-like
            Cation charge (absolute value)
        Z_X : array-like
            Anion charge (absolute value)
        r_M : array-like
            Cation ionic radius (pm)
        r_X : array-like
            Anion ionic radius (pm)
        
        Returns:
        --------
        feature : array-like
            Computed feature for Beta(1) prediction
        """
        Z_M = np.abs(Z_M)
        Z_X = np.abs(Z_X)
        
        feature = (Z_M**2) * (Z_X**0.6) * ((1 + np.abs(r_M - 1.2*r_X)**0.2)**2)
        return feature
    
    def fit(self, X, y):
        """
        Fit the baseline model.
        
        Parameters:
        -----------
        X : DataFrame
            Must contain columns: 'Cation_Charge (formal)', 'Anion_Charge',
            'Cation_Ionic radius (pm) (molecular radius if polyatomic)',
            'Anion_Ionic radius (pm) (molecular radius if polyatomic).1'
        y : DataFrame
            Must contain columns: 'Beta(0)', 'Beta(1)'
        """
        # Extract ionic properties
        Z_M = X['Cation_Charge (formal)'].values
        Z_X = X['Anion_Charge'].values
        r_M = X['Cation_Ionic radius (pm) (molecular radius if polyatomic)'].values
        r_X = X['Anion_Ionic radius (pm) (molecular radius if polyatomic).1'].values
        
        # Get target values
        beta0_true = y['Beta(0)'].values
        beta1_true = y['Beta(1)'].values
        
        # Fit Beta(0) model: linear regression
        # Beta(0) = a * feature + b
        feature_beta0 = self.compute_beta0_feature(Z_M, Z_X, r_M, r_X).reshape(-1, 1)
        self.beta0_model = LinearRegression()
        self.beta0_model.fit(feature_beta0, beta0_true)
        self.beta0_coef_ = self.beta0_model.coef_[0]
        self.beta0_intercept_ = self.beta0_model.intercept_
        
        # Fit Beta(1) model: polynomial regression
        # Beta(1) * Z_X^-0.4 = a * X^2 + b * X + c
        # So we need to transform the target: y_transformed = Beta(1) * Z_X^-0.4
        Z_X_abs = np.abs(Z_X)
        y_beta1_transformed = beta1_true * (Z_X_abs**(-0.4))
        
        feature_beta1 = self.compute_beta1_feature(Z_M, Z_X, r_M, r_X)
        
        # Create polynomial features (X and X^2)
        X_poly = np.column_stack([feature_beta1, feature_beta1**2])
        
        self.beta1_model = LinearRegression()
        self.beta1_model.fit(X_poly, y_beta1_transformed)
        self.beta1_coef_ = self.beta1_model.coef_
        self.beta1_intercept_ = self.beta1_model.intercept_
        
        return self
    
    def predict(self, X):
        """
        Predict Beta(0) and Beta(1) values.
        
        Parameters:
        -----------
        X : DataFrame
            Must contain columns: 'Cation_Charge (formal)', 'Anion_Charge',
            'Cation_Ionic radius (pm) (molecular radius if polyatomic)',
            'Anion_Ionic radius (pm) (molecular radius if polyatomic).1'
        
        Returns:
        --------
        predictions : DataFrame
            Contains columns 'Beta(0)' and 'Beta(1)' with predicted values
        """
        # Extract ionic properties
        Z_M = X['Cation_Charge (formal)'].values
        Z_X = X['Anion_Charge'].values
        r_M = X['Cation_Ionic radius (pm) (molecular radius if polyatomic)'].values
        r_X = X['Anion_Ionic radius (pm) (molecular radius if polyatomic).1'].values
        
        # Predict Beta(0)
        feature_beta0 = self.compute_beta0_feature(Z_M, Z_X, r_M, r_X).reshape(-1, 1)
        beta0_pred = self.beta0_model.predict(feature_beta0)
        
        # Predict Beta(1)
        feature_beta1 = self.compute_beta1_feature(Z_M, Z_X, r_M, r_X)
        X_poly = np.column_stack([feature_beta1, feature_beta1**2])
        y_beta1_transformed_pred = self.beta1_model.predict(X_poly)
        
        # Transform back: Beta(1) = y_transformed * Z_X^0.4
        Z_X_abs = np.abs(Z_X)
        beta1_pred = y_beta1_transformed_pred * (Z_X_abs**0.4)
        
        return pd.DataFrame({
            'Beta(0)': beta0_pred,
            'Beta(1)': beta1_pred
        })
    
    def score(self, X, y):
        """
        Compute R² scores for both Beta(0) and Beta(1) predictions.
        
        Returns:
        --------
        scores : dict
            Dictionary with R² scores for Beta(0) and Beta(1)
        """
        predictions = self.predict(X)
        
        r2_beta0 = r2_score(y['Beta(0)'], predictions['Beta(0)'])
        r2_beta1 = r2_score(y['Beta(1)'], predictions['Beta(1)'])
        
        return {
            'R2_Beta(0)': r2_beta0,
            'R2_Beta(1)': r2_beta1,
            'R2_mean': (r2_beta0 + r2_beta1) / 2
        }


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate the baseline model and print metrics.
    """
    predictions = model.predict(X)
    
    print(f"\n{'='*60}")
    print(f"Evaluation on {dataset_name}")
    print(f"{'='*60}")
    
    for target in ['Beta(0)', 'Beta(1)']:
        y_true = y[target]
        y_pred = predictions[target]
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"\n{target}:")
        print(f"  R² Score:  {r2:.4f}")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  MAE:       {mae:.4f}")
    
    # Overall R²
    all_true = np.concatenate([y['Beta(0)'], y['Beta(1)']])
    all_pred = np.concatenate([predictions['Beta(0)'], predictions['Beta(1)']])
    overall_r2 = r2_score(all_true, all_pred)
    print(f"\n{'Overall R² (both targets):':<30} {overall_r2:.4f}")
    
    return predictions


def plot_predictions(y_true, y_pred, model, save_path='model/baseline/baseline_predictions.png'):
    """
    Create prediction plots for Beta(0) and Beta(1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, target in enumerate(['Beta(0)', 'Beta(1)']):
        ax = axes[idx]
        
        true_vals = y_true[target]
        pred_vals = y_pred[target]
        
        # Scatter plot
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate metrics
        r2 = r2_score(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        
        ax.set_xlabel(f'True {target}', fontsize=12)
        ax.set_ylabel(f'Predicted {target}', fontsize=12)
        ax.set_title(f'{target} Predictions\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPrediction plots saved to: {save_path}")
    plt.close()


def plot_empirical_correlations(X, y, model, save_path='model/baseline/empirical_correlations.png'):
    """
    Create plots showing the empirical correlations (matching Figures 3 and 5 from literature).
    
    Figure 3-style: Beta(0) vs feature
    Figure 5-style: Beta(1) vs feature (after transformation)
    """
    # Extract ionic properties
    Z_M = np.abs(X['Cation_Charge (formal)'].values)
    Z_X = np.abs(X['Anion_Charge'].values)
    r_M = X['Cation_Ionic radius (pm) (molecular radius if polyatomic)'].values
    r_X = X['Anion_Ionic radius (pm) (molecular radius if polyatomic).1'].values
    
    beta0_true = y['Beta(0)'].values
    beta1_true = y['Beta(1)'].values
    
    # Compute features
    feature_beta0 = model.compute_beta0_feature(Z_M, Z_X, r_M, r_X)
    feature_beta1 = model.compute_beta1_feature(Z_M, Z_X, r_M, r_X)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== Plot 1: Beta(0) correlation (Figure 3 style) =====
    ax = axes[0]
    
    # Classify electrolytes by type (based on charges)
    electrolyte_types = []
    markers = []
    for zm, zx in zip(Z_M, Z_X):
        if zm == 1 and zx == 1:
            electrolyte_types.append('1-1')
            markers.append('o')
        elif zm == 2 and zx == 1:
            electrolyte_types.append('2-1')
            markers.append('+')
        elif zm == 3 and zx == 1:
            electrolyte_types.append('3-1')
            markers.append('^')
        elif zm == 4 and zx == 1:
            electrolyte_types.append('4-1')
            markers.append('x')
        elif zm == 2 and zx == 2:
            electrolyte_types.append('2-2')
            markers.append('s')
        else:
            electrolyte_types.append('other')
            markers.append('d')
    
    # Plot points by type
    unique_types = list(set(electrolyte_types))
    marker_map = {'1-1': 'o', '2-1': '+', '3-1': '^', '4-1': 'x', '2-2': 's', 'other': 'd'}
    
    for etype in unique_types:
        mask = np.array([t == etype for t in electrolyte_types])
        ax.scatter(feature_beta0[mask], beta0_true[mask], 
                  marker=marker_map[etype], s=100, alpha=0.7,
                  label=f'{etype} electrolytes')
    
    # Linear regression line
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(feature_beta0.reshape(-1, 1), beta0_true)
    x_line = np.linspace(feature_beta0.min(), feature_beta0.max(), 100)
    y_line = lr.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, 'k--', lw=2, label='Linear regression')
    
    # Calculate R² and equation
    r2 = r2_score(beta0_true, lr.predict(feature_beta0.reshape(-1, 1)))
    slope = lr.coef_[0]
    intercept = lr.intercept_
    
    ax.set_xlabel(r'$Z_M^{1.62} Z_X^{-1.35} |r_M - 1.5 r_X|^{1.2}$', fontsize=13)
    ax.set_ylabel(r'$B_{MX}^{(0)}$', fontsize=14)
    ax.set_title(f'Relationship between $B_{{MX}}^{{(0)}}$ and ionic properties\n' +
                f'y = {slope:.5f}x + {intercept:.5f}\n' +
                f'R² = {r2:.5f}', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 2: Beta(1) correlation (Figure 5 style) =====
    ax = axes[1]
    
    # Transform Beta(1) by dividing by Z_X^0.4
    y_beta1_transformed = beta1_true * (Z_X**(-0.4))
    
    # Plot points by type
    for etype in unique_types:
        mask = np.array([t == etype for t in electrolyte_types])
        ax.scatter(feature_beta1[mask], y_beta1_transformed[mask], 
                  marker=marker_map[etype], s=100, alpha=0.7,
                  label=f'{etype} electrolytes')
    
    # Polynomial regression line (quadratic)
    X_poly = np.column_stack([feature_beta1, feature_beta1**2])
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_beta1_transformed)
    
    x_line = np.linspace(feature_beta1.min(), feature_beta1.max(), 100)
    X_line_poly = np.column_stack([x_line, x_line**2])
    y_line = poly_model.predict(X_line_poly)
    ax.plot(x_line, y_line, 'k--', lw=2, label='Polynomial regression')
    
    # Calculate R²
    r2 = r2_score(y_beta1_transformed, poly_model.predict(X_poly))
    a2 = poly_model.coef_[1]  # coefficient of X^2
    a1 = poly_model.coef_[0]  # coefficient of X
    a0 = poly_model.intercept_
    
    ax.set_xlabel(r'$Z_M^2 Z_X^{0.6} (1 + |r_M - 1.2 r_X|^{0.2})^2$', fontsize=13)
    ax.set_ylabel(r'$B_{MX}^{(1)} \times Z_X^{-0.4}$', fontsize=14)
    ax.set_title(f'Relationship between $B_{{MX}}^{{(1)}}$ and ionic properties\n' +
                f'y = {a2:.5f}x² + {a1:.5f}x + {a0:.5f}\n' +
                f'R² = {r2:.5f}', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Empirical correlation plots saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to train and evaluate the baseline model.
    """
    print("="*60)
    print("Pitzer Parameter Baseline Model")
    print("Based on empirical correlations of charge and ionic radius")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = pd.read_csv('/Users/clara/activity-prediction/data/clean_activity_dataset.csv')
    
    # Remove rows with missing values in key columns
    required_cols = [
        'Cation_Charge (formal)',
        'Anion_Charge',
        'Cation_Ionic radius (pm) (molecular radius if polyatomic)',
        'Anion_Ionic radius (pm) (molecular radius if polyatomic).1',
        'Beta(0)',
        'Beta(1)'
    ]
    
    # Convert to numeric, coercing errors to NaN
    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data_clean = data[required_cols].dropna()
    print(f"Loaded {len(data_clean)} samples with complete data")
    
    # Prepare features and targets
    feature_cols = [
        'Cation_Charge (formal)',
        'Anion_Charge',
        'Cation_Ionic radius (pm) (molecular radius if polyatomic)',
        'Anion_Ionic radius (pm) (molecular radius if polyatomic).1'
    ]
    target_cols = ['Beta(0)', 'Beta(1)']
    
    X = data_clean[feature_cols]
    y = data_clean[target_cols]
    
    # Train model
    print("\nTraining baseline model...")
    model = PitzerBaselineModel()
    model.fit(X, y)
    
    print("\nModel coefficients:")
    print(f"  Beta(0): {model.beta0_coef_:.5f} * feature + {model.beta0_intercept_:.5f}")
    print(f"  Beta(1): [{model.beta1_coef_[0]:.5f}, {model.beta1_coef_[1]:.5f}] * [X, X²] + {model.beta1_intercept_:.5f}")
    
    # Evaluate model
    predictions = evaluate_model(model, X, y, "Full Dataset")
    
    # Create plots
    plot_predictions(y, predictions, model)
    
    # Create empirical correlation plots (Figures 3 & 5 style)
    print("\nCreating empirical correlation plots...")
    plot_empirical_correlations(X, y, model)
    
    # Save model
    model_path = '/Users/clara/activity-prediction/model/baseline/baseline_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Print comparison to reference values from the figures
    print("\n" + "="*60)
    print("Comparison to reference values from literature:")
    print("="*60)
    print("\nBeta(0) coefficients:")
    print(f"  Reference: 0.04850 * feature + 0.03898")
    print(f"  Our fit:   {model.beta0_coef_:.5f} * feature + {model.beta0_intercept_:.5f}")
    
    print("\nNote: Beta(1) uses polynomial regression, see coefficients above")


if __name__ == "__main__":
    main()

