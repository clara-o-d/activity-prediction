"""
Baseline model using exact empirical equations from literature.

This version uses the exact equations from Figures 3 and 5 without refitting:
- Beta(0) = 0.04850 * Z_M^1.62 * Z_X^-1.35 * |r_M - 1.5*r_X|^1.2 + 0.03898
- Beta(1) = 0.00738 * Z_X^-0.4 * X^2 + 0.16800 * Z_X^-0.4 * X - 0.09320 * Z_X^-0.4
  where X = Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)^2

No fitting is performed - these are the literature values.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle


class PitzerEmpiricalModel:
    """
    Baseline model using exact empirical equations from literature.
    No training required - uses fixed coefficients from published correlations.
    """
    
    def __init__(self):
        # Beta(0) coefficients from Figure 3
        self.beta0_slope = 0.04850
        self.beta0_intercept = 0.03898
        
        # Beta(1) coefficients from Figure 5 (polynomial equation 18)
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
        
        Feature = Z_M^2 * Z_X^0.6 * (1 + |r_M - 1.2*r_X|^0.2)^2
        """
        Z_M = np.abs(Z_M)
        Z_X = np.abs(Z_X)
        
        feature = (Z_M**2) * (Z_X**0.6) * ((1 + np.abs(r_M - 1.2*r_X)**0.2)**2)
        return feature
    
    def fit(self, X, y):
        """
        Dummy fit method for compatibility with scikit-learn interface.
        Does not actually train - uses fixed literature coefficients.
        """
        print("Note: This model uses fixed empirical coefficients from literature.")
        print("No training is performed.")
        return self
    
    def predict(self, X):
        """
        Predict Beta(0) and Beta(1) values using empirical equations.
        
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
        
        # Convert ionic radii from picometers (pm) to Angstroms (Å)
        # Literature equations use Angstroms: 1 Å = 100 pm
        r_M = r_M / 100.0  # Convert pm to Å
        r_X = r_X / 100.0  # Convert pm to Å
        
        # Predict Beta(0) using empirical equation from Figure 3
        feature_beta0 = self.compute_beta0_feature(Z_M, Z_X, r_M, r_X)
        beta0_pred = self.beta0_slope * feature_beta0 + self.beta0_intercept
        
        # Predict Beta(1) using empirical equation from Figure 5 (Equation 18)
        feature_beta1 = self.compute_beta1_feature(Z_M, Z_X, r_M, r_X)
        
        # y_transformed = a2*X^2 + a1*X + a0
        y_beta1_transformed = (self.beta1_a2 * feature_beta1**2 + 
                              self.beta1_a1 * feature_beta1 + 
                              self.beta1_a0)
        
        # Transform back: Beta(1) = y_transformed * Z_X^0.4
        Z_X_abs = np.abs(Z_X)
        beta1_pred = y_beta1_transformed * (Z_X_abs**0.4)
        
        return pd.DataFrame({
            'Beta(0)': beta0_pred,
            'Beta(1)': beta1_pred
        })
    
    def score(self, X, y):
        """
        Compute R² scores for both Beta(0) and Beta(1) predictions.
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
    Evaluate the empirical model and print metrics.
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


def plot_predictions(y_true, y_pred, save_path='model/baseline/empirical_predictions.png'):
    """
    Create prediction plots for Beta(0) and Beta(1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, target in enumerate(['Beta(0)', 'Beta(1)']):
        ax = axes[idx]
        
        true_vals = y_true[target]
        pred_vals = y_pred[target]
        
        # Scatter plot
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=50, color='purple')
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate metrics
        r2 = r2_score(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        
        ax.set_xlabel(f'True {target}', fontsize=12)
        ax.set_ylabel(f'Predicted {target}', fontsize=12)
        ax.set_title(f'{target} Predictions (Empirical)\nR² = {r2:.4f}, RMSE = {rmse:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPrediction plots saved to: {save_path}")
    plt.close()


def plot_empirical_correlations(X, y, model, save_path='model/baseline/empirical_literature_fit.png'):
    """
    Create plots showing the literature empirical correlations applied to our data.
    """
    # Extract ionic properties
    Z_M = np.abs(X['Cation_Charge (formal)'].values)
    Z_X = np.abs(X['Anion_Charge'].values)
    r_M = X['Cation_Ionic radius (pm) (molecular radius if polyatomic)'].values
    r_X = X['Anion_Ionic radius (pm) (molecular radius if polyatomic).1'].values
    
    # Convert ionic radii from pm to Angstroms (Å) for literature equations
    r_M = r_M / 100.0  # Convert pm to Å
    r_X = r_X / 100.0  # Convert pm to Å
    
    beta0_true = y['Beta(0)'].values
    beta1_true = y['Beta(1)'].values
    
    # Compute features
    feature_beta0 = model.compute_beta0_feature(Z_M, Z_X, r_M, r_X)
    feature_beta1 = model.compute_beta1_feature(Z_M, Z_X, r_M, r_X)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== Plot 1: Beta(0) correlation =====
    ax = axes[0]
    
    # Classify electrolytes by type
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
        elif zm == 2 and zx == 2:
            electrolyte_types.append('2-2')
        else:
            electrolyte_types.append('other')
    
    # Plot points by type
    unique_types = list(set(electrolyte_types))
    marker_map = {'1-1': 'o', '2-1': '+', '3-1': '^', '4-1': 'x', '2-2': 's', 'other': 'd'}
    
    for etype in unique_types:
        mask = np.array([t == etype for t in electrolyte_types])
        ax.scatter(feature_beta0[mask], beta0_true[mask], 
                  marker=marker_map[etype], s=100, alpha=0.7,
                  label=f'{etype} electrolytes')
    
    # Literature empirical line (no fitting!)
    x_line = np.linspace(feature_beta0.min(), feature_beta0.max(), 100)
    y_line = model.beta0_slope * x_line + model.beta0_intercept
    ax.plot(x_line, y_line, 'k--', lw=2.5, label='Literature equation')
    
    # Calculate R² with literature equation
    beta0_pred = model.beta0_slope * feature_beta0 + model.beta0_intercept
    r2 = r2_score(beta0_true, beta0_pred)
    
    ax.set_xlabel(r'$Z_M^{1.62} Z_X^{-1.35} |r_M - 1.5 r_X|^{1.2}$', fontsize=13)
    ax.set_ylabel(r'$B_{MX}^{(0)}$', fontsize=14)
    ax.set_title(f'Beta(0) - Literature Empirical Equation\n' +
                f'y = {model.beta0_slope:.5f}x + {model.beta0_intercept:.5f}\n' +
                f'R² = {r2:.5f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # ===== Plot 2: Beta(1) correlation =====
    ax = axes[1]
    
    # Transform Beta(1) by dividing by Z_X^0.4
    y_beta1_transformed = beta1_true * (Z_X**(-0.4))
    
    # Plot points by type
    for etype in unique_types:
        mask = np.array([t == etype for t in electrolyte_types])
        ax.scatter(feature_beta1[mask], y_beta1_transformed[mask], 
                  marker=marker_map[etype], s=100, alpha=0.7,
                  label=f'{etype} electrolytes')
    
    # Literature empirical curve (polynomial)
    x_line = np.linspace(feature_beta1.min(), feature_beta1.max(), 100)
    y_line = model.beta1_a2 * x_line**2 + model.beta1_a1 * x_line + model.beta1_a0
    ax.plot(x_line, y_line, 'k--', lw=2.5, label='Literature equation')
    
    # Calculate R²
    y_pred_transformed = model.beta1_a2 * feature_beta1**2 + model.beta1_a1 * feature_beta1 + model.beta1_a0
    r2 = r2_score(y_beta1_transformed, y_pred_transformed)
    
    ax.set_xlabel(r'$Z_M^2 Z_X^{0.6} (1 + |r_M - 1.2 r_X|^{0.2})^2$', fontsize=13)
    ax.set_ylabel(r'$B_{MX}^{(1)} \times Z_X^{-0.4}$', fontsize=14)
    ax.set_title(f'Beta(1) - Literature Empirical Equation\n' +
                f'y = {model.beta1_a2:.5f}x² + {model.beta1_a1:.5f}x + {model.beta1_a0:.5f}\n' +
                f'R² = {r2:.5f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Literature empirical correlation plots saved to: {save_path}")
    plt.close()


def main():
    """
    Main function to evaluate the literature empirical model.
    """
    print("="*60)
    print("Pitzer Parameter - Literature Empirical Model")
    print("Using fixed coefficients from published correlations")
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
    
    # Create model (no training needed!)
    print("\nCreating empirical model with literature coefficients...")
    model = PitzerEmpiricalModel()
    
    print("\nLiterature coefficients:")
    print(f"  Beta(0): {model.beta0_slope:.5f} * feature + {model.beta0_intercept:.5f}")
    print(f"  Beta(1): {model.beta1_a2:.5f} * X² + {model.beta1_a1:.5f} * X + {model.beta1_a0:.5f}")
    print("\nNote: Ionic radii will be converted from pm to Angstroms (Å) for predictions")
    print("      Literature equations use Å: 1 Å = 100 pm")
    
    # Evaluate model
    predictions = evaluate_model(model, X, y, "Full Dataset")
    
    # Create plots
    print("\nCreating prediction plots...")
    plot_predictions(y, predictions)
    
    # Create empirical correlation plots
    print("\nCreating literature empirical correlation plots...")
    plot_empirical_correlations(X, y, model)
    
    # Save model
    model_path = '/Users/clara/activity-prediction/model/baseline/empirical_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("\nThis model uses the exact empirical equations from literature.")
    print("No parameters were fitted to the data - these are reference values.")
    print("\nComparison with fitted baseline:")
    print("  - Fitted baseline: Fits the equation form to YOUR data")
    print("  - Empirical model: Uses equations from LITERATURE data")
    print("\nDifferences in performance show how well literature")
    print("correlations generalize to your specific dataset.")


if __name__ == "__main__":
    main()

