"""
Compare the baseline model (empirical correlations) with the ML model.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/clara/activity-prediction/model/baseline')
from baseline import PitzerBaselineModel


def load_models():
    """Load both baseline and ML models."""
    print("Loading models...")
    
    with open('/Users/clara/activity-prediction/model/baseline/baseline_model.pkl', 'rb') as f:
        baseline_model = pickle.load(f)
    
    with open('/Users/clara/activity-prediction/model/best_pitzer_model.pkl', 'rb') as f:
        ml_model_dict = pickle.load(f)
        ml_model = ml_model_dict['model']
        ml_scaler = ml_model_dict['scaler']
    
    return baseline_model, ml_model, ml_scaler


def load_data():
    """Load both datasets."""
    print("Loading data...")
    
    # Load ML-ready dataset for ML model
    X_ml = pd.read_csv('/Users/clara/activity-prediction/data/ml_ready_dataset_X.csv')
    y_ml = pd.read_csv('/Users/clara/activity-prediction/data/ml_ready_dataset_y.csv')
    
    # Rename columns to match baseline naming
    y_ml = y_ml.rename(columns={'beta_0': 'Beta(0)', 'beta_1': 'Beta(1)'})
    
    # Load clean dataset for baseline model (has ionic radius columns)
    data_clean = pd.read_csv('/Users/clara/activity-prediction/data/clean_activity_dataset.csv')
    
    # Convert to numeric
    required_cols = [
        'Cation_Charge (formal)',
        'Anion_Charge',
        'Cation_Ionic radius (pm) (molecular radius if polyatomic)',
        'Anion_Ionic radius (pm) (molecular radius if polyatomic).1',
        'Beta(0)',
        'Beta(1)'
    ]
    
    for col in required_cols:
        data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
    
    data_clean = data_clean[required_cols].dropna()
    
    X_baseline = data_clean[required_cols[:4]]
    y_baseline = data_clean[required_cols[4:]]
    
    # Match the electrolytes between datasets using the Name column
    # We'll use the intersection of both datasets for fair comparison
    ml_names = set(X_ml['electrolyte_name'].values)
    clean_names = set(pd.read_csv('/Users/clara/activity-prediction/data/clean_activity_dataset.csv')['Name'].values)
    
    return X_ml, y_ml, X_baseline, y_baseline


def evaluate_baseline_on_ml_data(baseline_model, X_baseline):
    """
    Evaluate baseline model on baseline dataset features.
    The baseline model only uses charge and radius features.
    """
    predictions = baseline_model.predict(X_baseline)
    return predictions


def compute_metrics(y_true, y_pred, model_name):
    """Compute and print metrics for a model."""
    metrics = {}
    
    for target in ['Beta(0)', 'Beta(1)']:
        true_vals = y_true[target]
        pred_vals = y_pred[target]
        
        r2 = r2_score(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        mae = mean_absolute_error(true_vals, pred_vals)
        
        metrics[f'{target}_R2'] = r2
        metrics[f'{target}_RMSE'] = rmse
        metrics[f'{target}_MAE'] = mae
    
    # Overall metrics
    all_true = np.concatenate([y_true['Beta(0)'], y_true['Beta(1)']])
    all_pred = np.concatenate([y_pred['Beta(0)'], y_pred['Beta(1)']])
    metrics['Overall_R2'] = r2_score(all_true, all_pred)
    metrics['Overall_RMSE'] = np.sqrt(mean_squared_error(all_true, all_pred))
    metrics['Overall_MAE'] = mean_absolute_error(all_true, all_pred)
    
    return metrics


def print_comparison(baseline_metrics, ml_metrics):
    """Print a comparison table of metrics."""
    print("\n" + "="*80)
    print("MODEL COMPARISON: Baseline vs Machine Learning")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Baseline':<20} {'ML Model':<20} {'Improvement':<15}")
    print("-"*80)
    
    for metric_name in ['Beta(0)_R2', 'Beta(0)_RMSE', 'Beta(0)_MAE',
                       'Beta(1)_R2', 'Beta(1)_RMSE', 'Beta(1)_MAE',
                       'Overall_R2', 'Overall_RMSE', 'Overall_MAE']:
        
        baseline_val = baseline_metrics[metric_name]
        ml_val = ml_metrics[metric_name]
        
        # Calculate improvement
        if 'R2' in metric_name:
            # For R², higher is better
            improvement = ml_val - baseline_val
            improvement_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
        else:
            # For RMSE and MAE, lower is better
            improvement = (baseline_val - ml_val) / baseline_val * 100
            improvement_str = f"-{improvement:.1f}%" if improvement > 0 else f"+{-improvement:.1f}%"
        
        print(f"{metric_name:<25} {baseline_val:<20.4f} {ml_val:<20.4f} {improvement_str:<15}")
    
    print("-"*80)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nBaseline Model:")
    print(f"  - Uses only ionic charge and radius")
    print(f"  - Based on empirical correlations from literature")
    print(f"  - Overall R²: {baseline_metrics['Overall_R2']:.4f}")
    
    print(f"\nML Model:")
    print(f"  - Uses all available ionic properties")
    print(f"  - Trained with gradient boosting")
    print(f"  - Overall R²: {ml_metrics['Overall_R2']:.4f}")
    
    r2_improvement = ml_metrics['Overall_R2'] - baseline_metrics['Overall_R2']
    rmse_improvement = (baseline_metrics['Overall_RMSE'] - ml_metrics['Overall_RMSE']) / baseline_metrics['Overall_RMSE'] * 100
    
    print(f"\nImprovement:")
    print(f"  - R² improvement: +{r2_improvement:.4f}")
    print(f"  - RMSE improvement: {rmse_improvement:.1f}% reduction")


def plot_comparison(y_true, baseline_pred, ml_pred, save_path='model/model_comparison.png'):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, target in enumerate(['Beta(0)', 'Beta(1)']):
        true_vals = y_true[target]
        
        # Baseline predictions
        ax = axes[idx, 0]
        baseline_vals = baseline_pred[target]
        r2_baseline = r2_score(true_vals, baseline_vals)
        rmse_baseline = np.sqrt(mean_squared_error(true_vals, baseline_vals))
        
        ax.scatter(true_vals, baseline_vals, alpha=0.6, s=50, label='Predictions')
        min_val = min(true_vals.min(), baseline_vals.min())
        max_val = max(true_vals.max(), baseline_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        ax.set_xlabel(f'True {target}', fontsize=11)
        ax.set_ylabel(f'Predicted {target}', fontsize=11)
        ax.set_title(f'Baseline Model - {target}\nR² = {r2_baseline:.4f}, RMSE = {rmse_baseline:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ML predictions
        ax = axes[idx, 1]
        ml_vals = ml_pred[target]
        r2_ml = r2_score(true_vals, ml_vals)
        rmse_ml = np.sqrt(mean_squared_error(true_vals, ml_vals))
        
        ax.scatter(true_vals, ml_vals, alpha=0.6, s=50, color='green', label='Predictions')
        min_val = min(true_vals.min(), ml_vals.min())
        max_val = max(true_vals.max(), ml_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        ax.set_xlabel(f'True {target}', fontsize=11)
        ax.set_ylabel(f'Predicted {target}', fontsize=11)
        ax.set_title(f'ML Model - {target}\nR² = {r2_ml:.4f}, RMSE = {rmse_ml:.4f}', 
                    fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plots saved to: {save_path}")
    plt.close()


def main():
    """Main comparison function."""
    print("="*80)
    print("PITZER PARAMETER PREDICTION: MODEL COMPARISON")
    print("="*80)
    
    # Load models and data
    baseline_model, ml_model, ml_scaler = load_models()
    X_ml, y_ml, X_baseline, y_baseline = load_data()
    
    print(f"ML dataset size: {len(X_ml)} samples")
    print(f"Baseline dataset size: {len(X_baseline)} samples")
    
    # Get predictions from both models
    print("\nGenerating predictions...")
    
    # Baseline predictions on baseline dataset
    baseline_pred = evaluate_baseline_on_ml_data(baseline_model, X_baseline)
    
    # ML predictions on ML dataset (need to exclude electrolyte_name column and scale)
    X_ml_features = X_ml.drop(columns=['electrolyte_name'])
    X_ml_scaled = ml_scaler.transform(X_ml_features)
    ml_pred_array = ml_model.predict(X_ml_scaled)
    ml_pred = pd.DataFrame(ml_pred_array, columns=['Beta(0)', 'Beta(1)'])
    
    # Compute metrics
    print("Computing metrics...")
    baseline_metrics = compute_metrics(y_baseline, baseline_pred, "Baseline")
    ml_metrics = compute_metrics(y_ml, ml_pred, "ML Model")
    
    # Print comparison
    print_comparison(baseline_metrics, ml_metrics)
    
    # Create separate plots for each model
    print("\nCreating plots...")
    
    # Plot baseline model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, target in enumerate(['Beta(0)', 'Beta(1)']):
        ax = axes[idx]
        true_vals = y_baseline[target]
        pred_vals = baseline_pred[target]
        
        r2 = r2_score(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=50)
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        ax.set_xlabel(f'True {target}', fontsize=12)
        ax.set_ylabel(f'Predicted {target}', fontsize=12)
        ax.set_title(f'Baseline Model - {target}\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model/baseline/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("Baseline model plots saved to: model/baseline/baseline_comparison.png")
    plt.close()
    
    # Plot ML model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, target in enumerate(['Beta(0)', 'Beta(1)']):
        ax = axes[idx]
        true_vals = y_ml[target]
        pred_vals = ml_pred[target]
        
        r2 = r2_score(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        
        ax.scatter(true_vals, pred_vals, alpha=0.6, s=50, color='green')
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        ax.set_xlabel(f'True {target}', fontsize=12)
        ax.set_ylabel(f'Predicted {target}', fontsize=12)
        ax.set_title(f'ML Model - {target}\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model/baseline/ml_comparison.png', dpi=300, bbox_inches='tight')
    print("ML model plots saved to: model/baseline/ml_comparison.png")
    plt.close()
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)
    print("\nNote: Models evaluated on their respective datasets.")
    print(f"  - Baseline: {len(y_baseline)} samples (needs ionic radius data)")
    print(f"  - ML Model: {len(y_ml)} samples (uses all available features)")


if __name__ == "__main__":
    main()

