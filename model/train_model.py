"""
Train Machine Learning Models to Predict Pitzer Coefficients

This script demonstrates a complete ML workflow:
1. Load and explore the ML-ready dataset
2. Split data into train/test sets
3. Scale features
4. Train multiple models
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
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='ml_ready_dataset.csv'):
    """Load the ML-ready dataset and separate features and targets."""
    print("=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Define feature and target columns
    target_cols = ['beta_0', 'beta_1', 'c_mx']  # Removed beta_2 (all zeros)
    feature_cols = [col for col in df.columns if col not in ['electrolyte_name'] + target_cols]
    
    X = df[feature_cols]
    y = df[target_cols]
    electrolyte_names = df['electrolyte_name']
    
    print(f"✓ Features: {X.shape[1]} columns")
    print(f"✓ Targets: {y.shape[1]} columns (Pitzer coefficients)")
    print(f"✓ No missing values: {not X.isna().any().any() and not y.isna().any().any()}")
    
    return X, y, electrolyte_names, feature_cols, target_cols


def explore_data(X, y, feature_cols, target_cols):
    """Display basic statistics about the dataset."""
    print("\n" + "=" * 80)
    print("Data Exploration")
    print("=" * 80)
    
    print("\nFeature Statistics:")
    print(X.describe().T[['mean', 'std', 'min', 'max']].head(10))
    print("...")
    
    print("\nTarget Statistics:")
    print(y.describe().T[['mean', 'std', 'min', 'max']])
    
    print("\nFeature Ranges:")
    mol_features = [c for c in feature_cols if c.startswith('mol_')]
    cat_features = [c for c in feature_cols if c.startswith('cat_')]
    an_features = [c for c in feature_cols if c.startswith('an_')]
    print(f"  Molecular features: {len(mol_features)}")
    print(f"  Cation features: {len(cat_features)}")
    print(f"  Anion features: {len(an_features)}")


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data and scale features."""
    print("\n" + "=" * 80)
    print("Data Preparation")
    print("=" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ Train set: {len(X_train)} samples ({100*(1-test_size):.0f}%)")
    print(f"✓ Test set: {len(X_test)} samples ({100*test_size:.0f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features scaled (mean=0, std=1)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train, target_cols):
    """Train multiple models and return them."""
    print("\n" + "=" * 80)
    print("Training Models")
    print("=" * 80)
    
    models = {}
    
    # 1. Random Forest
    print("\n[1/5] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print("  ✓ Complete")
    
    # 2. Gradient Boosting
    print("\n[2/5] Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    # Use MultiOutputRegressor for multi-target
    gb_multi = MultiOutputRegressor(gb)
    gb_multi.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_multi
    print("  ✓ Complete")
    
    # 3. Ridge Regression
    print("\n[3/5] Training Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    models['Ridge'] = ridge
    print("  ✓ Complete")
    
    # 4. Lasso Regression
    print("\n[4/5] Training Lasso Regression...")
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    models['Lasso'] = lasso
    print("  ✓ Complete")
    
    # 5. Elastic Net
    print("\n[5/5] Training Elastic Net...")
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
    elastic.fit(X_train, y_train)
    models['Elastic Net'] = elastic
    print("  ✓ Complete")
    
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test, target_cols):
    """Evaluate all models and display results."""
    print("\n" + "=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics for each target
        model_results = {'model': model_name}
        
        for i, target in enumerate(target_cols):
            # Train metrics
            train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])
            train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i]))
            
            # Test metrics
            test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])
            test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i]))
            test_mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
            
            print(f"  {target}:")
            print(f"    Train R² = {train_r2:.4f}, RMSE = {train_rmse:.6f}")
            print(f"    Test  R² = {test_r2:.4f}, RMSE = {test_rmse:.6f}, MAE = {test_mae:.6f}")
            
            model_results[f'{target}_test_r2'] = test_r2
            model_results[f'{target}_test_rmse'] = test_rmse
            model_results[f'{target}_test_mae'] = test_mae
        
        # Average performance
        avg_r2 = np.mean([model_results[f'{t}_test_r2'] for t in target_cols])
        avg_rmse = np.mean([model_results[f'{t}_test_rmse'] for t in target_cols])
        model_results['avg_test_r2'] = avg_r2
        model_results['avg_test_rmse'] = avg_rmse
        
        print(f"  Average Test R² = {avg_r2:.4f}, RMSE = {avg_rmse:.6f}")
        
        results.append(model_results)
    
    return pd.DataFrame(results)


def cross_validate_best_model(best_model, X, y, cv=5):
    """Perform cross-validation on the best model."""
    print("\n" + "=" * 80)
    print("Cross-Validation (Best Model)")
    print("=" * 80)
    
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation for R² score
    cv_scores = cross_val_score(best_model, X, y, cv=kfold, scoring='r2', n_jobs=-1)
    
    print(f"\n{cv}-Fold Cross-Validation R² Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print(f"\nMean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores


def plot_predictions(model, X_test, y_test, target_cols, model_name='Best Model', output_file='prediction_plots.png'):
    """Plot predicted vs actual values for each target."""
    print("\n" + "=" * 80)
    print("Generating Prediction Plots")
    print("=" * 80)
    
    y_pred = model.predict(X_test)
    
    # Dynamically set layout based on number of targets
    n_targets = len(target_cols)
    if n_targets == 1:
        n_rows, n_cols = 1, 1
    elif n_targets == 2:
        n_rows, n_cols = 1, 2
    elif n_targets == 3:
        n_rows, n_cols = 1, 3
    elif n_targets == 4:
        n_rows, n_cols = 2, 2
    else:
        # For more than 4, use a grid that fits
        n_cols = 3
        n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    fig.suptitle(f'{model_name}: Predicted vs Actual Values', fontsize=16, fontweight='bold')
    
    # Handle both single subplot and array of subplots
    if n_targets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, target in enumerate(target_cols):
        ax = axes[i]
        y_true = y_test.iloc[:, i]
        y_p = y_pred[:, i]
        
        # Scatter plot
        ax.scatter(y_true, y_p, alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_p.min())
        max_val = max(y_true.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Metrics
        r2 = r2_score(y_true, y_p)
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        
        ax.set_xlabel('Actual', fontsize=11)
        ax.set_ylabel('Predicted', fontsize=11)
        ax.set_title(f'{target}\nR² = {r2:.4f}, RMSE = {rmse:.6f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to '{output_file}'")
    
    return fig


def save_model(model, scaler, filepath='best_model.pkl'):
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
    print("\nTo load the model later:")
    print(f"  import pickle")
    print(f"  with open('{filepath}', 'rb') as f:")
    print(f"      data = pickle.load(f)")
    print(f"      model = data['model']")
    print(f"      scaler = data['scaler']")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("PITZER COEFFICIENT PREDICTION - ML PIPELINE")
    print("=" * 80)
    
    # Set up paths relative to project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'model')
    
    # 1. Load data
    X, y, electrolyte_names, feature_cols, target_cols = load_data(os.path.join(data_dir, 'ml_ready_dataset.csv'))
    
    # 2. Explore data
    explore_data(X, y, feature_cols, target_cols)
    
    # 3. Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, test_size=0.2)
    
    # 4. Train models
    models = train_models(X_train, y_train, target_cols)
    
    # 5. Evaluate models
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test, target_cols)
    
    # 6. Select best model
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print("\nAverage Test R² by Model:")
    comparison = results_df[['model', 'avg_test_r2', 'avg_test_rmse']].sort_values('avg_test_r2', ascending=False)
    print(comparison.to_string(index=False))
    
    best_model_name = comparison.iloc[0]['model']
    best_model = models[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name}")
    
    # 7. Cross-validation
    X_scaled = scaler.fit_transform(X)
    cv_scores = cross_validate_best_model(best_model, X_scaled, y)
    
    # 8. Plot predictions
    plot_output = os.path.join(model_dir, 'prediction_plots.png')
    plot_predictions(best_model, X_test, y_test, target_cols, best_model_name, plot_output)
    
    # 9. Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "=" * 80)
        print("Feature Importance (Top 10)")
        print("=" * 80)
        
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
    
    # 10. Save best model
    save_model(best_model, scaler, os.path.join(model_dir, 'best_pitzer_model.pkl'))
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE! 🎉")
    print("=" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Average Test R²: {comparison.iloc[0]['avg_test_r2']:.4f}")
    print(f"Average Test RMSE: {comparison.iloc[0]['avg_test_rmse']:.6f}")
    print("\nGenerated Files:")
    print("  - best_pitzer_model.pkl (trained model + scaler)")
    print("  - prediction_plots.png (visualization)")


if __name__ == "__main__":
    main()

