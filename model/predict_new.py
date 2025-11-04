"""
Use Trained Model to Predict Pitzer Coefficients for New Electrolytes

This script demonstrates how to load the trained model and make predictions.
"""

import pandas as pd
import numpy as np
import pickle


def load_trained_model(model_path='best_pitzer_model.pkl'):
    """Load the trained model and scaler."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
    return model, scaler


def predict_pitzer_coefficients(model, scaler, X_new):
    """
    Predict Pitzer coefficients for new electrolyte data.
    
    Args:
        model: Trained ML model
        scaler: Fitted StandardScaler
        X_new: DataFrame with feature columns matching training data
    
    Returns:
        DataFrame with predictions for beta_0, beta_1, c_mx
    """
    # Scale features
    X_scaled = scaler.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Create results DataFrame
    target_cols = ['beta_0', 'beta_1', 'c_mx']  # Removed beta_2 (all zeros)
    results = pd.DataFrame(predictions, columns=target_cols, index=X_new.index)
    
    return results


def main():
    """Demonstration of using the trained model."""
    print("=" * 80)
    print("PITZER COEFFICIENT PREDICTION - Using Trained Model")
    print("=" * 80)
    
    # Set up paths relative to project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    model_dir = os.path.join(project_root, 'model')
    
    # Load the trained model
    print("\n[1] Loading trained model...")
    model, scaler = load_trained_model(os.path.join(model_dir, 'best_pitzer_model.pkl'))
    print("✓ Model and scaler loaded successfully")
    
    # Load data for prediction
    print("\n[2] Loading data for prediction...")
    df = pd.read_csv(os.path.join(data_dir, 'ml_ready_dataset.csv'))
    
    # For demonstration, let's predict on the first 5 electrolytes
    electrolyte_names = df['electrolyte_name'].head(5)
    
    # Extract features (excluding name and targets)
    target_cols = ['beta_0', 'beta_1', 'c_mx']  # Removed beta_2 (all zeros)
    feature_cols = [col for col in df.columns if col not in ['electrolyte_name'] + target_cols]
    
    X_new = df[feature_cols].head(5)
    y_true = df[target_cols].head(5)
    
    print(f"✓ Loaded {len(X_new)} electrolytes for prediction")
    
    # Make predictions
    print("\n[3] Making predictions...")
    predictions = predict_pitzer_coefficients(model, scaler, X_new)
    print("✓ Predictions complete")
    
    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    for i, name in enumerate(electrolyte_names):
        print(f"\n{name}:")
        print("-" * 40)
        
        for target in target_cols:
            pred_val = predictions.loc[i, target]
            true_val = y_true.loc[i, target]
            error = abs(pred_val - true_val)
            
            print(f"  {target:8s}: Predicted = {pred_val:8.5f}, Actual = {true_val:8.5f}, Error = {error:8.5f}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL PREDICTION ACCURACY")
    print("=" * 80)
    
    for target in target_cols:
        pred = predictions[target].values
        true = y_true[target].values
        
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true)**2))
        
        print(f"\n{target}:")
        print(f"  Mean Absolute Error (MAE):  {mae:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
    
    print("\n" + "=" * 80)
    print("\nTo predict for new electrolytes:")
    print("  1. Prepare a DataFrame with the same features as the training data")
    print("  2. Ensure feature names match exactly")
    print("  3. Call predict_pitzer_coefficients(model, scaler, X_new)")
    print("=" * 80)


if __name__ == "__main__":
    main()

