#!/usr/bin/env python3
"""
Simple visualization of Pitzer coefficients by electrolyte type

This script plots the Pitzer coefficients (B_MX_0 and B_MX_1) as a function
of electrolyte type, showing both actual values and model predictions.

Usage:
    python interp/plot_by_electrolyte_type.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Set plotting style
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


def load_data_and_model():
    """Load the dataset and trained model."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'baseline_with_ion_properties.csv'
    model_path = project_root / 'model' / 'best_baseline_ion_model.pkl'
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    return data, model, scaler


def prepare_features(data, scaler):
    """Prepare features for prediction."""
    feature_cols = [
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
    
    X = data[feature_cols]
    X_scaled = scaler.transform(X)
    
    return X_scaled


def plot_by_electrolyte_type(data, predictions, output_path='interp/results/pitzer_by_electrolyte_type.png'):
    """Plot Pitzer coefficients vs electrolyte type."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define electrolyte type labels
    type_labels = {
        11: '1-1',
        12: '1-2',
        21: '2-1',
        22: '2-2',
        31: '3-1',
        41: '4-1'
    }
    
    # Target columns
    target_cols = ['B_MX_0_original', 'B_MX_1_original']
    
    for idx, (ax, target) in enumerate(zip(axes, target_cols)):
        # Get actual and predicted values
        actual = data[target].values
        pred = predictions[:, idx]
        electrolyte_type = data['electrolyte_type_numeric'].values
        
        # Create scatter plots
        # Actual values
        for etype in sorted(data['electrolyte_type_numeric'].unique()):
            mask = electrolyte_type == etype
            label = type_labels.get(etype, str(etype))
            
            # Plot actual
            ax.scatter(electrolyte_type[mask], actual[mask], 
                      s=80, alpha=0.6, edgecolors='black', linewidth=0.5,
                      label=f'{label} (actual)', marker='o')
            
            # Plot predicted (slightly offset for visibility)
            ax.scatter(electrolyte_type[mask] + 0.1, pred[mask], 
                      s=60, alpha=0.6, linewidth=1.5,
                      marker='x', color='red')
        
        # Add a separate legend entry for predictions
        ax.scatter([], [], marker='x', s=60, color='red', 
                  linewidth=1.5, 
                  label='Model predictions')
        
        # Formatting
        ax.set_xlabel('Electrolyte Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(target, fontsize=12, fontweight='bold')
        ax.set_title(f'{target} vs Electrolyte Type', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best', ncol=2)
        
        # Set x-axis ticks to electrolyte types
        unique_types = sorted(data['electrolyte_type_numeric'].unique())
        ax.set_xticks(unique_types)
        ax.set_xticklabels([type_labels.get(t, str(t)) for t in unique_types])
    
    plt.suptitle('Pitzer Coefficients by Electrolyte Type\n(Actual vs Model Predictions)', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")
    plt.close()


def plot_boxplot_by_type(data, predictions, output_path='interp/results/pitzer_boxplot_by_type.png'):
    """Create boxplots of Pitzer coefficients grouped by electrolyte type."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define electrolyte type labels
    type_labels = {
        11: '1-1',
        12: '1-2',
        21: '2-1',
        22: '2-2',
        31: '3-1',
        41: '4-1'
    }
    
    target_cols = ['B_MX_0_original', 'B_MX_1_original']
    
    for col_idx, target in enumerate(target_cols):
        # Actual values
        ax = axes[col_idx, 0]
        
        # Prepare data for boxplot
        data_grouped = []
        labels_grouped = []
        
        for etype in sorted(data['electrolyte_type_numeric'].unique()):
            mask = data['electrolyte_type_numeric'] == etype
            values = data[target][mask].values
            if len(values) > 0:
                data_grouped.append(values)
                labels_grouped.append(type_labels.get(etype, str(etype)))
        
        # Create boxplot
        bp = ax.boxplot(data_grouped, tick_labels=labels_grouped, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_grouped)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Electrolyte Type', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{target} (Actual)', fontsize=11, fontweight='bold')
        ax.set_title(f'Actual {target} Distribution by Type', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Predicted values
        ax = axes[col_idx, 1]
        
        # Prepare data for boxplot
        data_grouped_pred = []
        
        for etype in sorted(data['electrolyte_type_numeric'].unique()):
            mask = data['electrolyte_type_numeric'] == etype
            values = predictions[mask, col_idx]
            if len(values) > 0:
                data_grouped_pred.append(values)
        
        # Create boxplot
        bp = ax.boxplot(data_grouped_pred, tick_labels=labels_grouped, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Electrolyte Type', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{target} (Predicted)', fontsize=11, fontweight='bold')
        ax.set_title(f'Predicted {target} Distribution by Type', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Pitzer Coefficients Distribution by Electrolyte Type', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Boxplot saved to: {output_path}")
    plt.close()


def print_statistics_by_type(data, predictions):
    """Print statistics of coefficients grouped by electrolyte type."""
    print("\n" + "=" * 80)
    print("STATISTICS BY ELECTROLYTE TYPE")
    print("=" * 80)
    
    type_labels = {
        11: '1-1',
        12: '1-2',
        21: '2-1',
        22: '2-2',
        31: '3-1',
        41: '4-1'
    }
    
    target_cols = ['B_MX_0_original', 'B_MX_1_original']
    
    for target_idx, target in enumerate(target_cols):
        print(f"\n{target}:")
        print("-" * 80)
        print(f"{'Type':<10}{'Count':<10}{'Actual Mean':<15}{'Actual Std':<15}{'Pred Mean':<15}{'Pred Std':<15}")
        print("-" * 80)
        
        for etype in sorted(data['electrolyte_type_numeric'].unique()):
            mask = data['electrolyte_type_numeric'] == etype
            actual = data[target][mask].values
            pred = predictions[mask, target_idx]
            
            label = type_labels.get(etype, str(etype))
            count = len(actual)
            
            if count > 0:
                actual_mean = np.mean(actual)
                actual_std = np.std(actual)
                pred_mean = np.mean(pred)
                pred_std = np.std(pred)
                
                print(f"{label:<10}{count:<10}{actual_mean:<15.4f}{actual_std:<15.4f}"
                      f"{pred_mean:<15.4f}{pred_std:<15.4f}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("PITZER COEFFICIENTS BY ELECTROLYTE TYPE")
    print("=" * 80)
    
    # Load data and model
    print("\nLoading data and model...")
    data, model, scaler = load_data_and_model()
    print(f"✓ Loaded {len(data)} samples")
    
    # Prepare features and get predictions
    print("\nGenerating predictions...")
    X_scaled = prepare_features(data, scaler)
    predictions = model.predict(X_scaled)
    print(f"✓ Predictions generated")
    
    # Create output directory
    output_dir = Path('interp/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print statistics
    print_statistics_by_type(data, predictions)
    
    # Create plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    plot_by_electrolyte_type(data, predictions, 
                             output_path=output_dir / 'pitzer_by_electrolyte_type.png')
    
    plot_boxplot_by_type(data, predictions,
                         output_path=output_dir / 'pitzer_boxplot_by_type.png')
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nGenerated plots:")
    print("  - pitzer_by_electrolyte_type.png (scatter plot)")
    print("  - pitzer_boxplot_by_type.png (distribution boxplots)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

