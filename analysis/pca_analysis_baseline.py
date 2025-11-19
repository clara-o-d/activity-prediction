"""
PCA analysis script for baseline_with_ion_properties.csv.
Adapted from the original pca_analysis.py to work specifically with the baseline dataset.

Usage: python analysis/pca_analysis_baseline.py [options]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def identify_feature_columns(df):
    """Auto-detect features, targets, and identifiers from column names for baseline dataset."""
    # Pitzer parameters and other target patterns
    target_patterns = ['b_mx_0', 'b_mx_1', 'b_mx_2', 'c_mx_phi', 'beta_', 'target', 'y_']
    
    # Identification columns specific to baseline dataset - use exact matches or specific patterns
    # Note: 'cation' and 'anion' as standalone columns should be excluded, but not 'cation_1_*' or 'anion_1_*'
    id_exact_matches = ['electrolyte', 'cation', 'anion']  # Exact column name matches
    id_patterns = ['name', 'id', 'formula', 'smiles', 'molecule_formula']
    id_type_patterns = ['_type']  # Pattern that matches 'cation_type', 'anion_type', 'electrolyte_type'
    
    feature_cols = []
    target_cols = []
    id_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check exact matches first
        if col in id_exact_matches or col_lower in id_exact_matches:
            id_cols.append(col)
            continue
        
        # Check for type pattern (cation_type, anion_type, electrolyte_type)
        if any(pattern in col_lower for pattern in id_type_patterns):
            id_cols.append(col)
            continue
        
        # Check other ID patterns (but not if it's a feature column like cation_1_*)
        if any(pattern in col_lower for pattern in id_patterns):
            # Don't exclude if it's actually a feature (e.g., cation_1_* starts with cation but is a feature)
            if not (col_lower.startswith('cation_1_') or col_lower.startswith('anion_1_')):
                id_cols.append(col)
                continue
        
        # Check if it's a target column (Pitzer parameters)
        if any(pattern in col_lower for pattern in target_patterns):
            target_cols.append(col)
            continue
        
        # If it's numeric and not an ID or target, it's a feature
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    return feature_cols, target_cols, id_cols


def load_and_prepare_data(dataset_path, train_split=None, random_state=42):
    """Load CSV and separate features/targets. Optionally split train/test."""
    print("=" * 80)
    print("Loading Baseline Ion Properties Data")
    print("=" * 80)
    
    df = pd.read_csv(dataset_path)
    print(f"✓ Loaded {len(df)} electrolyte samples with {len(df.columns)} columns")
    
    feature_cols, target_cols, id_cols = identify_feature_columns(df)
    
    print(f"\nColumn Identification:")
    print(f"  Features: {len(feature_cols)} columns (ion properties, molecular descriptors)")
    print(f"  Targets: {len(target_cols)} columns" + (f" ({', '.join(target_cols)})" if target_cols else ""))
    print(f"  Identifiers: {len(id_cols)} columns" + (f" ({', '.join(id_cols[:5])}{'...' if len(id_cols) > 5 else ''})" if id_cols else ""))
    
    X = df[feature_cols].copy()
    y = df[target_cols].copy() if target_cols else None
    ids = df[id_cols].copy() if id_cols else None
    
    # Handle missing values
    missing_before = X.isna().sum().sum()
    if missing_before > 0:
        print(f"\n⚠️  Found {missing_before} missing values in features. Filling with column means...")
        X = X.fillna(X.mean())
    
    # Optional train/test split
    if train_split is not None:
        print(f"\n📊 Splitting data: {train_split*100:.0f}% train, {(1-train_split)*100:.0f}% test")
        if y is not None:
            X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
                X, y, ids, test_size=1-train_split, random_state=random_state
            )
        else:
            X_train, X_test, ids_train, ids_test = train_test_split(
                X, ids, test_size=1-train_split, random_state=random_state
            )
            y_train, y_test = None, None
            
        return X_train, X_test, y_train, y_test, ids_train, ids_test, feature_cols, target_cols, id_cols
    
    return X, y, ids, feature_cols, target_cols, id_cols


def perform_pca(X, n_components=None, standardize=True):
    """Fit PCA on features. Returns pca object, scaler, and transformed data."""
    print("\n" + "=" * 80)
    print("Performing PCA on Ion Properties Dataset")
    print("=" * 80)
    
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    
    scaler = None
    if standardize:
        print("📏 Standardizing features (mean=0, std=1)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)
    else:
        X_scaled = X_array
    
    if n_components is None:
        print("🔢 Fitting PCA with all components...")
        pca = PCA()
    elif isinstance(n_components, float) and 0 < n_components < 1:
        print(f"🔢 Fitting PCA to explain {n_components*100:.1f}% variance...")
        pca = PCA(n_components=n_components)
    else:
        print(f"🔢 Fitting PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
    
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"✓ PCA complete: {X_pca.shape[1]} components from {X_scaled.shape[1]} features")
    
    return pca, scaler, X_scaled, X_pca


def calculate_variance_stats(pca):
    """Get variance stats and number of components needed for 90%/95% variance."""
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_90 = np.argmax(cumulative_var >= 0.90) + 1
    
    return explained_var, cumulative_var, n_90, n_95


def print_summary_statistics(pca, feature_names, n_90, n_95):
    """Print variance stats and top feature loadings."""
    print("\n" + "=" * 80)
    print("PCA Summary Statistics")
    print("=" * 80)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"\nTotal Variance Explained: {cumulative_var[-1]*100:.2f}%")
    print(f"\nComponents to explain 90% variance: {n_90}")
    print(f"Components to explain 95% variance: {n_95}")
    
    print("\nTop 10 Components by Explained Variance:")
    print("-" * 60)
    print(f"{'Component':<12} {'Variance %':<15} {'Cumulative %':<15}")
    print("-" * 60)
    for i in range(min(10, len(explained_var))):
        print(f"PC{i+1:<11} {explained_var[i]*100:>6.2f}%      {cumulative_var[i]*100:>6.2f}%")
    
    if len(explained_var) > 10:
        print(f"... ({len(explained_var) - 10} more components)")
    
    print("\n" + "=" * 80)
    print("Top Feature Contributions to Principal Components")
    print("=" * 80)
    
    components_to_show = min(5, pca.n_components_)
    for i in range(components_to_show):
        loadings = pca.components_[i]
        top_indices = np.argsort(np.abs(loadings))[-5:][::-1]
        
        print(f"\nPC{i+1} (explains {explained_var[i]*100:.2f}% variance):")
        for idx in top_indices:
            feature = feature_names[idx] if feature_names else f"Feature_{idx+1}"
            loading = loadings[idx]
            print(f"  {feature:<40} {loading:>8.4f}")


def plot_scree_and_cumulative(pca, output_dir):
    """Create scree plot and cumulative variance plot."""
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scree plot
    components = range(1, len(explained_var) + 1)
    ax1.bar(components, explained_var * 100, alpha=0.7, color='steelblue')
    ax1.plot(components, explained_var * 100, 'ro-', linewidth=2, markersize=6)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', fontsize=12)
    ax1.set_title('Scree Plot - Baseline Ion Properties', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(components[:min(20, len(components))])
    
    # Cumulative variance plot
    ax2.plot(components, cumulative_var * 100, 'go-', linewidth=2, markersize=6, label='Cumulative Variance')
    ax2.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% Threshold')
    ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(components[:min(20, len(components))])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scree_and_cumulative_variance_baseline.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved scree and cumulative variance plots")
    plt.close()


def plot_2d_scatter(X_pca, pca, y=None, output_dir=None, sample_names=None):
    """Plot PC1 vs PC2. Color by Pitzer parameter if provided."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if y is not None:
        if isinstance(y, pd.DataFrame) and len(y.columns) > 0:
            # Prefer B_MX_0 parameters for coloring
            if 'B_MX_0_simplified' in y.columns:
                target_col = 'B_MX_0_simplified'
            elif 'B_MX_0_original' in y.columns:
                target_col = 'B_MX_0_original'
            elif 'B_MX_1_simplified' in y.columns:
                target_col = 'B_MX_1_simplified'
            else:
                target_col = y.columns[0]
            
            target_values = y[target_col].values
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=target_values, 
                               cmap='viridis', alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label=target_col)
            title_suffix = f" (colored by {target_col})"
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=100, edgecolors='k', linewidth=0.5)
            title_suffix = ""
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=100, edgecolors='k', linewidth=0.5, color='steelblue')
        title_suffix = ""
    
    if sample_names is not None and len(sample_names) <= 50:
        for i, name in enumerate(sample_names):
            ax.annotate(name, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title(f'First Two Principal Components{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_2d_scatter_baseline.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved 2D scatter plot")
    plt.close()


def plot_loadings_heatmap(pca, feature_names, output_dir, n_components=10):
    """Heatmap showing feature loadings for each component."""
    n_components = min(n_components, pca.n_components_)
    
    loadings = pca.components_[:n_components].T
    pc_names = [f'PC{i+1}' for i in range(n_components)]
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=pc_names)
    
    fig, ax = plt.subplots(figsize=(max(12, n_components), max(10, len(feature_names) * 0.3)))
    
    sns.heatmap(loadings_df, annot=False, cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, fmt='.2f', cbar_kws={'label': 'Loading'}, ax=ax)
    
    ax.set_title(f'Component Loadings - First {n_components} PCs (Baseline Dataset)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_loadings_heatmap_baseline.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved loadings heatmap")
    plt.close()


def plot_biplot(X_pca, pca, feature_names, output_dir, n_features_to_show=10, sample_names=None):
    """Biplot: samples as points, top features as arrows."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=80, edgecolors='k', linewidth=0.5, color='steelblue')
    
    pc1_loadings = pca.components_[0]
    pc2_loadings = pca.components_[1]
    
    # Find top contributing features
    contributions = np.sqrt(pc1_loadings**2 + pc2_loadings**2)
    top_indices = np.argsort(contributions)[-n_features_to_show:][::-1]
    
    # Scale arrows appropriately
    scale = min(
        (X_pca[:, 0].max() - X_pca[:, 0].min()) / (pc1_loadings[top_indices].max() - pc1_loadings[top_indices].min() + 1e-10),
        (X_pca[:, 1].max() - X_pca[:, 1].min()) / (pc2_loadings[top_indices].max() - pc2_loadings[top_indices].min() + 1e-10)
    ) * 0.7
    
    for idx in top_indices:
        feature = feature_names[idx] if feature_names else f"Feature_{idx+1}"
        ax.arrow(0, 0, pc1_loadings[idx] * scale, pc2_loadings[idx] * scale,
                head_width=0.02, head_length=0.02, fc='red', ec='red', linewidth=1.5, alpha=0.7)
        ax.text(pc1_loadings[idx] * scale * 1.1, pc2_loadings[idx] * scale * 1.1,
               feature, fontsize=9, color='red', weight='bold', alpha=0.8)
    
    if sample_names is not None and len(sample_names) <= 30:
        for i, name in enumerate(sample_names):
            ax.annotate(name, (X_pca[i, 0], X_pca[i, 1]), fontsize=7, alpha=0.6)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title(f'Biplot: Electrolytes and Top {n_features_to_show} Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_biplot_baseline.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved biplot")
    plt.close()


def save_pca_transformer(pca, scaler, feature_names, output_dir):
    """Save fitted pca/scaler to pickle file."""
    import pickle
    
    pca_data = {
        'pca': pca,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    output_path = os.path.join(output_dir, 'pca_transformer_baseline.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(pca_data, f)
    
    print(f"✓ Saved PCA transformer to '{output_path}'")


def analyze_feature_groups(pca, feature_names, n_components=5):
    """Analyze which feature groups contribute most to each PC."""
    print("\n" + "=" * 80)
    print("Feature Group Analysis")
    print("=" * 80)
    
    # Categorize features based on actual column names in baseline_with_ion_properties.csv
    # Order matters - check specific patterns first to avoid double-counting
    cation_features = [f for f in feature_names if f.startswith('cation_1_')]
    anion_features = [f for f in feature_names if f.startswith('anion_1_')]
    molecule_features = [f for f in feature_names if f.startswith('molecule_')]
    # Radius features: r_M_angstrom, r_X_angstrom, and radius features from cation/anion (excluding molecule)
    radius_features = [f for f in feature_names if (f == 'r_M_angstrom' or f == 'r_X_angstrom' or 
                          ('radius' in f.lower() and not f.startswith('molecule_')))]
    # Metadata/type features
    metadata_features = [f for f in feature_names if (f not in cation_features + anion_features + molecule_features + radius_features and
                          any(pattern in f.lower() for pattern in ['std_dev', 'ref_', 'max_molality', '_type_numeric', 'electrolyte_type_numeric']))]
    # All other features
    other_features = [f for f in feature_names if f not in cation_features + anion_features + 
                     molecule_features + radius_features + metadata_features]
    
    print(f"\nFeature Groups:")
    print(f"  Cation properties (cation_1_*): {len(cation_features)}")
    print(f"  Anion properties (anion_1_*): {len(anion_features)}")
    print(f"  Molecule properties (molecule_*): {len(molecule_features)}")
    print(f"  Radius/size features (r_M_angstrom, r_X_angstrom, *_radius_*): {len(radius_features)}")
    print(f"  Metadata/type features (std_dev, ref, max_molality, *_type_numeric): {len(metadata_features)}")
    print(f"  Other features: {len(other_features)}")
    
    for i in range(min(n_components, pca.n_components_)):
        loadings = pca.components_[i]
        
        # Calculate contribution from each group
        cation_contrib = np.sum([abs(loadings[feature_names.index(f)]) for f in cation_features if f in feature_names])
        anion_contrib = np.sum([abs(loadings[feature_names.index(f)]) for f in anion_features if f in feature_names])
        molecule_contrib = np.sum([abs(loadings[feature_names.index(f)]) for f in molecule_features if f in feature_names])
        radius_contrib = np.sum([abs(loadings[feature_names.index(f)]) for f in radius_features if f in feature_names])
        metadata_contrib = np.sum([abs(loadings[feature_names.index(f)]) for f in metadata_features if f in feature_names])
        other_contrib = np.sum([abs(loadings[feature_names.index(f)]) for f in other_features if f in feature_names])
        
        total = cation_contrib + anion_contrib + molecule_contrib + radius_contrib + metadata_contrib + other_contrib
        
        if total > 0:
            print(f"\nPC{i+1} Feature Group Contributions:")
            print(f"  Cation: {cation_contrib/total*100:.1f}%")
            print(f"  Anion: {anion_contrib/total*100:.1f}%")
            print(f"  Molecule: {molecule_contrib/total*100:.1f}%")
            print(f"  Radius/Size: {radius_contrib/total*100:.1f}%")
            print(f"  Metadata/Type: {metadata_contrib/total*100:.1f}%")
            print(f"  Other: {other_contrib/total*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Perform PCA analysis on baseline_with_ion_properties.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/pca_analysis_baseline.py
  python analysis/pca_analysis_baseline.py --output results/pca_baseline --n_components 15
  python analysis/pca_analysis_baseline.py --train_split 0.8 --save_transformer
        """
    )
    
    parser.add_argument('--dataset', type=str, default='data/baseline_with_ion_properties.csv',
                       help='Path to CSV dataset file (default: data/baseline_with_ion_properties.csv)')
    parser.add_argument('--output', type=str, default='analysis/pca_results_baseline',
                       help='Output directory for results (default: analysis/pca_results_baseline)')
    parser.add_argument('--n_components', type=int, default=None,
                       help='Number of PCA components (default: all)')
    parser.add_argument('--train_split', type=float, default=None,
                       help='Train/test split ratio (e.g., 0.8 for 80%% train). If provided, PCA is fit only on training data.')
    parser.add_argument('--save_transformer', action='store_true',
                       help='Save PCA transformer for later use')
    parser.add_argument('--no_standardize', action='store_true',
                       help='Skip feature standardization (not recommended)')
    parser.add_argument('--label_samples', action='store_true',
                       help='Label samples with electrolyte names in scatter plots (if <= 50 samples)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset file not found: {args.dataset}")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load data
    if args.train_split:
        X_train, X_test, y_train, y_test, ids_train, ids_test, feature_names, target_names, id_names = load_and_prepare_data(
            args.dataset, train_split=args.train_split
        )
        X_for_pca = X_train
        y_for_plot = y_train
        ids_for_plot = ids_train
    else:
        X, y, ids, feature_names, target_names, id_names = load_and_prepare_data(
            args.dataset, train_split=None
        )
        X_for_pca = X
        y_for_plot = y
        ids_for_plot = ids
    
    # Get sample names for labeling if requested
    sample_names = None
    if args.label_samples and ids_for_plot is not None and 'electrolyte' in ids_for_plot.columns:
        sample_names = ids_for_plot['electrolyte'].values[:50]  # Limit to 50
    
    # Perform PCA
    pca, scaler, X_scaled, X_pca = perform_pca(
        X_for_pca, 
        n_components=args.n_components,
        standardize=not args.no_standardize
    )
    
    # Calculate and print statistics
    explained_var, cumulative_var, n_90, n_95 = calculate_variance_stats(pca)
    print_summary_statistics(pca, feature_names, n_90, n_95)
    
    # Analyze feature groups (specific to baseline dataset)
    analyze_feature_groups(pca, feature_names, n_components=5)
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    plot_scree_and_cumulative(pca, output_dir)
    plot_2d_scatter(X_pca, pca, y_for_plot, output_dir, sample_names)
    plot_loadings_heatmap(pca, feature_names, output_dir, n_components=min(10, pca.n_components_))
    
    if len(X_pca) <= 200:
        plot_biplot(X_pca, pca, feature_names, output_dir, 
                   n_features_to_show=min(10, len(feature_names)))
    else:
        print("⚠️  Skipping biplot (too many samples, would be cluttered)")
    
    # Save transformer if requested
    if args.save_transformer:
        save_pca_transformer(pca, scaler, feature_names, output_dir)
    
    # Final recommendations
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    print(f"\n📊 For 90% variance retention: Use {n_90} components")
    print(f"📊 For 95% variance retention: Use {n_95} components")
    print(f"📊 Original features: {len(feature_names)}")
    print(f"📊 Dimensionality reduction: {len(feature_names)} → {n_90} features ({100*(1-n_90/len(feature_names)):.1f}% reduction)")
    
    print("\n💡 Insights for Baseline Ion Properties:")
    print(f"   - Dataset contains {len(X_for_pca)} electrolyte samples")
    print(f"   - Features include cation/anion properties, molecular descriptors, radii")
    print(f"   - Target variables: {', '.join(target_names) if target_names else 'None detected'}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - scree_and_cumulative_variance_baseline.png")
    print("  - pca_2d_scatter_baseline.png")
    print("  - component_loadings_heatmap_baseline.png")
    if len(X_pca) <= 200:
        print("  - pca_biplot_baseline.png")
    if args.save_transformer:
        print("  - pca_transformer_baseline.pkl")


if __name__ == "__main__":
    main()

