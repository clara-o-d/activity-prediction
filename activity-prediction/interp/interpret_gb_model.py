#!/usr/bin/env python3
"""
Comprehensive Interpretation of Gradient Boosting Model
for Pitzer Coefficient Prediction

This script provides in-depth analysis of the trained model:
1. Feature Importance Analysis (Gain, Split Count, Permutation)
2. Partial Dependence Plots
3. SHAP Analysis (TreeExplainer)
4. Individual Tree Interpretation
5. Leaf Path Analysis

Usage:
    python interp/interpret_gb_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from pathlib import Path
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import plot_tree
import shap

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class GradientBoostingInterpreter:
    """Class to interpret gradient boosting models for Pitzer coefficients."""
    
    def __init__(self, model_path, data_path, output_dir='interp/results'):
        """
        Initialize the interpreter.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model pickle file
        data_path : str
            Path to the dataset CSV file
        output_dir : str
            Directory to save output plots and results
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized results
        self.dirs = {
            'feature_importance': self.output_dir / 'feature_importance',
            'partial_dependence': self.output_dir / 'partial_dependence',
            'shap': self.output_dir / 'shap',
            'trees': self.output_dir / 'trees',
            'leaf_paths': self.output_dir / 'leaf_paths',
            'reports': self.output_dir / 'reports'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        self._load_model()
        self._load_data()
        
        print("=" * 80)
        print("GRADIENT BOOSTING MODEL INTERPRETATION")
        print("=" * 80)
        print(f"Model loaded from: {model_path}")
        print(f"Data loaded from: {data_path}")
        print(f"Output directory: {output_dir}")
        print(f"Dataset size: {len(self.data)} samples")
        print(f"Number of features: {len(self.feature_cols)}")
        print(f"Target variables: {self.target_cols}")
        print("=" * 80)
    
    def _load_model(self):
        """Load the saved model and scaler."""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        
        # Extract individual models from MultiOutputRegressor
        if hasattr(self.model, 'estimators_'):
            self.estimators = self.model.estimators_
            print(f"\n✓ Model type: MultiOutputRegressor with {len(self.estimators)} estimators")
        else:
            self.estimators = [self.model]
            print(f"\n✓ Model type: Single estimator")
    
    def _load_data(self):
        """Load and prepare the dataset."""
        self.data = pd.read_csv(self.data_path)
        
        # Define features and targets (same as training script)
        self.target_cols = ['B_MX_0_original', 'B_MX_1_original']
        
        self.feature_cols = [
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
        
        # Verify all features exist
        missing_features = [f for f in self.feature_cols if f not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing features in dataset: {missing_features}")
        
        # Prepare feature and target matrices
        self.X = self.data[self.feature_cols]
        self.y = self.data[self.target_cols]
        
        # Scale features
        self.X_scaled = self.scaler.transform(self.X)
        
        # Get predictions
        self.y_pred = self.model.predict(self.X_scaled)
        
        print(f"\n✓ Features: {len(self.feature_cols)}")
        print(f"✓ Targets: {len(self.target_cols)}")
        print(f"✓ Samples: {len(self.X)}")
    
    def analyze_feature_importance_all(self):
        """Comprehensive feature importance analysis."""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        # 1. Gain-based importance (default)
        gain_importance = self._get_gain_importance()
        
        # 2. Split-based importance
        split_importance = self._get_split_importance()
        
        # 3. Permutation importance
        perm_importance = self._get_permutation_importance()
        
        # Combine all importance metrics
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'gain_importance': gain_importance,
            'split_importance': split_importance,
            'perm_importance_mean': perm_importance['importances_mean'],
            'perm_importance_std': perm_importance['importances_std']
        })
        
        # Normalize to 0-1 for comparison
        for col in ['gain_importance', 'split_importance', 'perm_importance_mean']:
            max_val = importance_df[col].max()
            if max_val > 0:
                importance_df[f'{col}_norm'] = importance_df[col] / max_val
        
        # Sort by gain importance
        importance_df = importance_df.sort_values('gain_importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(self.dirs['feature_importance'] / 'feature_importance_all.csv', index=False)
        
        # Print results
        print("\nFEATURE IMPORTANCE RANKINGS")
        print("-" * 80)
        print(f"{'Rank':<6}{'Feature':<35}{'Gain':<12}{'Splits':<12}{'Permutation':<15}")
        print("-" * 80)
        for idx, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{idx:<6}{row['feature']:<35}{row['gain_importance']:<12.6f}"
                  f"{row['split_importance']:<12.0f}{row['perm_importance_mean']:<15.6f}")
        
        # Plot comparison
        self._plot_importance_comparison(importance_df)
        
        return importance_df
    
    def _get_gain_importance(self):
        """Get gain-based feature importance (averaged across targets)."""
        importances = []
        for estimator in self.estimators:
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)
        
        return np.mean(importances, axis=0) if importances else np.zeros(len(self.feature_cols))
    
    def _get_split_importance(self):
        """Get split count-based importance."""
        split_counts = np.zeros(len(self.feature_cols))
        
        for estimator in self.estimators:
            if hasattr(estimator, 'estimators_'):
                # GradientBoostingRegressor has estimators_ attribute
                for tree_group in estimator.estimators_:
                    tree = tree_group[0]  # Each is a 1-element array
                    tree_struct = tree.tree_
                    
                    # Count splits for each feature
                    for i in range(tree_struct.node_count):
                        if tree_struct.feature[i] >= 0:  # Not a leaf node
                            split_counts[tree_struct.feature[i]] += 1
        
        return split_counts
    
    def _get_permutation_importance(self, n_repeats=10):
        """Calculate permutation importance."""
        print("\nCalculating permutation importance (this may take a moment)...")
        
        perm_importance = permutation_importance(
            self.model, 
            self.X_scaled, 
            self.y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        return perm_importance
    
    def _plot_importance_comparison(self, importance_df):
        """Plot comparison of different importance metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Top 15 features for each metric
        top_n = 15
        
        metrics = [
            ('gain_importance', 'Gain-Based Importance'),
            ('split_importance', 'Split Count'),
            ('perm_importance_mean', 'Permutation Importance')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx]
            
            # Get top features for this metric
            top_features = importance_df.nlargest(top_n, metric)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            values = top_features[metric].values
            
            bars = ax.barh(y_pos, values, color=plt.cm.viridis(values / values.max()))
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'].values, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                if metric == 'split_importance':
                    label = f'{val:.0f}'
                else:
                    label = f'{val:.4f}'
                ax.text(val, i, f' {label}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.dirs['feature_importance'] / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved feature importance comparison to {self.dirs['feature_importance'] / 'feature_importance_comparison.png'}")
        plt.close()
    
    def plot_partial_dependence(self, top_n=10):
        """Create partial dependence plots for top features."""
        print("\n" + "=" * 80)
        print("PARTIAL DEPENDENCE ANALYSIS")
        print("=" * 80)
        
        # Get top features by importance
        gain_importance = self._get_gain_importance()
        top_features_idx = np.argsort(gain_importance)[-top_n:][::-1]
        
        print(f"\nCreating partial dependence plots for top {top_n} features...")
        
        # Create PDP for each target
        for target_idx, target_name in enumerate(self.target_cols):
            print(f"\n  Target: {target_name}")
            
            # Get the estimator for this target
            estimator = self.estimators[target_idx]
            
            # Create figure with subplots
            n_cols = 3
            n_rows = (top_n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
            axes = axes.flatten()
            
            for plot_idx, feat_idx in enumerate(top_features_idx):
                ax = axes[plot_idx]
                feature_name = self.feature_cols[feat_idx]
                
                # Calculate partial dependence
                pd_result = partial_dependence(
                    estimator,
                    self.X_scaled,
                    features=[feat_idx],
                    grid_resolution=50
                )
                
                # Get the actual feature values for x-axis (unscaled)
                feature_values_scaled = pd_result['grid_values'][0]
                # Inverse transform to get original scale
                dummy_data = np.zeros((len(feature_values_scaled), len(self.feature_cols)))
                dummy_data[:, feat_idx] = feature_values_scaled
                feature_values_original = self.scaler.inverse_transform(dummy_data)[:, feat_idx]
                
                # Plot
                ax.plot(feature_values_original, pd_result['average'][0], linewidth=2.5, color='steelblue')
                ax.set_xlabel(feature_name, fontsize=10, fontweight='bold')
                ax.set_ylabel(f'Partial Dependence', fontsize=10)
                ax.set_title(f'{feature_name}\n(Importance: {gain_importance[feat_idx]:.4f})', 
                           fontsize=10, fontweight='bold')
                ax.grid(alpha=0.3)
                
                # Add rug plot to show data distribution
                ax_rug = ax.twinx()
                ax_rug.hist(self.X.iloc[:, feat_idx], bins=30, alpha=0.2, color='gray')
                ax_rug.set_ylabel('Data Distribution', fontsize=8, alpha=0.7)
                ax_rug.tick_params(axis='y', labelsize=8)
            
            # Remove empty subplots
            for idx in range(top_n, len(axes)):
                fig.delaxes(axes[idx])
            
            plt.suptitle(f'Partial Dependence Plots - {target_name}', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            output_file = self.dirs['partial_dependence'] / f'partial_dependence_{target_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved to {output_file}")
            plt.close()
        
        # Also create 2D interaction plots for top 2 features
        self._plot_2d_partial_dependence(top_features_idx[:2])
    
    def _plot_2d_partial_dependence(self, feature_indices):
        """Create 2D partial dependence plots for feature interactions."""
        print("\n  Creating 2D interaction plots for top 2 features...")
        
        for target_idx, target_name in enumerate(self.target_cols):
            estimator = self.estimators[target_idx]
            
            # Calculate 2D partial dependence
            # Convert to tuple for 2D PDP
            pd_result = partial_dependence(
                estimator,
                self.X_scaled,
                features=[(int(feature_indices[0]), int(feature_indices[1]))],
                grid_resolution=30
            )
            
            # Get original scale values
            XX, YY = np.meshgrid(pd_result['grid_values'][0], pd_result['grid_values'][1])
            
            # Inverse transform (approximate)
            dummy_x = np.zeros((len(pd_result['grid_values'][0]), len(self.feature_cols)))
            dummy_x[:, feature_indices[0]] = pd_result['grid_values'][0]
            x_original = self.scaler.inverse_transform(dummy_x)[:, feature_indices[0]]
            
            dummy_y = np.zeros((len(pd_result['grid_values'][1]), len(self.feature_cols)))
            dummy_y[:, feature_indices[1]] = pd_result['grid_values'][1]
            y_original = self.scaler.inverse_transform(dummy_y)[:, feature_indices[1]]
            
            XX_orig, YY_orig = np.meshgrid(x_original, y_original)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            contour = ax.contourf(XX_orig, YY_orig, pd_result['average'][0].T, 
                                 levels=20, cmap='RdYlBu_r')
            plt.colorbar(contour, ax=ax, label='Partial Dependence')
            
            ax.set_xlabel(self.feature_cols[feature_indices[0]], fontsize=12, fontweight='bold')
            ax.set_ylabel(self.feature_cols[feature_indices[1]], fontsize=12, fontweight='bold')
            ax.set_title(f'2D Partial Dependence - {target_name}\n' +
                        f'{self.feature_cols[feature_indices[0]]} × {self.feature_cols[feature_indices[1]]}',
                        fontsize=13, fontweight='bold')
            
            # Add scatter of actual data points
            ax.scatter(self.X.iloc[:, feature_indices[0]], 
                      self.X.iloc[:, feature_indices[1]], 
                      c='black', s=5, alpha=0.3, label='Data points')
            ax.legend()
            
            output_file = self.dirs['partial_dependence'] / f'partial_dependence_2d_{target_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"    ✓ Saved to {output_file}")
            plt.close()
    
    def shap_analysis(self, sample_size=None):
        """Perform SHAP analysis using TreeExplainer."""
        print("\n" + "=" * 80)
        print("SHAP ANALYSIS")
        print("=" * 80)
        
        # Use subset of data if specified
        if sample_size and sample_size < len(self.X_scaled):
            print(f"\nUsing {sample_size} samples for SHAP analysis (for computational efficiency)...")
            indices = np.random.RandomState(42).choice(len(self.X_scaled), sample_size, replace=False)
            X_shap = self.X_scaled[indices]
            X_df_shap = self.X.iloc[indices]
        else:
            X_shap = self.X_scaled
            X_df_shap = self.X
            print(f"\nUsing all {len(X_shap)} samples for SHAP analysis...")
        
        # Analyze each target separately
        for target_idx, target_name in enumerate(self.target_cols):
            print(f"\n  Analyzing target: {target_name}")
            estimator = self.estimators[target_idx]
            
            # Create TreeExplainer
            print("    Creating SHAP TreeExplainer...")
            explainer = shap.TreeExplainer(estimator)
            
            # Calculate SHAP values
            print("    Calculating SHAP values...")
            shap_values = explainer.shap_values(X_shap)
            
            # Save SHAP values
            shap_df = pd.DataFrame(shap_values, columns=self.feature_cols)
            shap_df.to_csv(self.dirs['shap'] / f'shap_values_{target_name}.csv', index=False)
            
            # 1. Summary plot (bar)
            print("    Creating summary plots...")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_df_shap, feature_names=self.feature_cols, 
                            plot_type='bar', show=False)
            plt.title(f'SHAP Feature Importance - {target_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.dirs['shap'] / f'shap_summary_bar_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Summary plot (beeswarm)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_df_shap, feature_names=self.feature_cols, show=False)
            plt.title(f'SHAP Summary Plot - {target_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.dirs['shap'] / f'shap_summary_beeswarm_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Dependence plots for top 5 features
            print("    Creating dependence plots for top features...")
            shap_importance = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(shap_importance)[-5:][::-1]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for plot_idx, feat_idx in enumerate(top_features_idx):
                shap.dependence_plot(
                    feat_idx, 
                    shap_values, 
                    X_df_shap,
                    feature_names=self.feature_cols,
                    ax=axes[plot_idx],
                    show=False
                )
                axes[plot_idx].set_title(f'{self.feature_cols[feat_idx]}', fontsize=11, fontweight='bold')
            
            # Remove last empty subplot
            fig.delaxes(axes[5])
            
            plt.suptitle(f'SHAP Dependence Plots (Top 5 Features) - {target_name}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.dirs['shap'] / f'shap_dependence_{target_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Force plot for a few example predictions
            print("    Creating force plots for example predictions...")
            
            # Select interesting examples (high, medium, low prediction)
            predictions = estimator.predict(X_shap)
            example_indices = [
                np.argmax(predictions),  # Highest prediction
                np.argsort(predictions)[len(predictions)//2],  # Median
                np.argmin(predictions)  # Lowest prediction
            ]
            
            for ex_type, ex_idx in zip(['high', 'medium', 'low'], example_indices):
                # Create force plot
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[ex_idx],
                    X_df_shap.iloc[ex_idx],
                    feature_names=self.feature_cols,
                    matplotlib=True,
                    show=False
                )
                plt.title(f'SHAP Force Plot ({ex_type} prediction) - {target_name}\n' +
                         f'Prediction: {predictions[ex_idx]:.4f}', fontsize=11, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.dirs['shap'] / f'shap_force_{ex_type}_{target_name}.png', 
                          dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"    ✓ SHAP analysis complete for {target_name}")
            print(f"      - Summary plots saved")
            print(f"      - Dependence plots saved")
            print(f"      - Force plots saved")
            print(f"      - SHAP values saved to CSV")
    
    def analyze_trees(self, n_trees=3):
        """Visualize and analyze individual trees."""
        print("\n" + "=" * 80)
        print("TREE INTERPRETATION")
        print("=" * 80)
        
        for target_idx, target_name in enumerate(self.target_cols):
            print(f"\n  Analyzing trees for target: {target_name}")
            estimator = self.estimators[target_idx]
            
            if not hasattr(estimator, 'estimators_'):
                print("    ⚠ Estimator doesn't have tree structure")
                continue
            
            n_estimators = len(estimator.estimators_)
            print(f"    Total trees in ensemble: {n_estimators}")
            
            # Analyze first, middle, and last trees
            tree_indices = [0, n_estimators // 2, n_estimators - 1]
            tree_indices = tree_indices[:n_trees]
            
            for tree_idx in tree_indices:
                tree = estimator.estimators_[tree_idx][0]
                
                # Get tree statistics
                tree_stats = self._get_tree_statistics(tree)
                
                print(f"\n    Tree {tree_idx} statistics:")
                print(f"      - Nodes: {tree_stats['n_nodes']}")
                print(f"      - Leaves: {tree_stats['n_leaves']}")
                print(f"      - Max depth: {tree_stats['max_depth']}")
                print(f"      - Avg depth: {tree_stats['avg_depth']:.2f}")
                
                # Visualize tree (if not too large)
                if tree_stats['n_nodes'] < 100:
                    fig, ax = plt.subplots(figsize=(20, 12))
                    plot_tree(
                        tree,
                        feature_names=self.feature_cols,
                        filled=True,
                        rounded=True,
                        fontsize=8,
                        ax=ax
                    )
                    plt.title(f'Tree {tree_idx} - {target_name}\n' +
                             f'Nodes: {tree_stats["n_nodes"]}, Max Depth: {tree_stats["max_depth"]}',
                             fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(self.dirs['trees'] / f'tree_{tree_idx}_{target_name}.png', 
                              dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"      ✓ Tree visualization saved")
                else:
                    print(f"      ⚠ Tree too large to visualize ({tree_stats['n_nodes']} nodes)")
            
            # Aggregate statistics across all trees
            all_tree_stats = self._aggregate_tree_statistics(estimator)
            self._plot_tree_statistics(all_tree_stats, target_name)
    
    def _get_tree_statistics(self, tree):
        """Get statistics for a single tree."""
        tree_struct = tree.tree_
        
        # Count leaves
        n_leaves = np.sum(tree_struct.feature < 0)
        n_nodes = tree_struct.node_count
        
        # Calculate depths
        def get_node_depth(tree_struct, node_id, depth=0):
            if tree_struct.feature[node_id] < 0:  # Leaf
                return depth
            left_depth = get_node_depth(tree_struct, tree_struct.children_left[node_id], depth + 1)
            right_depth = get_node_depth(tree_struct, tree_struct.children_right[node_id], depth + 1)
            return max(left_depth, right_depth)
        
        def get_all_depths(tree_struct, node_id=0, depth=0):
            depths = []
            if tree_struct.feature[node_id] < 0:  # Leaf
                depths.append(depth)
            else:
                depths.extend(get_all_depths(tree_struct, tree_struct.children_left[node_id], depth + 1))
                depths.extend(get_all_depths(tree_struct, tree_struct.children_right[node_id], depth + 1))
            return depths
        
        max_depth = get_node_depth(tree_struct, 0)
        all_depths = get_all_depths(tree_struct)
        avg_depth = np.mean(all_depths)
        
        return {
            'n_nodes': n_nodes,
            'n_leaves': n_leaves,
            'max_depth': max_depth,
            'avg_depth': avg_depth
        }
    
    def _aggregate_tree_statistics(self, estimator):
        """Aggregate statistics across all trees in the ensemble."""
        all_stats = {
            'n_nodes': [],
            'n_leaves': [],
            'max_depth': [],
            'avg_depth': []
        }
        
        for tree_group in estimator.estimators_:
            tree = tree_group[0]
            stats = self._get_tree_statistics(tree)
            for key in all_stats:
                all_stats[key].append(stats[key])
        
        return all_stats
    
    def _plot_tree_statistics(self, tree_stats, target_name):
        """Plot aggregated tree statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        stats_to_plot = [
            ('n_nodes', 'Number of Nodes'),
            ('n_leaves', 'Number of Leaves'),
            ('max_depth', 'Maximum Depth'),
            ('avg_depth', 'Average Depth')
        ]
        
        for idx, (stat_key, stat_label) in enumerate(stats_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            data = tree_stats[stat_key]
            
            # Histogram
            ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Add statistics
            mean_val = np.mean(data)
            median_val = np.median(data)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            
            ax.set_xlabel(stat_label, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Distribution of {stat_label}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Tree Statistics Across Ensemble - {target_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.dirs['trees'] / f'tree_statistics_{target_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Tree statistics plot saved")
    
    def analyze_leaf_paths(self, n_samples=100):
        """Analyze decision paths and leaf distributions."""
        print("\n" + "=" * 80)
        print("LEAF PATH ANALYSIS")
        print("=" * 80)
        
        # Sample data for analysis
        if n_samples < len(self.X_scaled):
            indices = np.random.RandomState(42).choice(len(self.X_scaled), n_samples, replace=False)
            X_sample = self.X_scaled[indices]
            X_df_sample = self.X.iloc[indices]
            y_sample = self.y.iloc[indices]
        else:
            X_sample = self.X_scaled
            X_df_sample = self.X
            y_sample = self.y
        
        for target_idx, target_name in enumerate(self.target_cols):
            print(f"\n  Analyzing leaf paths for target: {target_name}")
            estimator = self.estimators[target_idx]
            
            if not hasattr(estimator, 'estimators_'):
                print("    ⚠ Estimator doesn't have tree structure")
                continue
            
            # Get decision paths and leaf IDs
            leaf_info = self._get_leaf_information(estimator, X_sample)
            
            # Analyze feature usage in paths
            feature_usage = self._analyze_feature_usage_in_paths(estimator, X_sample)
            
            # Plot feature usage
            self._plot_feature_usage(feature_usage, target_name)
            
            # Analyze leaf value distributions
            self._analyze_leaf_distributions(estimator, X_sample, target_name)
            
            # Create path length distribution
            self._plot_path_lengths(leaf_info, target_name)
            
            # Example: show detailed path for a few samples
            self._show_example_paths(estimator, X_df_sample, y_sample, target_name, n_examples=3)
    
    def _get_leaf_information(self, estimator, X_sample):
        """Get leaf indices and path information for samples."""
        leaf_info = {
            'leaf_indices': [],
            'path_lengths': []
        }
        
        # Analyze first few trees
        for tree_idx in range(min(10, len(estimator.estimators_))):
            tree = estimator.estimators_[tree_idx][0]
            
            # Get leaf indices
            leaf_indices = tree.apply(X_sample)
            leaf_info['leaf_indices'].append(leaf_indices)
            
            # Calculate path lengths
            decision_path = tree.decision_path(X_sample)
            path_lengths = decision_path.sum(axis=1).A1
            leaf_info['path_lengths'].append(path_lengths)
        
        return leaf_info
    
    def _analyze_feature_usage_in_paths(self, estimator, X_sample):
        """Analyze which features are used in decision paths."""
        feature_usage = np.zeros(len(self.feature_cols))
        
        # Sample first 20 trees
        for tree_idx in range(min(20, len(estimator.estimators_))):
            tree = estimator.estimators_[tree_idx][0]
            tree_struct = tree.tree_
            
            # Get decision path for samples
            decision_path = tree.decision_path(X_sample)
            
            # Count feature usage
            for sample_idx in range(len(X_sample)):
                # Get nodes in path
                node_indicator = decision_path.toarray()[sample_idx]
                nodes_in_path = np.where(node_indicator)[0]
                
                # Count features used
                for node in nodes_in_path:
                    if tree_struct.feature[node] >= 0:  # Not a leaf
                        feature_usage[tree_struct.feature[node]] += 1
        
        return feature_usage / feature_usage.sum() if feature_usage.sum() > 0 else feature_usage
    
    def _plot_feature_usage(self, feature_usage, target_name):
        """Plot feature usage frequency in decision paths."""
        # Create dataframe and sort
        usage_df = pd.DataFrame({
            'feature': self.feature_cols,
            'usage_frequency': feature_usage
        }).sort_values('usage_frequency', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(usage_df))
        bars = ax.barh(y_pos, usage_df['usage_frequency'].values, 
                      color=plt.cm.viridis(usage_df['usage_frequency'].values / 
                                          usage_df['usage_frequency'].max()))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(usage_df['feature'].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Usage Frequency in Decision Paths', fontsize=11)
        ax.set_title(f'Feature Usage in Decision Paths - {target_name}', 
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, usage_df['usage_frequency'].values)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.dirs['leaf_paths'] / f'feature_usage_paths_{target_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Feature usage plot saved")
    
    def _analyze_leaf_distributions(self, estimator, X_sample, target_name):
        """Analyze the distribution of leaf node predictions."""
        # Get predictions from first 10 trees
        tree_predictions = []
        
        for tree_idx in range(min(10, len(estimator.estimators_))):
            tree = estimator.estimators_[tree_idx][0]
            tree_pred = tree.predict(X_sample)
            tree_predictions.append(tree_pred)
        
        tree_predictions = np.array(tree_predictions)
        
        # Plot
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for tree_idx in range(min(10, len(tree_predictions))):
            ax = axes[tree_idx]
            
            ax.hist(tree_predictions[tree_idx], bins=30, alpha=0.7, 
                   color='steelblue', edgecolor='black')
            ax.set_xlabel('Prediction Value', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title(f'Tree {tree_idx}', fontsize=10, fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Leaf Prediction Distributions - {target_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.dirs['leaf_paths'] / f'leaf_distributions_{target_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Leaf distribution plot saved")
    
    def _plot_path_lengths(self, leaf_info, target_name):
        """Plot distribution of path lengths."""
        # Combine path lengths from all trees
        all_path_lengths = np.concatenate(leaf_info['path_lengths'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(all_path_lengths, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        mean_length = np.mean(all_path_lengths)
        median_length = np.median(all_path_lengths)
        
        ax.axvline(mean_length, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_length:.1f}')
        ax.axvline(median_length, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_length:.1f}')
        
        ax.set_xlabel('Path Length (Number of Decisions)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution of Decision Path Lengths - {target_name}', 
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['leaf_paths'] / f'path_lengths_{target_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Path length plot saved")
    
    def _show_example_paths(self, estimator, X_df_sample, y_sample, target_name, n_examples=3):
        """Show detailed decision paths for example predictions."""
        print(f"\n    Example decision paths (first tree):")
        
        tree = estimator.estimators_[0][0]
        tree_struct = tree.tree_
        
        # Get predictions
        predictions = tree.predict(X_df_sample.values)
        
        # Select diverse examples
        indices = [
            np.argmax(predictions),
            np.argsort(predictions)[len(predictions)//2],
            np.argmin(predictions)
        ][:n_examples]
        
        for idx in indices:
            print(f"\n    Sample {idx} (Actual: {y_sample.iloc[idx, self.target_cols.index(target_name)]:.4f}, Predicted: {predictions[idx]:.4f})")
            print(f"    {'-' * 70}")
            
            # Get decision path
            decision_path = tree.decision_path(X_df_sample.iloc[idx:idx+1].values)
            node_indicator = decision_path.toarray()[0]
            nodes_in_path = np.where(node_indicator)[0]
            
            for node_id in nodes_in_path:
                if tree_struct.feature[node_id] >= 0:  # Not a leaf
                    feature_idx = tree_struct.feature[node_id]
                    threshold = tree_struct.threshold[node_id]
                    feature_value = X_df_sample.iloc[idx, feature_idx]
                    
                    # Determine direction
                    if feature_value <= threshold:
                        direction = "≤"
                        next_node = tree_struct.children_left[node_id]
                    else:
                        direction = ">"
                        next_node = tree_struct.children_right[node_id]
                    
                    print(f"      Node {node_id}: {self.feature_cols[feature_idx]} = {feature_value:.4f} {direction} {threshold:.4f}")
                else:
                    print(f"      Leaf {node_id}: Prediction = {tree_struct.value[node_id][0, 0]:.4f}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("GRADIENT BOOSTING MODEL INTERPRETATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Model information
        report_lines.append("MODEL INFORMATION")
        report_lines.append("-" * 80)
        report_lines.append(f"Model path: {self.model_path}")
        report_lines.append(f"Data path: {self.data_path}")
        report_lines.append(f"Model type: {type(self.model).__name__}")
        report_lines.append(f"Number of estimators: {len(self.estimators)}")
        report_lines.append(f"Target variables: {', '.join(self.target_cols)}")
        report_lines.append("")
        
        # Data information
        report_lines.append("DATA INFORMATION")
        report_lines.append("-" * 80)
        report_lines.append(f"Number of samples: {len(self.data)}")
        report_lines.append(f"Number of features: {len(self.feature_cols)}")
        report_lines.append(f"Features: {', '.join(self.feature_cols)}")
        report_lines.append("")
        
        # Model performance
        report_lines.append("MODEL PERFORMANCE")
        report_lines.append("-" * 80)
        for target_idx, target_name in enumerate(self.target_cols):
            y_true = self.y.iloc[:, target_idx]
            y_pred = self.y_pred[:, target_idx]
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            report_lines.append(f"{target_name}:")
            report_lines.append(f"  R² Score: {r2:.4f}")
            report_lines.append(f"  RMSE: {rmse:.6f}")
            report_lines.append(f"  MAE: {mae:.6f}")
        report_lines.append("")
        
        # Files generated
        report_lines.append("OUTPUT FILES GENERATED")
        report_lines.append("-" * 80)
        
        # List files by directory
        for dir_name, dir_path in self.dirs.items():
            files = sorted(dir_path.glob('*'))
            if files:
                report_lines.append(f"\n{dir_name.replace('_', ' ').title()}:")
                for f in files:
                    report_lines.append(f"  - {f.name}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(self.dirs['reports'] / 'interpretation_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved to {self.dirs['reports'] / 'interpretation_report.txt'}")


def main():
    """Main execution function."""
    import os
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / 'model' / 'best_baseline_ion_model.pkl'
    data_path = project_root / 'data' / 'baseline_with_ion_properties.csv'
    output_dir = project_root / 'interp' / 'results'
    
    # Verify paths exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Create interpreter
    interpreter = GradientBoostingInterpreter(
        model_path=str(model_path),
        data_path=str(data_path),
        output_dir=str(output_dir)
    )
    
    # Run all analyses
    print("\n" + "=" * 80)
    print("STARTING COMPREHENSIVE INTERPRETATION")
    print("=" * 80)
    
    # 1. Feature Importance Analysis
    importance_df = interpreter.analyze_feature_importance_all()
    
    # 2. Partial Dependence Plots
    interpreter.plot_partial_dependence(top_n=10)
    
    # 3. SHAP Analysis
    interpreter.shap_analysis(sample_size=200)  # Use 200 samples for SHAP
    
    # 4. Tree Interpretation
    interpreter.analyze_trees(n_trees=3)
    
    # 5. Leaf Path Analysis
    interpreter.analyze_leaf_paths(n_samples=100)
    
    # 6. Generate Summary Report
    interpreter.generate_summary_report()
    
    print("\n" + "=" * 80)
    print("INTERPRETATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nResults organized in subdirectories:")
    print("  - feature_importance/  : Feature importance analysis (gain, split, permutation)")
    print("  - partial_dependence/  : Partial dependence plots (1D and 2D)")
    print("  - shap/                : SHAP analysis (summary, dependence, force plots)")
    print("  - trees/               : Tree visualizations and statistics")
    print("  - leaf_paths/          : Leaf path and distribution analysis")
    print("  - reports/             : Comprehensive summary report")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

