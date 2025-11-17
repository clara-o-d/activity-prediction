#!/usr/bin/env python3
"""
Script to merge baseline_data.csv with ml_ready_pitzer_ions.csv
Adds ion properties from ml_ready_pitzer_ions.csv to baseline_data.csv
for matching molecules, excluding the pitzer coefficients (beta0 and beta1).
Creates a machine learning ready dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def normalize_molecule_name(name):
    """
    Normalize molecule names to improve matching.
    Removes (aq) suffix and handles common variations.
    """
    # Remove (aq) suffix if present
    name = name.replace('(aq)', '').strip()
    return name


def load_and_prepare_data():
    """Load both datasets and prepare them for merging."""
    
    # Load baseline data with error handling for malformed data
    # Using engine='python' for more robust parsing
    baseline_df = pd.read_csv('data/baseline_data.csv', 
                             engine='python',
                             na_values=['-', ''],
                             keep_default_na=True,
                             on_bad_lines='warn')
    print(f"Loaded baseline_data.csv: {len(baseline_df)} rows, {len(baseline_df.columns)} columns")
    
    # Load ml_ready_pitzer_ions data
    pitzer_ions_df = pd.read_csv('data/ml_ready_pitzer_ions.csv')
    print(f"Loaded ml_ready_pitzer_ions.csv: {len(pitzer_ions_df)} rows, {len(pitzer_ions_df.columns)} columns")
    
    # Remove pitzer coefficients AND charge/radii columns from pitzer_ions_df
    # We want to use ONLY the charges and radii from baseline_data.csv
    columns_to_exclude = [
        'pitzer_Beta0', 'pitzer_Beta1',  # Pitzer coefficients
        'cation_1_charge', 'anion_1_charge',  # Charges (use baseline values)
        'cation_1_radius_ionic', 'anion_1_radius_ionic'  # Ionic radii (use baseline values)
    ]
    pitzer_ions_features = pitzer_ions_df.drop(columns=columns_to_exclude, errors='ignore')
    print(f"Excluded pitzer coefficients and charge/radii columns, remaining columns: {len(pitzer_ions_features.columns)}")
    
    # Normalize molecule names for matching
    baseline_df['electrolyte_normalized'] = baseline_df['electrolyte'].apply(normalize_molecule_name)
    pitzer_ions_features['molecule_normalized'] = pitzer_ions_features['molecule'].apply(normalize_molecule_name)
    
    return baseline_df, pitzer_ions_features


def merge_datasets(baseline_df, pitzer_ions_features):
    """Merge the datasets on molecule names."""
    
    # Merge on normalized molecule names
    merged_df = baseline_df.merge(
        pitzer_ions_features,
        left_on='electrolyte_normalized',
        right_on='molecule_normalized',
        how='inner',  # Only keep molecules present in both datasets
        suffixes=('_baseline', '_pitzer')
    )
    
    print(f"\nMerged dataset: {len(merged_df)} rows")
    print(f"Matched {len(merged_df)} out of {len(baseline_df)} baseline molecules")
    
    # Report unmatched molecules
    matched_electrolytes = set(merged_df['electrolyte'])
    unmatched_baseline = set(baseline_df['electrolyte']) - matched_electrolytes
    if unmatched_baseline:
        print(f"\nUnmatched molecules from baseline_data.csv ({len(unmatched_baseline)}):")
        for mol in sorted(unmatched_baseline):
            print(f"  - {mol}")
    
    return merged_df


def prepare_ml_ready_dataset(merged_df):
    """
    Prepare the dataset for machine learning:
    - Drop unnecessary columns
    - Handle missing values
    - Convert categorical variables to numeric
    - Organize columns logically
    """
    
    # Drop normalized columns and duplicate molecule names
    columns_to_drop = ['electrolyte_normalized', 'molecule_normalized', 'molecule']
    ml_df = merged_df.drop(columns=columns_to_drop, errors='ignore')
    
    # Convert categorical columns to numeric
    # cation_type and anion_type: 'c' -> 0, 'k' -> 1
    if 'cation_type' in ml_df.columns:
        ml_df['cation_type_numeric'] = ml_df['cation_type'].map({'c': 0, 'k': 1})
    if 'anion_type' in ml_df.columns:
        ml_df['anion_type_numeric'] = ml_df['anion_type'].map({'c': 0, 'k': 1})
    
    # Convert electrolyte_type to numeric (1-1, 1-2, 2-1, 2-2, 3-1, 4-1)
    if 'electrolyte_type' in ml_df.columns:
        electrolyte_type_map = {
            '1-1': 11, '1-2': 12, '2-1': 21, '2-2': 22,
            '3-1': 31, '4-1': 41
        }
        ml_df['electrolyte_type_numeric'] = ml_df['electrolyte_type'].map(electrolyte_type_map)
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_columns = ml_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if ml_df[col].isna().any():
            median_val = ml_df[col].median()
            ml_df[col] = ml_df[col].fillna(median_val)
            print(f"Filled {ml_df[col].isna().sum()} missing values in {col} with median: {median_val}")
    
    # For categorical columns that weren't mapped, keep as is
    
    # Reorganize columns: identifiers first, then features, then targets (pitzer coefficients)
    identifier_cols = ['electrolyte', 'electrolyte_type', 'cation', 'anion', 'molecule_formula']
    target_cols = [
        'B_MX_0_simplified', 'B_MX_1_simplified',
        'B_MX_0_original', 'B_MX_1_original', 'B_MX_2_original', 'C_MX_phi_original'
    ]
    
    # Get feature columns (everything that's not identifier or target)
    all_cols = ml_df.columns.tolist()
    feature_cols = [col for col in all_cols 
                   if col not in identifier_cols + target_cols]
    
    # Reorder columns
    ordered_cols = []
    for col in identifier_cols:
        if col in ml_df.columns:
            ordered_cols.append(col)
    
    ordered_cols.extend([col for col in feature_cols if col in ml_df.columns])
    
    for col in target_cols:
        if col in ml_df.columns:
            ordered_cols.append(col)
    
    ml_df = ml_df[ordered_cols]
    
    return ml_df


def generate_feature_summary(ml_df):
    """Generate a summary of features in the dataset."""
    
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    
    print(f"\nTotal rows: {len(ml_df)}")
    print(f"Total columns: {len(ml_df.columns)}")
    
    # Identify column types
    identifier_cols = ['electrolyte', 'electrolyte_type', 'cation', 'anion', 'molecule_formula']
    target_cols = [
        'B_MX_0_simplified', 'B_MX_1_simplified',
        'B_MX_0_original', 'B_MX_1_original', 'B_MX_2_original', 'C_MX_phi_original'
    ]
    
    numeric_cols = ml_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in target_cols]
    
    print(f"\nIdentifier columns: {len([c for c in identifier_cols if c in ml_df.columns])}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns: {len([c for c in target_cols if c in ml_df.columns])}")
    
    print("\nFeature categories:")
    
    # Categorize features
    ion_features = [col for col in feature_cols if 'anion_' in col or 'cation_' in col]
    molecule_features = [col for col in feature_cols if 'molecule_' in col]
    other_features = [col for col in feature_cols if col not in ion_features + molecule_features]
    
    print(f"  - Ion properties: {len(ion_features)}")
    print(f"  - Molecule properties: {len(molecule_features)}")
    print(f"  - Other features: {len(other_features)}")
    
    print("\nTarget variables (Pitzer coefficients):")
    for col in target_cols:
        if col in ml_df.columns:
            print(f"  - {col}: range [{ml_df[col].min():.4f}, {ml_df[col].max():.4f}]")
    
    print("\nData types:")
    print(ml_df.dtypes.value_counts())
    
    print("\nMissing values:")
    missing = ml_df.isna().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values!")
    
    return {
        'n_samples': len(ml_df),
        'n_features': len(feature_cols),
        'n_targets': len([c for c in target_cols if c in ml_df.columns])
    }


def main():
    """Main execution function."""
    
    print("="*80)
    print("BASELINE DATA + ION PROPERTIES MERGER")
    print("="*80)
    
    # Load and prepare data
    print("\n1. Loading data...")
    baseline_df, pitzer_ions_features = load_and_prepare_data()
    
    # Merge datasets
    print("\n2. Merging datasets...")
    merged_df = merge_datasets(baseline_df, pitzer_ions_features)
    
    # Prepare ML-ready dataset
    print("\n3. Preparing ML-ready dataset...")
    ml_df = prepare_ml_ready_dataset(merged_df)
    
    # Generate summary
    stats = generate_feature_summary(ml_df)
    
    # Save the merged dataset
    output_path = 'data/baseline_with_ion_properties.csv'
    ml_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"SUCCESS! ML-ready dataset saved to: {output_path}")
    print(f"{'='*80}")
    
    # Also save a feature-only version (excluding identifiers)
    identifier_cols = ['electrolyte', 'electrolyte_type', 'cation', 'anion', 'molecule_formula',
                      'cation_type', 'anion_type', 'ref_simplified', 'ref_original']
    
    # Create X (features) and y (targets) datasets
    target_cols = [
        'B_MX_0_simplified', 'B_MX_1_simplified',
        'B_MX_0_original', 'B_MX_1_original', 'B_MX_2_original', 'C_MX_phi_original'
    ]
    
    feature_cols = [col for col in ml_df.columns 
                   if col not in identifier_cols + target_cols]
    
    # Save feature matrix (X)
    X = ml_df[['electrolyte'] + feature_cols]
    X.to_csv('data/baseline_features_X.csv', index=False)
    print(f"\nFeature matrix saved to: data/baseline_features_X.csv")
    print(f"  Shape: {X.shape}")
    
    # Save target matrix (y)
    y_cols = ['electrolyte'] + [col for col in target_cols if col in ml_df.columns]
    y = ml_df[y_cols]
    y.to_csv('data/baseline_targets_y.csv', index=False)
    print(f"\nTarget matrix saved to: data/baseline_targets_y.csv")
    print(f"  Shape: {y.shape}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

