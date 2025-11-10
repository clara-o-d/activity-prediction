"""
Prepare ML-Ready Dataset from Clean Activity Data

This script:
1. Removes non-numeric columns
2. Removes columns with sparse data (< 70% filled)
3. Renames columns to be concise and ML-friendly
4. Handles missing values
5. Outputs feature matrix X and target matrix y
"""

import pandas as pd
import numpy as np


def analyze_data_quality(df):
    """Analyze data completeness for each column."""
    completeness = {}
    for col in df.columns:
        if col == 'Name':
            continue
        non_null = df[col].notna().sum()
        total = len(df)
        completeness[col] = (non_null / total) * 100
    
    return completeness


def remove_sparse_columns(df, threshold=70):
    """
    Remove columns with less than threshold% data filled.
    
    Args:
        df: DataFrame
        threshold: Minimum percentage of non-null values (default 70%)
    
    Returns:
        DataFrame with sparse columns removed, list of removed columns
    """
    completeness = analyze_data_quality(df)
    
    cols_to_remove = []
    cols_to_keep = ['Name']  # Always keep the name column
    
    for col, pct in completeness.items():
        if pct < threshold:
            cols_to_remove.append(col)
        else:
            cols_to_keep.append(col)
    
    df_filtered = df[cols_to_keep]
    
    return df_filtered, cols_to_remove


def convert_to_numeric(df):
    """
    Convert all columns (except Name) to numeric, coercing errors to NaN.
    
    Returns:
        DataFrame with numeric columns, list of columns that had non-numeric values
    """
    non_numeric_found = []
    
    for col in df.columns:
        if col == 'Name':
            continue
        
        # Check if column has any non-numeric values
        original_nulls = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        new_nulls = df[col].isna().sum()
        
        if new_nulls > original_nulls:
            non_numeric_found.append(col)
    
    return df, non_numeric_found


def create_concise_names(columns):
    """
    Create concise, ML-friendly column names.
    
    Rules:
    - Remove prefixes (Molecular_, Cation_, Anion_)
    - Shorten common terms
    - Use snake_case
    - Keep it under 20 characters when possible
    """
    name_mapping = {}
    
    for col in columns:
        if col == 'Name':
            name_mapping[col] = 'electrolyte_name'
            continue
        
        # Start with original name
        new_name = col
        
        # Remove prefixes but add abbreviation
        if col.startswith('Molecular_'):
            new_name = 'mol_' + col.replace('Molecular_', '')
        elif col.startswith('Cation_'):
            new_name = 'cat_' + col.replace('Cation_', '')
        elif col.startswith('Anion_'):
            new_name = 'an_' + col.replace('Anion_', '')
        
        # Shorten common terms
        replacements = {
            'Molar mass (g/mol)': 'molar_mass',
            'Molar mass': 'molar_mass',
            'SMILES representation': 'smiles',
            'Formula': 'formula',
            'Organic? (1 if organic, 0 if not)': 'is_organic',
            'Acid? (1 if acid, 0 if not)': 'is_acid',
            '# Ionic species': 'ionic_species',
            'Saltwater solubility limit (g/100mL water, 20 C)': 'solubility',
            'Charge (formal)': 'charge',
            '# Valence electrons': 'valence_e',
            'Hydrated radius (nm)': 'hydrated_r',
            'Ionic radius (pm) (molecular radius if polyatomic)': 'ionic_r',
            'Polarizability (au)': 'polarizability',
            'Hydration number': 'hydration_num',
            'Hydration enthalpy (kJ/mol)': 'hydration_H',
            'HSAB type (1 if hard, 0 if soft)': 'hsab',
            'HSAB type (1 if hard, 0 if soft, 0.5 if \'borderline\')': 'hsab',
            'Atomic number': 'atomic_num',
            'Periodic table group': 'pt_group',
            'Periodic table period': 'pt_period',
            'Polyatomic (1 if so, 0 if not)': 'is_polyatomic',
            'Dipole moment (if polyatomic)': 'dipole',
            'Dipole moment (D, if polyatomic)': 'dipole',
            'Electronegativity': 'electroneg',
            'Electrion affinity (kJ/mol)': 'electron_aff',
            'Ionization energy (eV)': 'ionization_E',
            '# Valence electrons (neutral)': 'valence_e',
            'Beta(0)': 'beta_0',
            'Beta(1)': 'beta_1',
            'Beta(2)': 'beta_2',
            'C(MX)': 'c_mx',
        }
        
        for old, new in replacements.items():
            if old in new_name:
                new_name = new_name.replace(old, new)
        
        # Handle .1, .2 suffixes
        if '.1' in new_name:
            new_name = new_name.replace('.1', '')
        if '.2' in new_name:
            new_name = new_name.replace('.2', '_2')
        
        # Convert to snake_case and clean up
        new_name = new_name.lower()
        new_name = new_name.replace(' ', '_')
        new_name = new_name.replace('(', '').replace(')', '')
        new_name = new_name.replace(',', '')
        new_name = new_name.replace('/', '_')
        new_name = new_name.replace('-', '_')
        new_name = new_name.replace('__', '_')
        new_name = new_name.strip('_')
        
        name_mapping[col] = new_name
    
    return name_mapping


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        df: DataFrame
        strategy: 'mean', 'median', 'drop', or 'zero'
    
    Returns:
        DataFrame with missing values handled
    """
    df_processed = df.copy()
    
    if strategy == 'drop':
        # Drop rows with any missing values
        df_processed = df_processed.dropna()
    elif strategy == 'zero':
        # Fill with zeros
        df_processed = df_processed.fillna(0)
    elif strategy == 'mean':
        # Fill with column mean (only numeric columns)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    elif strategy == 'median':
        # Fill with column median
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed


def prepare_ml_data(input_file='clean_activity_dataset.csv', 
                    output_file='ml_ready_dataset.csv',
                    completeness_threshold=70,
                    missing_value_strategy='mean'):
    """
    Main function to prepare ML-ready dataset.
    
    Args:
        input_file: Path to clean activity dataset
        output_file: Path to save ML-ready dataset
        completeness_threshold: Minimum % of data required per column
        missing_value_strategy: How to handle missing values
    """
    
    print("=" * 80)
    print("ML Data Preparation Pipeline")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[Step 1] Loading clean dataset...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} electrolytes with {len(df.columns)} columns")
    
    # Step 2: Analyze data quality
    print("\n[Step 2] Analyzing data quality...")
    completeness = analyze_data_quality(df)
    sparse_cols = [col for col, pct in completeness.items() if pct < completeness_threshold]
    print(f"Found {len(sparse_cols)} columns with < {completeness_threshold}% data")
    
    # Step 3: Remove sparse columns
    print(f"\n[Step 3] Removing sparse columns (< {completeness_threshold}% filled)...")
    df_filtered, removed_cols = remove_sparse_columns(df, completeness_threshold)
    print(f"Removed {len(removed_cols)} sparse columns")
    print(f"Remaining: {len(df_filtered.columns) - 1} feature columns + Name")
    
    # Step 4: Convert to numeric
    print("\n[Step 4] Converting to numeric and removing non-numeric columns...")
    df_numeric, non_numeric_cols = convert_to_numeric(df_filtered)
    
    # Remove non-numeric columns (keep Name for now)
    non_numeric_to_remove = [col for col in non_numeric_cols if col not in ['Name']]
    if non_numeric_to_remove:
        print(f"Removing {len(non_numeric_to_remove)} non-numeric columns:")
        for col in non_numeric_to_remove[:5]:
            print(f"  - {col}")
        if len(non_numeric_to_remove) > 5:
            print(f"  ... and {len(non_numeric_to_remove) - 5} more")
        df_numeric = df_numeric.drop(columns=non_numeric_to_remove)
    
    # Step 5: Rename columns
    print("\n[Step 5] Creating concise column names...")
    name_mapping = create_concise_names(df_numeric.columns)
    df_renamed = df_numeric.rename(columns=name_mapping)
    
    # Step 6: Handle missing values
    print(f"\n[Step 6] Handling missing values (strategy: {missing_value_strategy})...")
    missing_before = df_renamed.isna().sum().sum()
    df_final = handle_missing_values(df_renamed, strategy=missing_value_strategy)
    missing_after = df_final.isna().sum().sum()
    print(f"Missing values: {missing_before} -> {missing_after}")
    
    # Step 7: Remove beta_2 (all zeros) and separate features and targets
    print("\n[Step 7] Removing beta_2 and separating features (X) and targets (y)...")
    
    # Drop beta_2 and c_mx columns
    cols_to_drop = []
    if 'beta_2' in df_final.columns:
        cols_to_drop.append('beta_2')
    if 'c_mx' in df_final.columns:
        cols_to_drop.append('c_mx')
    
    if cols_to_drop:
        df_final = df_final.drop(columns=cols_to_drop)
        print(f"Removed columns: {', '.join(cols_to_drop)} (beta_2 is all zeros, c_mx has poor prediction)")
    
    # Identify target columns (Pitzer coefficients)
    # Only using beta_0 and beta_1
    target_cols = [col for col in df_final.columns if any(x in col for x in ['beta_0', 'beta_1'])]
    feature_cols = [col for col in df_final.columns if col not in target_cols + ['electrolyte_name']]
    
    print(f"Features (X): {len(feature_cols)} columns")
    print(f"Targets (y): {len(target_cols)} columns")
    
    # Step 8: Save datasets
    print("\n[Step 8] Saving ML-ready dataset...")
    df_final.to_csv(output_file, index=False)
    print(f"✓ Saved to '{output_file}'")
    
    # Also save separate X and y files
    import os
    base_dir = os.path.dirname(output_file)
    base_name = os.path.basename(output_file).replace('.csv', '')
    X_file = os.path.join(base_dir, f'{base_name}_X.csv')
    y_file = os.path.join(base_dir, f'{base_name}_y.csv')
    
    df_final[['electrolyte_name'] + feature_cols].to_csv(X_file, index=False)
    df_final[['electrolyte_name'] + target_cols].to_csv(y_file, index=False)
    
    print(f"✓ Saved features to '{X_file}'")
    print(f"✓ Saved targets to '{y_file}'")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Final dataset: {len(df_final)} electrolytes × {len(df_final.columns)} columns")
    print(f"  - Electrolyte names: 1 column")
    print(f"  - Features (X): {len(feature_cols)} columns")
    print(f"  - Targets (y): {len(target_cols)} columns")
    print(f"\nData completeness: {100 - (df_final.isna().sum().sum() / (df_final.shape[0] * df_final.shape[1]) * 100):.1f}%")
    
    # Show feature categories
    mol_features = [c for c in feature_cols if c.startswith('mol_')]
    cat_features = [c for c in feature_cols if c.startswith('cat_')]
    an_features = [c for c in feature_cols if c.startswith('an_')]
    
    print(f"\nFeature breakdown:")
    print(f"  - Molecular: {len(mol_features)} features")
    print(f"  - Cation: {len(cat_features)} features")
    print(f"  - Anion: {len(an_features)} features")
    
    print("\n" + "=" * 80)
    print("ML-Ready Dataset Complete!")
    print("=" * 80)
    
    return df_final, feature_cols, target_cols


if __name__ == "__main__":
    # Set up paths relative to project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    # Run with default parameters
    df, X_cols, y_cols = prepare_ml_data(
        input_file=os.path.join(data_dir, 'clean_activity_dataset.csv'),
        output_file=os.path.join(data_dir, 'ml_ready_dataset.csv'),
        completeness_threshold=70,
        missing_value_strategy='mean'
    )
    
    print("\nYou can now use these files for machine learning:")
    print("  - ml_ready_dataset.csv (complete dataset)")
    print("  - ml_ready_dataset_X.csv (features only)")
    print("  - ml_ready_dataset_y.csv (targets only)")

