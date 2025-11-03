"""
Process Activity Database to create clean dataset for machine learning.

This script:
1. Reads electrolyte properties from 'Activity Database.xlsx - Electrolyte Properties.csv'
2. Reads Pitzer coefficients from 'Activity Database.xlsx - Electrolyte Activity.csv'
3. Filters electrolytes with complete molecular, cation, and anion data
4. Merges properties (X) with Pitzer coefficients (y)
5. Outputs clean dataset to 'clean_activity_dataset.csv'
"""

import pandas as pd
import numpy as np
import re


def read_properties_data(filepath):
    """
    Read and process electrolyte properties file.
    
    Returns:
        DataFrame with complete property data for each electrolyte
    """
    # Read the properties file - skip first row (section headers) and use second row as column names
    df = pd.read_csv(filepath, skiprows=1)
    
    # Rename columns to avoid duplicates by prefixing with section
    # First 8 columns are molecular properties, next ~20 are cation, rest are anion
    new_columns = ['Electrolyte_Name']
    
    # Molecular properties (columns 1-7 after Name)
    molecular_count = 0
    for i in range(1, len(df.columns)):
        col = df.columns[i]
        if molecular_count < 7:  # First 7 columns after name are molecular
            new_columns.append(f"Molecular_{col}")
            molecular_count += 1
        elif molecular_count < 27:  # Next 20 are cation
            if molecular_count == 7:
                new_columns.append(f"Cation_{col}")
            else:
                new_columns.append(f"Cation_{col}")
            molecular_count += 1
        else:  # Rest are anion
            new_columns.append(f"Anion_{col}")
            molecular_count += 1
    
    # Apply new column names
    df.columns = new_columns
    
    # Clean up electrolyte names (remove quotes and extra spaces)
    df['Electrolyte_Name'] = df['Electrolyte_Name'].astype(str).str.strip().str.replace("'", "").str.strip()
    
    # Remove rows with empty or invalid names
    df = df[df['Electrolyte_Name'].notna() & (df['Electrolyte_Name'] != '') & (df['Electrolyte_Name'] != 'nan')]
    
    # Rename back to 'Name' for compatibility
    df.rename(columns={'Electrolyte_Name': 'Name'}, inplace=True)
    
    return df


def check_completeness(row, molecular_cols, cation_cols, anion_cols):
    """
    Check if a row has sufficient data in molecular, cation, and anion columns.
    
    Returns:
        True if the row has enough non-null values in each category
    """
    # Count non-null values in each category
    molecular_filled = row[molecular_cols].notna().sum()
    cation_filled = row[cation_cols].notna().sum()
    anion_filled = row[anion_cols].notna().sum()
    
    # Require at least some data in each category
    # Lowered thresholds to include more electrolytes
    molecular_threshold = len(molecular_cols) * 0.15  # At least 15% filled
    cation_threshold = len(cation_cols) * 0.15
    anion_threshold = len(anion_cols) * 0.15
    
    return (molecular_filled >= molecular_threshold and 
            cation_filled >= cation_threshold and 
            anion_filled >= anion_threshold)


def filter_complete_electrolytes(df):
    """
    Filter electrolytes with sufficient molecular, cation, and anion data.
    """
    # Identify column groups based on prefixes
    molecular_cols = [col for col in df.columns if col.startswith('Molecular_')]
    cation_cols = [col for col in df.columns if col.startswith('Cation_')]
    anion_cols = [col for col in df.columns if col.startswith('Anion_')]
    
    print(f"Identified {len(molecular_cols)} molecular columns")
    print(f"Identified {len(cation_cols)} cation columns")
    print(f"Identified {len(anion_cols)} anion columns")
    
    # Filter rows with complete data
    complete_mask = df.apply(
        lambda row: check_completeness(row, molecular_cols, cation_cols, anion_cols),
        axis=1
    )
    
    filtered_df = df[complete_mask].copy()
    
    print(f"\nFiltered from {len(df)} to {len(filtered_df)} electrolytes with complete data")
    print(f"Electrolytes with complete data: {filtered_df['Name'].tolist()[:20]}")  # Show first 20
    
    return filtered_df


def read_pitzer_coefficients(filepath):
    """
    Read and extract Pitzer coefficients from the activity file.
    Handles two formats:
    1. Name line followed by "m (mol/kg)" header with coefficients
    2. Name line followed directly by data rows with coefficients
    
    Returns:
        DataFrame with electrolyte names and their Pitzer coefficients
    """
    # Read the file as plain text
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    pitzer_data = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        parts = [p.strip() for p in line.split(',')]
        
        # Look for electrolyte name line: first column starts with a letter
        if len(parts) >= 1 and parts[0] and parts[0][0].isalpha() and parts[0] not in ['m (mol/kg)', 'm']:
            electrolyte_name = parts[0].strip()
            
            # Pad current line parts to 8 columns
            while len(parts) < 8:
                parts.append('')
            
            # Helper function to extract Pitzer coefficients
            def extract_pitzer(parts_list):
                has_pitzer = False
                beta0 = np.nan
                beta1 = np.nan
                beta2 = np.nan
                cmx = np.nan
                
                if parts_list[4]:
                    try:
                        beta0 = float(parts_list[4])
                        has_pitzer = True
                    except ValueError:
                        pass
                if parts_list[5]:
                    try:
                        beta1 = float(parts_list[5])
                        has_pitzer = True
                    except ValueError:
                        pass
                if parts_list[6]:
                    try:
                        beta2 = float(parts_list[6])
                        has_pitzer = True
                    except ValueError:
                        pass
                if parts_list[7]:
                    try:
                        cmx = float(parts_list[7])
                        has_pitzer = True
                    except ValueError:
                        pass
                
                return has_pitzer, beta0, beta1, beta2, cmx
            
            # Pattern 1: Check current line for Pitzer coefficients
            has_pitzer, beta0, beta1, beta2, cmx = extract_pitzer(parts)
            
            # Pattern 2: Check next line if current line doesn't have coefficients
            if not has_pitzer and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                next_parts = [p.strip() for p in next_line.split(',')]
                
                # Ensure we have at least 8 columns
                while len(next_parts) < 8:
                    next_parts.append('')
                
                has_pitzer, beta0, beta1, beta2, cmx = extract_pitzer(next_parts)
            
            # Add if at least one coefficient is present
            if has_pitzer:
                pitzer_data.append({
                    'Name': electrolyte_name,
                    'Beta(0)': beta0,
                    'Beta(1)': beta1,
                    'Beta(2)': beta2,
                    'C(MX)': cmx
                })
        
        i += 1
    
    pitzer_df = pd.DataFrame(pitzer_data)
    
    # Clean up names
    pitzer_df['Name'] = pitzer_df['Name'].str.strip().str.replace("'", "").str.strip()
    
    # Remove duplicates (keep first occurrence)
    pitzer_df = pitzer_df.drop_duplicates(subset='Name', keep='first')
    
    print(f"\nExtracted Pitzer coefficients for {len(pitzer_df)} electrolytes")
    print(f"Electrolytes with at least one Pitzer coefficient: {pitzer_df[['Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']].notna().any(axis=1).sum()}")
    
    return pitzer_df


def create_name_mappings():
    """
    Create mapping between different naming conventions.
    Maps activity file names to property file names.
    """
    mappings = {
        'HCl': 'hydrogen chloride',
        'HBr': 'hydrobromic acid',
        'HI': 'hydroiodic acid',
        'HClO4': 'perchloric acid',
        'HNO3': 'nitric acid',
        'LiOH': 'lithium hydroxide',
        'LiCl': 'lithium chloride',
        'LiBr': 'lithium bromide',
        'LiI': 'lithium iodide',
        'LiClO4': 'lithium perchlorate',
        'LiNO3': 'lithium nitrate',
        'LiCH3COO': 'lithium acetate',
        'Silver Nitrate (AgNO3)': 'AgNO3',
    }
    return mappings


def merge_datasets(properties_df, pitzer_df):
    """
    Merge properties (X) with Pitzer coefficients (y).
    Uses name matching with mappings and formula matching.
    """
    # Rename Pitzer coefficient columns to avoid conflicts
    pitzer_renamed = pitzer_df.rename(columns={'Name': 'Pitzer_Name'})
    
    all_matches = []
    matched_property_indices = set()
    
    # Strategy 1: Direct name matching
    for idx, prop_row in properties_df.iterrows():
        prop_name = prop_row['Name']
        # Check if this name exists in pitzer data
        pitzer_match = pitzer_df[pitzer_df['Name'] == prop_name]
        if not pitzer_match.empty and idx not in matched_property_indices:
            result = prop_row.copy()
            for col in ['Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']:
                result[col] = pitzer_match.iloc[0][col]
            all_matches.append(result)
            matched_property_indices.add(idx)
    
    print(f"\nDirect name matches: {len(all_matches)}")
    
    # Strategy 2: Name mapping
    name_mappings = create_name_mappings()
    for idx, prop_row in properties_df.iterrows():
        if idx in matched_property_indices:
            continue
        prop_name = prop_row['Name']
        # Check each pitzer name
        for pitzer_name in pitzer_df['Name']:
            mapped_name = name_mappings.get(pitzer_name, pitzer_name)
            if prop_name == mapped_name:
                pitzer_match = pitzer_df[pitzer_df['Name'] == pitzer_name]
                result = prop_row.copy()
                for col in ['Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']:
                    result[col] = pitzer_match.iloc[0][col]
                all_matches.append(result)
                matched_property_indices.add(idx)
                break
    
    print(f"After name mapping: {len(all_matches)} matches")
    
    # Strategy 3: Formula matching
    formula_col = 'Molecular_Formula' if 'Molecular_Formula' in properties_df.columns else None
    if formula_col:
        for idx, prop_row in properties_df.iterrows():
            if idx in matched_property_indices:
                continue
            prop_formula = str(prop_row[formula_col]).strip()
            if prop_formula and prop_formula != 'nan':
                for _, pitzer_row in pitzer_df.iterrows():
                    pitzer_name = pitzer_row['Name']
                    # Extract formula from parentheses
                    import re
                    match = re.search(r'\(([^)]+)\)', pitzer_name)
                    pitzer_formula = match.group(1) if match else pitzer_name
                    
                    if prop_formula == pitzer_formula:
                        result = prop_row.copy()
                        for col in ['Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']:
                            result[col] = pitzer_row[col]
                        all_matches.append(result)
                        matched_property_indices.add(idx)
                        break
    
    print(f"After formula matching: {len(all_matches)} matches")
    
    if not all_matches:
        # Return empty dataframe with correct structure
        result_df = properties_df.iloc[:0].copy()
        for col in ['Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']:
            result_df[col] = pd.Series(dtype='float64')
        return result_df
    
    # Create final dataframe
    merged_df = pd.DataFrame(all_matches)
    
    print(f"\nTotal merged dataset contains {len(merged_df)} electrolytes")
    
    # Ensure Pitzer coefficient columns exist
    for col in ['Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']:
        if col not in merged_df.columns:
            merged_df[col] = np.nan
    
    # Move Name to first column and Pitzer coefficients to last columns
    cols = ['Name'] + [col for col in merged_df.columns if col not in ['Name', 'Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']]
    cols = cols + ['Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']
    merged_df = merged_df[cols]
    
    return merged_df


def main():
    """Main processing pipeline."""
    
    print("=" * 70)
    print("Activity Database Processing Pipeline")
    print("=" * 70)
    
    # File paths
    properties_file = 'Activity Database.xlsx - Electrolyte Properties.csv'
    activity_file = 'Activity Database.xlsx - Electrolyte Activity.csv'
    output_file = 'clean_activity_dataset.csv'
    
    # Step 1: Read and filter properties data
    print("\n[Step 1] Reading electrolyte properties...")
    properties_df = read_properties_data(properties_file)
    
    print("\n[Step 2] Filtering electrolytes with complete data...")
    complete_properties_df = filter_complete_electrolytes(properties_df)
    
    # Step 2: Read Pitzer coefficients
    print("\n[Step 3] Extracting Pitzer coefficients...")
    pitzer_df = read_pitzer_coefficients(activity_file)
    
    # Step 3: Merge datasets
    print("\n[Step 4] Merging datasets...")
    final_df = merge_datasets(complete_properties_df, pitzer_df)
    
    # Step 4: Save to CSV
    print("\n[Step 5] Saving clean dataset...")
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Clean dataset saved to '{output_file}'")
    print(f"  - Total electrolytes: {len(final_df)}")
    print(f"  - Total features: {len(final_df.columns)}")
    print(f"  - X features (properties): {len(final_df.columns) - 5}")  # -5 for Name + 4 Pitzer coeffs
    print(f"  - Y features (Pitzer coefficients): 4")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"\nElectrolytes included:\n{final_df['Name'].tolist()}")
    
    print("\nPitzer coefficients coverage:")
    for col in ['Beta(0)', 'Beta(1)', 'Beta(2)', 'C(MX)']:
        non_null = final_df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(final_df)} ({100*non_null/len(final_df):.1f}%)")
    
    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

