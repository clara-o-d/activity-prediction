# Activity Database Data Processing

## Overview
This directory contains scripts to process activity database CSV files for electrolyte activity prediction using machine learning.

## Files Generated

### Input Files
- `Activity Database.xlsx - Electrolyte Properties.csv`: Contains molecular, cation, and anion properties for electrolytes
- `Activity Database.xlsx - Electrolyte Activity.csv`: Contains Pitzer coefficients (Beta(0), Beta(1), Beta(2), C(MX)) for electrolytes

### Processing Script
- `process_activity_data.py`: Main Python script that processes the data

### Output Files
- `clean_activity_dataset.csv`: Clean dataset with X features (properties) and Y features (Pitzer coefficients)

## Dataset Structure

### X Features (Properties)
The properties are organized into three categories:

#### Molecular Properties (7 features)
- Formula
- SMILES representation
- Molar mass (g/mol)
- Organic? (1 if organic, 0 if not)
- Acid? (1 if acid, 0 if not)
- Number of ionic species
- Saltwater solubility limit (g/100mL water, 20°C)

#### Cation Properties (20 features)
- Name
- Molar mass (g/mol)
- Charge (formal)
- Number of valence electrons
- Hydrated radius (nm)
- Ionic radius (pm)
- Polarizability (au)
- Hydration number
- Hydration enthalpy (kJ/mol)
- HSAB type (1 if hard, 0 if soft)
- Atomic number
- Periodic table group
- Periodic table period
- Polyatomic (1 if so, 0 if not)
- Dipole moment (if polyatomic)
- Electronegativity
- Electron affinity (kJ/mol)
- Ionization energy (eV)

#### Anion Properties (16 features)
Similar structure to cation properties with appropriate anion characteristics

### Y Features (Pitzer Coefficients)
- **Beta(0)**: First Pitzer parameter
- **Beta(1)**: Second Pitzer parameter
- **Beta(2)**: Third Pitzer parameter
- **C(MX)**: Pitzer mixing parameter

## Data Filtering

The script filters electrolytes to include only those with sufficient data:
- At least 30% of molecular properties filled
- At least 30% of cation properties filled
- At least 30% of anion properties filled

## Name Matching

The script uses three strategies to match electrolytes between the properties and activity files:
1. **Direct name matching**: Exact string matches
2. **Name mapping**: Maps common formula names to full names (e.g., HCl → hydrogen chloride)
3. **Formula matching**: Extracts formulas from parentheses in activity file names

## Usage

```bash
python process_activity_data.py
```

## Results Summary

From the final run:
- **Total electrolytes in properties file**: 212
- **Electrolytes with complete data**: 57 (meeting 15% completeness threshold in all three categories)
- **Electrolytes with Pitzer coefficients**: 95 extracted from activity file
- **Final matched electrolytes**: 30

### Electrolytes in Final Dataset
1. NaF
2. NaCl
3. NaBr
4. NaI
5. NaClO4
6. NaNO3
7. HClO4
8. CsNO2
9. KNO2
10. NaNO2
11. LiNO2
12. ZnCl2
13. CdClO4
14. NiCl2
15. CoCl2
16. Hydrogen chloride (HCl)
17. Hydrobromic acid (HBr)
18. Hydroiodic acid (HI)
19. Perchloric acid (HClO4)
20. Nitric acid (HNO3)
21. Lithium hydroxide (LiOH)
22. Lithium chloride (LiCl)
23. Lithium bromide (LiBr)
24. Lithium iodide (LiI)
25. Lithium perchlorate (LiClO4)
26. Lithium nitrate (LiNO3)
27. AgNO3
28. Sodium hydroxide (NaOH)
29. Sodium chlorate (NaClO3)
30. Potassium fluoride (KF)

All 30 electrolytes have complete Pitzer coefficients (100% coverage for all four parameters).

## Total Features
- **Total columns**: 48
- **X features (input)**: 43 properties (7 molecular + 20 cation + 16 anion)
- **Y features (target)**: 4 Pitzer coefficients
- **Identifier**: 1 (electrolyte name)

## Dependencies

```python
pandas>=2.0.0
numpy>=1.20.0
```

Install with:
```bash
pip install pandas numpy
```

## Future Improvements

To increase the dataset size, consider:
1. Relaxing completeness criteria for electrolytes with partial data
2. Imputing missing values using domain knowledge or ML techniques
3. Adding more electrolytes from additional data sources
4. Collecting Pitzer coefficients for more electrolytes in the properties database

