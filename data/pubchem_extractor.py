import re
import requests
import json
import time
import pandas as pd

# --------------------------------------------------
# Normalization Logic
# --------------------------------------------------

def move_charge_after_numeral(name):
    """Move formal charge sign after the numeral (e.g., Ac+3 → Ac3+)."""
    # Pattern matches: +1, +2, +3, -1, -2, -3, etc. at the end of string
    # Or after a closing parenthesis or letter
    pattern = r'([+\-])(\d+)$'
    match = re.search(pattern, name)
    if match:
        sign = match.group(1)  # + or -
        num = match.group(2)    # the number
        # Replace +num or -num with num+ or num-
        name = re.sub(pattern, f'{num}{sign}', name)
    return name


# --------------------------------------------------
# Read and Process Ions
# --------------------------------------------------

# Read electrolytes from cleaned_ions.csv
df_ions = pd.read_csv("cleaned_ions.csv")
# Filter out empty rows and get all normalized electrolytes
df_ions = df_ions[df_ions["normalized"].notna() & (df_ions["normalized"] != "")]
# Remove leading dash and space from normalized names, then move charge after numeral
IONS = df_ions["normalized"].str.replace(r"^-\s+", "", regex=True).apply(move_charge_after_numeral).tolist()
# Keep original names for reference
IONS_ORIGINAL = df_ions["original"].tolist()


def normalize_ion(name):
    """Convert bracket-charge syntax into PubChem-compatible form."""
    
    # Remove (aq)
    name = name.replace("(aq)", "")
    
    # e.g. X[+3] → X3+, X[-2] → X2-
    m = re.search(r"\[([+-]\d+)\]", name)
    if m:
        charge = m.group(1)  # +3 or -2
        abs_charge = charge.replace("+", "").replace("-", "")
        sign = "+" if charge.startswith("+") else "-"
        
        # Replace the [±n] with proper charge notation
        name = re.sub(r"\[[+-]\d+\]", f"{abs_charge}{sign}", name)
        
        # Example: Ag becomes Ag1+, but PubChem prefers Ag+
        name = name.replace("1+", "+").replace("1-", "-")
    
    # Replace double minus `--` with `(2-)` format PubChem supports
    name = re.sub(r"(\d+)\-\-", r"(\1-)", name)

    return name


# --------------------------------------------------
# PubChem API
# --------------------------------------------------

def get_cid(query):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(query)}/cids/JSON"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    try:
        return r.json()["IdentifierList"]["CID"][0]
    except:
        return None


def get_compound_data(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def extract_numeric_values(data_tree):
    numeric = {}
    def recurse(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                recurse(v, f"{path}[{i}]")
        else:
            if isinstance(obj, (int, float)):
                numeric[path] = obj
            if isinstance(obj, str):
                try:
                    val = float(obj)
                    numeric[path] = val
                except:
                    pass
    recurse(data_tree)
    return numeric


# --------------------------------------------------
# Main Loop
# --------------------------------------------------

results = []

for i, normalized in enumerate(IONS):
    original = IONS_ORIGINAL[i] if i < len(IONS_ORIGINAL) else normalized
    print(f"\nProcessing: {original}")
    print(f" → Searching for: {normalized} (already normalized)")

    cid = get_cid(normalized)
    
    if cid is None:
        print("   No CID found")
        results.append({"original": original, "normalized": normalized, "cid": None})
        continue
    
    print(f"   CID = {cid}")

    data = get_compound_data(cid)
    numeric = extract_numeric_values(data)

    record = {"original": original, "normalized": normalized, "cid": cid}
    record.update(numeric)
    results.append(record)

    time.sleep(0.2)


# --------------------------------------------------
# Save Output
# --------------------------------------------------

with open("ions_data.json", "w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results)
df.to_csv("ions_data.csv", index=False)

print("\nDone.")
