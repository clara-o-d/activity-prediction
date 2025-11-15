import json
import multiprocessing as mp
import re
from itertools import product
from math import gcd

import pandas as pd
from pyEQL.solution import Solution


def extract_charge(formula: str) -> int:
    """Extract the integer charge from a formula string like 'Na[+1]'."""
    m = re.search(r'\[([+-]\d+)\]', formula)
    if m:
        return int(m.group(1))
    return 0


# Use a low overall concentration but keep stoichiometric ratios
CONC_FACTOR = 0.001

def run_solution(args, out_q):
    """
    Worker function run in a separate process.

    args: (cat, n_cat, an, n_an)
    out_q: multiprocessing.Queue to send result back to parent.

    Puts either a result dict or None on out_q.
    """
    cat, n_cat, an, n_an = args

    try:
        # Use stoichiometry for output, but scaled molalities for the model
        m_cat = float(n_cat) * CONC_FACTOR
        m_an = float(n_an) * CONC_FACTOR

        sol = Solution([
            (cat, m_cat),
            (an, m_an),
        ])

        # Activity coefficients for each ion
        gamma_cation = sol.get_activity_coefficient(cat).magnitude
        gamma_anion = sol.get_activity_coefficient(an).magnitude

        # Bulk properties
        osmotic_coeff = sol.get_osmotic_coefficient().magnitude
        water_activity = sol.get_water_activity().magnitude

        activity_coefficient = ((gamma_cation ** n_cat) * (gamma_anion ** n_an))**(1/(n_cat + n_an))

        result = {
            "cation": cat,
            "cat_count": n_cat,   # original stoichiometric count
            "anion": an,
            "an_count": n_an,     # original stoichiometric count
            "gamma_cation": gamma_cation,
            "gamma_anion": gamma_anion,
            "activity_coefficient": activity_coefficient,
            "osmotic_coeff": osmotic_coeff,
            "water_activity": water_activity
        }
        out_q.put(result)
    except Exception as e:
        out_q.put(None)


if __name__ == "__main__":
    with open("pyeql_db.json", "r", encoding="utf-8") as f:
        db = json.load(f)

    cations = []
    anions = []

    for entry in db:
        charge = entry.get("charge")
        formula = entry.get("formula")
        if charge is None or formula is None:
            continue
        if charge > 0:
            cations.append(formula)
        elif charge < 0:
            anions.append(formula)

    print("Number of cations:", len(cations))
    print("Number of anions:", len(anions))

    salts = []

    for cat, an in product(cations, anions):
        cat_charge = extract_charge(cat)
        # anion charge is stored as negative; flip sign to get magnitude
        an_charge = -extract_charge(an)

        if cat_charge == 0 or an_charge == 0:
            continue

        divisor = gcd(cat_charge, an_charge)
        cat_count = an_charge // divisor
        an_count = cat_charge // divisor

        salts.append((cat, cat_count, an, an_count))

    print(f"Generated {len(salts)} possible salts.")
    print(salts[:10])

    data = []
    timeout_seconds = 6

    for idx, salt in enumerate(salts):
        print(idx)

        out_q = mp.Queue()
        p = mp.Process(target=run_solution, args=(salt, out_q))
        p.start()
        p.join(timeout_seconds)

        if p.is_alive():
            cat, _, an, _ = salt
            print(f"Skipped (timeout > {timeout_seconds}s) for {cat}, {an}")
            p.terminate()
            p.join()
            continue

        if not out_q.empty():
            result = out_q.get()
            if result is not None:
                data.append(result)

        p.join()

    df = pd.DataFrame(data)
    df.to_csv("ionic_salt_properties.csv", index=False)
    print("Wrote", len(df), "salts with valid activity coefficients.")
