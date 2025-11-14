import json
import concurrent.futures as cf
import multiprocessing as mp
from pyEQL.solution import Solution
import pandas as pd

def extract_charge(formula):
    """Extract the integer charge from a formula string like 'Na[+1]'."""
    m = re.search(r'\[([+-]\d+)\]', formula)
    if m:
        return int(m.group(1))
    return 0

def run_solution(args):
    (cat, n_cat, an, n_an) = args
    sol = Solution([
        (cat, float(n_cat)),
        (an, float(n_an))
    ])
    gammas = sol.get_activity_coefficient()
    data.append({
                     "cation": cat,
                     "cat_count": n_cat,
                     "anion": an,
                     "an_count": n_an,
                     "gamma_cation": gammas.get(cat),
                     "gamma_anion": gammas.get(an),
                     "osmotic_coeff": sol.osmotic_coefficient(),
                     "water_activity": sol.water_activity(),
                 })

if __name__ == "__main__":

    # Load the database
    with open(r"C:\Users\gmjam\miniconda3\envs\gmjam\Lib\site-packages\pyEQL\database\pyeql_db.json", "r", encoding="utf-8") as f:
        db = json.load(f)



    # Separate cations and anions
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

    import re
    from itertools import product
    from math import gcd




    salts = []

    for cat, an in product(cations, anions):
        cat_charge = extract_charge(cat)
        an_charge = -extract_charge(an)

        if cat_charge == 0 or an_charge == 0:
            continue

        divisor = gcd(cat_charge, an_charge)
        cat_count = an_charge // divisor
        an_count = cat_charge // divisor

        # Save as tuple instead of string
        salts.append((cat, cat_count, an, an_count))


    print(f"Generated {len(salts)} possible salts.")
    print(salts[:20])

    import pandas as pd
    from pyEQL.solution import Solution



    # ---- Compute activity coefficients ----
    data = []
    time = 0



    for salt in salts:
        print(time)
        time += 1
        p = mp.Process(target=run_solution, args=(salt,))
        p.start()
        p.join(timeout=10)

        if p.is_alive():
            print("Skipped (timeout)")
            p.terminate()
            p.join()
            continue


    # with cf.ProcessPoolExecutor() as ex:
    #     for cat, n_cat, an, n_an in salts:
    #         print(time)
    #         time += 1
    #         future =ex.submit(Solution, [
    #                 (cat, float(n_cat)),
    #                 (an, float(n_an))
    #             ] )
    #         try:
    #             sol = future.result(timeout=10)
    #
    #             gammas = sol.activity_coefficients()
    #             data.append({
    #                 "cation": cat,
    #                 "cat_count": n_cat,
    #                 "anion": an,
    #                 "an_count": n_an,
    #                 "gamma_cation": gammas.get(cat),
    #                 "gamma_anion": gammas.get(an),
    #                 "osmotic_coeff": sol.osmotic_coefficient(),
    #                 "water_activity": sol.water_activity(),
    #             })
    #         except cf.TimeoutError:
    #             print("Skipped")
    #             continue




    df = pd.DataFrame(data)
    df.to_csv("ionic_salt_properties.csv", index=False)
    print("Wrote", len(df), "salts with valid activity coefficients.")
