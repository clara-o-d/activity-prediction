import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import inspect

def calculate_mf_CaCl(RH, T):
    """
    This function calculates the mass fraction of Calcium Chloride as a
    function of the vapor partial pressure and temperature.
    Based on: https://doi.org/10.1016/j.ijthermalsci.2003.09.003

    Parameters:
    RH : float
        Relative humidity (0 < RH < 1)
    T : float
        Temperature in degrees Celsius (T <= 100)

    Returns:
    mf : float
        Mass fraction of CaCl2
    """

    if RH > 1 or RH <= 0:
        raise ValueError("The relative humidity should be 0 < RH < 1")
    if T > 100:
        raise ValueError("T too high. T should be given in °C and T <= 100")

    # Parameters for the vapor pressure equation
    q_0 = 0.31
    q_1 = 3.698
    q_2 = 0.60
    q_3 = 0.231
    q_4 = 4.584
    q_5 = 0.49
    q_6 = 0.478
    q_7 = -5.20
    q_8 = -0.40
    q_9 = 0.018

    theta = (T + 273.15) / 647  # Reduced temperature

    def f(xi):
        term1 = 1 - (1 + (xi / q_6)**q_7)**q_8 - q_9 * np.exp(-((xi - 0.1)**2) / 0.005)
        term2 = 2 - (1 + (xi / q_0)**q_1)**q_2 + ((1 + (xi / q_3)**q_4)**q_5 - 1) * theta
        return RH - term1 * term2

    mf_initial_guess = 0.1
    mf_solution = fsolve(f, mf_initial_guess)

    return float(mf_solution[0])

def calculate_mf_HCl(RH):
    """
    Calculates the mass fraction of HCl as a function of relative humidity at 25°C.
    Based on: https://doi.org/10.1063/1.3253108

    Parameters:
    RH : float
        Relative humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of HCl
    """

    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    # Polynomial coefficients
    A_4 = 0
    A_3 = 14.704
    A_2 = -10.183
    A_1 = -0.4859
    A_0 = 0.9965

    # Define the function to solve
    def f(X):
        return RH - (A_0 + A_1 * X + A_2 * X**2 + A_3 * X**3 + A_4 * X**4)

    mf_initial_guess = 0.3
    mf_solution = fsolve(f, mf_initial_guess)

    return float(mf_solution[0])

def calculate_mf_LiBr(RH):
    """
    Calculates the mass fraction of Lithium Bromide (LiBr)
    as a function of relative humidity at 25°C.
    Based on: https://doi.org/10.1002/er.1790

    Parameters:
    RH : float
        Relative humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of LiBr
    """
    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    # Polynomial coefficients
    A_4 = 18.838641746059196
    A_3 = -19.878321345922046
    A_2 = 3.871666208430489
    A_1 = -0.766050042576149
    A_0 = 1.004948395640807

    # Define the residual function for least squares
    def residual(X):
        return RH - (A_0 + A_1 * X + A_2 * X**2 + A_3 * X**3 + A_4 * X**4)

    # Initial guess and bounds
    initial_guess = [0.20]
    bounds = (0, 0.65)

    result = least_squares(residual, initial_guess, bounds=bounds, xtol=1e-6, ftol=1e-6)

    return float(result.x[0])

def calculate_mf_LiCl(RH, T):
    """
    Calculates the mass fraction of Lithium Chloride (LiCl)
    as a function of relative humidity and temperature (°C).
    Based on: https://doi.org/10.1016/j.ijthermalsci.2003.09.003

    Parameters:
    RH : float
        Relative humidity (0 < RH < 1)
    T : float
        Temperature in Celsius (T ≤ 100)

    Returns:
    mf : float
        Mass fraction of LiCl
    """
    if RH <= 0 or RH > 1:
        raise ValueError("The relative humidity should be 0 < RH < 1")
    if T > 100:
        raise ValueError("T too high. T should be given in °C and ≤ 100")

    # Parameters for the vapor pressure equation
    p_0 = 0.28
    p_1 = 4.3
    p_2 = 0.60
    p_3 = 0.21
    p_4 = 5.10
    p_5 = 0.49
    p_6 = 0.362
    p_7 = -4.75
    p_8 = -0.40
    p_9 = 0.03

    theta = (T + 273.15) / 647  # Reduced temperature

    def f(xi):
        term1 = 1 - (1 + (xi / p_6) ** p_7) ** p_8 - p_9 * np.exp(-((xi - 0.1) ** 2) / 0.005)
        term2 = 2 - (1 + (xi / p_0) ** p_1) ** p_2 + ((1 + (xi / p_3) ** p_4) ** p_5 - 1) * theta
        return RH - term1 * term2

    mf_initial_guess = 0.1
    mf_solution = fsolve(f, mf_initial_guess)

    return float(mf_solution[0])

def calculate_mf_LiI(RH):
    """
    Calculates the mass fraction of Lithium Iodide (LiI)
    as a function of relative humidity at 30°C.
    Based on: https://pubs.acs.org/doi/pdf/10.1021/je00060a020

    Parameters:
    RH : float
        Relative humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of LiI
    """
    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    # Polynomial coefficients
    A_5 = -1.121229305663804
    A_4 = -6.616780479521403
    A_3 = 4.951108742483354
    A_2 = -2.2094102170
    A_1 = -0.16556692518410
    A_0 = 1.000055934279976

    def f(xi):
        return RH - (A_0 + A_1 * xi + A_2 * xi**2 + A_3 * xi**3 + A_4 * xi**4 + A_5 * xi**5)

    mf_initial_guess = 0.1
    mf_solution = fsolve(f, mf_initial_guess)[0]

    if mf_solution > 0.62:
        raise ValueError("Below deliquescence relative humidity — solution not valid")

    return float(mf_solution)

def calculate_mf_MgCl(RH):
    """
    Calculates the mass fraction of Magnesium Chloride (MgCl2)
    as a function of relative humidity at 20–24°C.
    Based on: https://doi.org/10.1080/027868299304219

    Parameters:
    RH : float
        Relative humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of MgCl2
    """
    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    # Polynomial coefficients
    A_4 = 186.32487108
    A_3 = -153.67496570
    A_2 = 38.21982328
    A_1 = -4.86704441
    A_0 = 1.16231287

    def f(X):
        return RH - (A_0 + A_1 * X + A_2 * X**2 + A_3 * X**3 + A_4 * X**4)

    mf_initial_guess = 0.3
    mf_solution = fsolve(f, mf_initial_guess)[0]

    return float(mf_solution)

def calculate_mf_MgNO3(RH):
    """
    Calculates the mass fraction of Magnesium Nitrate (Mg(NO3)2)
    as a function of relative humidity at 20–24°C.
    Based on: https://doi.org/10.1080/027868299304219

    Parameters:
    RH : float
        Relative humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of Mg(NO3)2
    """
    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    # Polynomial coefficients
    A_4 = 0
    A_3 = 6.075040
    A_2 = -8.649495
    A_1 = 1.944451
    A_0 = 0.795876

    def f(X):
        return RH - (A_0 + A_1 * X + A_2 * X**2 + A_3 * X**3 + A_4 * X**4)

    mf_initial_guess = 0.3
    mf_solution = fsolve(f, mf_initial_guess)[0]

    return float(mf_solution)

def calculate_mf_NaOH(RH):
    """
    Calculates the mass fraction of Sodium Hydroxide (NaOH)
    as a function of relative humidity at 25°C.
    Based on: https://www.sciencedirect.com/science/article/pii/036031998590093X

    Parameters:
    RH : float
        Relative humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of NaOH
    """
    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    # Polynomial coefficients
    A_4 = 22.452177
    A_3 = -8.992202
    A_2 = -3.713377
    A_1 = -0.506960
    A_0 = 1.003610

    def f(X):
        return RH - (A_0 + A_1 * X + A_2 * X**2 + A_3 * X**3 + A_4 * X**4)

    # Use least_squares to replicate MATLAB's lsqnonlin with bounds
    result = least_squares(
        f,
        x0=0.30,
        bounds=(0, 0.65),
        xtol=1e-6,
        ftol=1e-6,
        verbose=0
    )
    mf_solution = result.x[0]
    return float(mf_solution)

def calculate_mf_ZnBr(RH):
    """
    Calculates the mass fraction of Zinc Bromide as a function
    of the Relative Humidity at 25°C.
    Based on isopiestic data from DOI: https://doi.org/10.1039/TF9403600733

    Parameters:
    RH : float
        Relative Humidity (0 < RH < 0.87)

    Returns:
    mf : float
        Mass fraction of Zinc Bromide
    """
    if RH <= 0 or RH > 0.87:
        raise ValueError("RH should be 0 < RH < 0.87")

    MM = 225.198  # molar mass of ZnBr2 [g/mol]

    A_4 = -2.441539790429647e-05
    A_3 = 0.001164035024612
    A_2 = -0.016402311078937
    A_1 = 0.015286715512812
    A_0 = 0.921783882286604

    def func(molality):
        return RH - (A_0 + A_1*molality + A_2*molality**2 + A_3*molality**3 + A_4*molality**4)

    molality_initial_guess = 0.5
    molality_solution, = fsolve(func, molality_initial_guess)

    mf = (molality_solution * MM) / (1000 + molality_solution * MM)

    return float(mf)

def calculate_mf_ZnCl(RH):
    """
    Calculates the mass fraction of Zinc Chloride as a function of
    the Relative Humidity at 60°C.
    Based on pressure data from: https://pubs.acs.org/doi/pdf/10.1021/ic50195a058

    Parameters:
    RH : float
        Relative Humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of Zinc Chloride
    """
    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    A_4 = 13.389423002312885
    A_3 = -19.289870089311886
    A_2 = 6.562516975212110
    A_1 = -0.984267594678104
    A_0 = 1.009182113918304

    def func(x):
        return RH - (A_0 + A_1*x + A_2*x**2 + A_3*x**3 + A_4*x**4)

    # Initial guess
    x0 = 0.4
    # Bounds [0, 0.8]
    bounds = (0, 0.8)

    result = least_squares(func, x0, bounds=bounds, ftol=1e-6, xtol=1e-6, verbose=0)

    mf = result.x[0]
    return float(mf)

def calculate_mf_ZnI(RH):
    """
    Calculates the mass fraction of Zinc Iodide as a function of the Relative Humidity at 25°C.
    Based on pressure data from: https://srd.nist.gov/jpcrdreprint/1.555639.pdf

    Parameters:
    RH : float
        Relative Humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of Zinc Iodide
    """
    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    A_4 = 7.750630
    A_3 = -24.365130
    A_2 = 22.380262
    A_1 = -8.657558
    A_0 = 2.104603

    def func(x):
        return RH - (A_0 + A_1*x + A_2*x**2 + A_3*x**3 + A_4*x**4)

    x0 = 0.24
    bounds = (0, 0.9)

    result = least_squares(func, x0, bounds=bounds, ftol=1e-6, xtol=1e-6)
    mf = result.x[0]
    return float(mf)

def calculate_mf_LiOH(RH):
    """
    Calculate the mass fraction of Lithium Hydroxide as a function of Relative Humidity at 25°C.
    Based on experimental data from: https://pubs.acs.org/doi/epdf/10.1021/ie0489148

    Parameters:
    RH : float
        Relative Humidity (0 < RH < 1)

    Returns:
    mf : float
        Mass fraction of Lithium Hydroxide
    """
    if RH <= 0 or RH > 1:
        raise ValueError("RH should be 0 < RH < 1")

    A_4 = 0
    A_3 = 0
    A_2 = -0.8825
    A_1 = -1.3171
    A_0 = 0.9993

    def func(X):
        return RH - (A_0 + A_1 * X + A_2 * X**2 + A_3 * X**3 + A_4 * X**4)

    x0 = 0.20
    bounds = (0, 0.65)

    result = least_squares(func, x0, bounds=bounds, ftol=1e-6, xtol=1e-6)
    mf = result.x[0]
    return float(mf)

# === Molecular Weights and Constants ===
T = 25  # °C
MWw = 18  # Molecular weight of water (g/mol)

# Helper function to calculate U (uptake) from mf
def calculate_U_gg(RH_array, T, mf_function):
    result = []
    n_params = len(inspect.signature(mf_function).parameters)
    for RH in RH_array:
        if n_params == 2:
            mf = mf_function(RH, T)
        else:
            mf = mf_function(RH)
        result.append(1 / mf - 1)
    return np.array(result)

def calculate_U_molmol(U_gg, MW_salt, n_water, MWw=18):
    return U_gg * ((n_water * MWw) / MW_salt) ** -1

# === LiCl ===
MW_LiCl = 42.4
RH_LiCl = np.linspace(0.12, 0.9)
U_LiCl_gg = calculate_U_gg(RH_LiCl, T, calculate_mf_LiCl)
U_LiCl_molmol = calculate_U_molmol(U_LiCl_gg, MW_LiCl, 2)

# === LiOH ===
MW_LiOH = 24
RH_LiOH = np.linspace(0.85, 0.9)
U_LiOH_gg = calculate_U_gg(RH_LiOH, T, calculate_mf_LiOH)
U_LiOH_molmol = calculate_U_molmol(U_LiOH_gg, MW_LiOH, 2)

# === NaOH ===
MW_NaOH = 40
RH_NaOH = np.linspace(0.23, 0.9)
U_NaOH_gg = calculate_U_gg(RH_NaOH, T, calculate_mf_NaOH)
U_NaOH_molmol = calculate_U_molmol(U_NaOH_gg, MW_NaOH, 2)

# === HCl ===
MW_HCl = 36.5
RH_HCl = np.linspace(0.17, 0.9)
U_HCl_gg = calculate_U_gg(RH_HCl, T, calculate_mf_HCl)
U_HCl_molmol = calculate_U_molmol(U_HCl_gg, MW_HCl, 2)

# === CaCl2 ===
MW_CaCl2 = 111
RH_CaCl2 = np.linspace(0.31, 0.9)
U_CaCl2_gg = calculate_U_gg(RH_CaCl2, T, calculate_mf_CaCl)
U_CaCl2_molmol = calculate_U_molmol(U_CaCl2_gg, MW_CaCl2, 3)

# === MgCl2 ===
MW_MgCl2 = 95.2
RH_MgCl2 = np.linspace(0.33, 0.9)
U_MgCl2_gg = calculate_U_gg(RH_MgCl2, T, calculate_mf_MgCl)
U_MgCl2_molmol = calculate_U_molmol(U_MgCl2_gg, MW_MgCl2, 3)

# === Mg(NO3)2 ===
MW_MgNO32 = 148.3
RH_MgNO32 = np.linspace(0.55, 0.9)
U_MgNO32_gg = calculate_U_gg(RH_MgNO32, T, calculate_mf_MgNO3)
U_MgNO32_molmol = calculate_U_molmol(U_MgNO32_gg, MW_MgNO32, 3)

# === LiBr ===
MW_LiBr = 86.85
RH_LiBr = np.linspace(0.07, 0.9)
U_LiBr_gg = calculate_U_gg(RH_LiBr, T, calculate_mf_LiBr)
U_LiBr_molmol = calculate_U_molmol(U_LiBr_gg, MW_LiBr, 2)

# === ZnCl2 ===
MW_ZnCl2 = 136.3
RH_ZnCl2 = np.linspace(0.07, 0.8)
U_ZnCl2_gg = calculate_U_gg(RH_ZnCl2, T, calculate_mf_ZnCl)
U_ZnCl2_molmol = calculate_U_molmol(U_ZnCl2_gg, MW_ZnCl2, 3)

# === ZnI2 ===
MW_ZnI2 = 319.18
RH_ZnI2 = np.linspace(0.25, 0.9)
U_ZnI2_gg = calculate_U_gg(RH_ZnI2, T, calculate_mf_ZnI)
U_ZnI2_molmol = calculate_U_molmol(U_ZnI2_gg, MW_ZnI2, 3)

# === ZnBr2 ===
MW_ZnBr2 = 225.2
RH_ZnBr2 = np.linspace(0.08, 0.85)
U_ZnBr2_gg = calculate_U_gg(RH_ZnBr2, T, calculate_mf_ZnBr)
U_ZnBr2_molmol = calculate_U_molmol(U_ZnBr2_gg, MW_ZnBr2, 3)

# === LiI ===
MW_LiI = 133.85
RH_LiI = np.linspace(0.18, 0.9)
U_LiI_gg = calculate_U_gg(RH_LiI, T, calculate_mf_LiI)
U_LiI_molmol = calculate_U_molmol(U_LiI_gg, MW_LiI, 2)

# === Ideal Solution ===
RHideal = np.linspace(0.01, 0.9)
U_ideal_molmol = RHideal / (1 - RHideal)

# === Plot g/g Uptake ===
plt.figure(figsize=(6, 4))
plt.plot(RH_LiCl, U_LiCl_gg, color=(0, 0.5, 0), label="LiCl")
plt.plot(RH_CaCl2, U_CaCl2_gg, color=(0.9290, 0.6940, 0.1250), label="CaCl₂")
plt.plot(RH_MgCl2, U_MgCl2_gg, color=(0.8500, 0.3250, 0.0980), label="MgCl₂")
plt.plot(RH_LiBr, U_LiBr_gg, color=(0.3010, 0.7450, 0.9330), label="LiBr")
plt.plot(RH_HCl, U_HCl_gg, label="HCl")
plt.plot(RH_MgNO32, U_MgNO32_gg, color=(0, 1, 1), label="Mg(NO₃)₂")
plt.plot(RH_NaOH, U_NaOH_gg, color=(1, 0, 1), label="NaOH")

plt.xlabel("Relative Humidity")
plt.ylabel("Uptake (g/g)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Uptake_screening_gg.tiff", dpi=600)
plt.close()

# === Plot mol/mol Uptake ===
plt.figure(figsize=(5 / 1.6, 4 / 1.6))
plt.plot(RH_LiCl * 100, U_LiCl_molmol, color=(0, 0.5, 0), label="LiCl")
plt.plot(RH_CaCl2 * 100, U_CaCl2_molmol, color=(0.9290, 0.6940, 0.1250), label="CaCl₂")
plt.plot(RH_MgCl2 * 100, U_MgCl2_molmol, color=(0.8500, 0.3250, 0.0980), label="MgCl₂")
plt.plot(RH_LiBr * 100, U_LiBr_molmol, color=(0.3010, 0.7450, 0.9330), label="LiBr")
plt.plot(RH_ZnCl2 * 100, U_ZnCl2_molmol, color=(0.6350, 0.0780, 0.1840), label="ZnCl₂")
plt.plot(RH_LiI * 100, U_LiI_molmol, color=(0.4940, 0.1840, 0.5560), label="LiI")
plt.plot(RH_ZnBr2 * 100, U_ZnBr2_molmol, color=(0.4660, 0.6740, 0.1880), label="ZnBr₂")
plt.plot(RH_HCl * 100, U_HCl_molmol, label="HCl")
plt.plot(RH_MgNO32 * 100, U_MgNO32_molmol, color=(0, 1, 1), label="Mg(NO₃)₂")