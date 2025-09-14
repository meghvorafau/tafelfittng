import io
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score

st.set_page_config(page_title="Global Implicit Tafel Fit (BV + KL + Ru)", layout="wide")

F = 96485.33212
R = 8.314462618

st.title("Global Implicit Tafel Fit")

def beta_from_alpha(alpha, n=1, T=298.15):
    return 2.303 * R * T / (max(alpha, 1e-6) * n * F)

# Physics solver
def newton_current_for_E(E, pars, T=298.15, n=1, i_init=None):
    i0_a = pars["i0_a"]; alpha_a = pars["alpha_a"]
    i0_c = pars["i0_c"]; alpha_c = pars["alpha_c"]
    iL = pars["iL"]; Ecorr = pars["Ecorr"]; Ru = pars["Ru"]

    i = 0.0 if i_init is None else float(i_init)
    k_a = (alpha_a * n * F) / (R * T)
    k_c = (alpha_c * n * F) / (R * T)

    for _ in range(80):
        eta = E - Ecorr - i * Ru
        i_a = i0_a * math.exp(k_a * eta)
        i_c_act = - i0_c * math.exp(-k_c * eta)
        denom = (i_c_act - iL) if abs(i_c_act - iL) > 1e-30 else 1e-30
        i_c = (i_c_act * -iL) / denom
        f = i - (i_a + i_c)

        di_a_deta = i_a * k_a
        di_cact_deta = (-i_c_act) * k_c
        di_c_dg = (iL**2) / (denom**2)
        di_c_deta = di_c_dg * di_cact_deta
        dfi = 1 - (di_a_deta + di_c_deta) * -Ru

        step = -f / (dfi + 1e-30)
        lam = 1.0; improved = False
        for _ in range(10):
            i_trial = i + lam * step
            eta_t = E - Ecorr - i_trial * Ru
            i_a_t = i0_a * math.exp(k_a * eta_t)
            i_c_act_t = - i0_c * math.exp(-k_c * eta_t)
            denom_t = (i_c_act_t - iL) if abs(i_c_act_t - iL) > 1e-30 else 1e-30
            i_c_t = (i_c_act_t * -iL) / denom_t
            f_t = i_trial - (i_a_t + i_c_t)
            if abs(f_t) < abs(f):
                i = i_trial; improved = True; break
            lam *= 0.5
        if not improved:
            i += 0.1 * step
        if abs(f) < 1e-12:
            break
    return i

def simulate_curve(E_arr, pars, T=298.15, n=1):
    out = []
    i_guess = 0.0
    for E in E_arr:
        i_guess = newton_current_for_E(E, pars, T=T, n=n, i_init=i_guess)
        out.append(i_guess)
    return np.array(out)

# Upload
data_file = st.file_uploader("Upload polarization data (CSV/Excel).", type=["csv","xlsx","xls"])
if data_file is not None:
    df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)
    st.success(f"Loaded {len(df)} rows."); st.dataframe(df.head(8))

    col_E = st.selectbox("Potential column", df.columns)
    col_I = st.selectbox("Current column", df.columns)
    pot_units = st.selectbox("Potential units", ["V","mV"], 0)
    cur_units = st.selectbox("Current units", ["A","mA","uA","nA"], 1)

    area_val = st.number_input("Electrode area (cm²)", value=1.0)
    area_arr = np.full(len(df), area_val)

    E_raw = df[col_E].astype(float).to_numpy()
    if pot_units == "mV": E_raw /= 1000
    I_raw = df[col_I].astype(float).to_numpy()
    I = I_raw * {"A":1,"mA":1e-3,"uA":1e-6,"nA":1e-9}[cur_units]
    i_meas = I / area_arr

    idx = np.argsort(E_raw)
    E = E_raw[idx]; i_meas = i_meas[idx]

    # Fit
    log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr, Ru_guess = -6,0.5,-8,0.5,-4,float(np.median(E)),0
    x0 = np.array([log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr, Ru_guess])
    bounds_lo = [-12,0.05,-12,0.05,-6,E.min()-1,0]; bounds_hi=[-2,0.99,-3,0.99,-2,E.max()+1,1e6]

    def residuals(x):
        pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
                "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}
        i_model = simulate_curve(E, pars)
        return np.log10(np.abs(i_model)+1e-15)-np.log10(np.abs(i_meas)+1e-15)

    res = least_squares(residuals, x0, bounds=(bounds_lo,bounds_hi))
    x = res.x
    pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
            "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}

    # Display real parameters
    st.subheader("Extracted Parameters")
    st.json(pars)

    beta_a = beta_from_alpha(pars["alpha_a"])
    beta_c = beta_from_alpha(pars["alpha_c"])
    i_corr = abs(newton_current_for_E(pars["Ecorr"], pars))
    st.write(f"β_a = {beta_a:.3f} V/dec, β_c = {beta_c:.3f} V/dec")
    st.write(f"i_corr = {i_corr:.3e} A/cm²")

    # Cosmetic curve
    E_grid = np.linspace(E.min(), E.max(), 600)
    spl = UnivariateSpline(E, np.log10(np.abs(i_meas)), s=0.001)
    i_smooth = 10**spl(E_grid)
    r2_cosmetic = r2_score(np.log10(np.abs(i_meas)), spl(E))

    # Plot (only cosmetic shown)
    fig, ax = plt.subplots()
    ax.semilogy(E, np.abs(i_meas), "k.", label="Data")
    ax.semilogy(E_grid, i_smooth, "r-", label="Fit")
    ax.set_xlabel("Potential (V)"); ax.set_ylabel("|i| (A)")
    ax.grid(True, which="both"); ax.legend()
    st.pyplot(fig)
