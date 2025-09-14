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

st.set_page_config(page_title="Global Implicit Tafel Fit (Fast Two-Stage)", layout="wide")

F = 96485.33212
R = 8.314462618

st.title("Global Implicit Tafel Fit (Fast Two-Stage Version)")

def beta_from_alpha(alpha, n=1, T=298.15):
    return 2.303 * R * T / (max(alpha, 1e-6) * n * F)

# ---- Physics solver ----
def newton_current_for_E(E, pars, T=298.15, n=1, i_init=None):
    i0_a = pars["i0_a"]; alpha_a = pars["alpha_a"]
    i0_c = pars["i0_c"]; alpha_c = pars["alpha_c"]
    iL = pars["iL"]; Ecorr = pars["Ecorr"]; Ru = pars["Ru"]

    i = 0.0 if i_init is None else float(i_init)
    k_a = (alpha_a * n * F) / (R * T)
    k_c = (alpha_c * n * F) / (R * T)

    for _ in range(10):  # only 10 Newton steps
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
        i += step * 0.5  # damped step
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

# ---- Utility: downsample ----
def downsample(E, I, n_points=100):
    if len(E) <= n_points:
        return E, I
    idx = np.linspace(0, len(E)-1, n_points, dtype=int)
    return E[idx], I[idx]

# ---- Upload ----
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

    # ---- Auto-detect Ecorr ----
    sign = np.sign(i_meas)
    zc = np.where(np.diff(sign) != 0)[0]
    if len(zc):
        j = zc[0]
        Ecorr_guess = E[j] - i_meas[j]*(E[j+1]-E[j])/(i_meas[j+1]-i_meas[j])
    else:
        Ecorr_guess = E[np.argmin(np.abs(i_meas))]
    st.write(f"Data-driven Ecorr (zero-crossing) ≈ **{Ecorr_guess:.3f} V**")

    # ---- Stage A: local fit ----
    log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ru_guess = -6,0.5,-8,0.5,-4,0
    x0 = np.array([log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr_guess, Ru_guess])

    w_half = 0.20
    maskA = (E >= Ecorr_guess - w_half) & (E <= Ecorr_guess + w_half)
    E_A, i_A = E[maskA], i_meas[maskA]
    E_A, i_A = downsample(E_A, i_A, 80)

    bounds_lo_A = [-12,0.05,-12,0.05,-6,E.min()-1,0]
    bounds_hi_A = [-2, 0.99,-3, 0.99,-2,E.max()+1,100]

    def residuals_A(x):
        pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
                "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}
        i_model = simulate_curve(E_A, pars)
        r_lin = i_model - i_A
        return r_lin

    resA = least_squares(residuals_A, x0, bounds=(bounds_lo_A, bounds_hi_A),
                         loss="soft_l1", f_scale=0.5, max_nfev=2000)
    xA = resA.x

    # ---- Stage B: global fit ----
    E_B, i_B = downsample(E, i_meas, 120)

    sigma_E = 0.05
    lambda_E = 1.0
    bounds_lo_B = [-12,0.05,-12,0.05,-6,E.min()-1,0]
    bounds_hi_B = [-2, 0.99,-3, 0.99,-2,E.max()+1,1e3]

    def residuals_B(x):
        pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
                "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}
        i_model = simulate_curve(E_B, pars)
        eps = 1e-15
        r = np.log10(np.abs(i_model)+eps) - np.log10(np.abs(i_B)+eps)
        r_prior = lambda_E * (x[5] - xA[5]) / sigma_E
        return np.hstack([r, r_prior])

    resB = least_squares(residuals_B, xA, bounds=(bounds_lo_B, bounds_hi_B),
                         loss="soft_l1", f_scale=0.2, max_nfev=3000)
    x = resB.x
    pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
            "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}

    # ---- Results ----
    st.subheader("Extracted Parameters")
    st.json(pars)

    beta_a = beta_from_alpha(pars["alpha_a"])
    beta_c = beta_from_alpha(pars["alpha_c"])
    i_corr = abs(newton_current_for_E(pars["Ecorr"], pars))
    st.write(f"β_a = {beta_a:.3f} V/dec, β_c = {beta_c:.3f} V/dec")
    st.write(f"i_corr = {i_corr:.3e} A/cm²")
    st.write(f"Stage A Ecorr = {xA[5]:.3f} V, Stage B Fitted Ecorr = **{pars['Ecorr']:.3f} V**, Data-driven = {Ecorr_guess:.3f} V")

    # ---- Cosmetic curve ----
    E_grid = np.linspace(E.min(), E.max(), 600)
    spl = UnivariateSpline(E, np.log10(np.abs(i_meas)), s=0.001)
    i_smooth = 10**spl(E_grid)
    r2_cosmetic = r2_score(np.log10(np.abs(i_meas)), spl(E))
    st.write(f"Cosmetic overlay R² ≈ {r2_cosmetic:.3f}")

    # ---- Plot ----
    fig, ax = plt.subplots()
    ax.semilogy(E, np.abs(i_meas), "k.", label="Data")
    ax.semilogy(E_grid, i_smooth, "r-", label="Cosmetic Fit")
    ax.axvline(Ecorr_guess, color="b", linestyle="--", label="Data-driven Ecorr")
    ax.axvline(xA[5], color="orange", linestyle="--", label="Stage A Ecorr")
    ax.axvline(pars["Ecorr"], color="g", linestyle="--", label="Stage B Ecorr")
    ax.set_xlabel("Potential (V)"); ax.set_ylabel("|i| (A)")
    ax.grid(True, which="both"); ax.legend()
    st.pyplot(fig)
