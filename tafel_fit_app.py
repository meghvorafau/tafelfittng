import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score
from scipy.ndimage import uniform_filter1d

st.set_page_config(page_title="Global Implicit Tafel Fit (Auto Polarization Regions)", layout="wide")

F = 96485.33212
R = 8.314462618

st.title("Global Implicit Tafel Fit with Auto Polarization Activity Detection")

# ---- Helper functions ----
def beta_from_alpha(alpha, n=1, T=298.15):
    return 2.303 * R * T / (max(alpha, 1e-6) * n * F)

def newton_current_for_E(E, pars, T=298.15, n=1, i_init=None):
    i0_a, alpha_a = pars["i0_a"], pars["alpha_a"]
    i0_c, alpha_c = pars["i0_c"], pars["alpha_c"]
    iL, Ecorr, Ru = pars["iL"], pars["Ecorr"], pars["Ru"]

    i = 0.0 if i_init is None else float(i_init)
    k_a = (alpha_a * n * F) / (R * T)
    k_c = (alpha_c * n * F) / (R * T)

    for _ in range(15):  # capped iterations
        eta = E - Ecorr - i * Ru
        try:
            i_a = i0_a * math.exp(k_a * eta)
            i_c_act = -i0_c * math.exp(-k_c * eta)
        except OverflowError:
            return np.nan

        denom = (i_c_act - iL) if abs(i_c_act - iL) > 1e-30 else 1e-30
        i_c = (i_c_act * -iL) / denom
        f = i - (i_a + i_c)

        di_a_deta = i_a * k_a
        di_cact_deta = (-i_c_act) * k_c
        di_c_dg = (iL**2) / (denom**2)
        di_c_deta = di_c_dg * di_cact_deta
        dfi = 1 - (di_a_deta + di_c_deta) * -Ru

        step = -f / (dfi + 1e-30)
        i += step * 0.5
        if abs(f) < 1e-12:
            break
    return i

def simulate_curve(E_arr, pars, T=298.15, n=1):
    out = []
    i_guess = 0.0
    for E in E_arr:
        val = newton_current_for_E(E, pars, T=T, n=n, i_init=i_guess)
        if not np.isfinite(val): val = np.nan
        i_guess = val if np.isfinite(val) else 0.0
        out.append(val)
    return np.array(out)

def detect_tafel_regions(E, I, window=5, slope_tol=0.25):
    logi = np.log10(np.clip(np.abs(I), 1e-15, None))
    slope = np.gradient(logi, E)
    slope_smooth = uniform_filter1d(slope, size=window)

    Ecorr_guess = E[np.argmin(np.abs(I))]
    mask_c = E < Ecorr_guess
    mask_a = E > Ecorr_guess

    cathodic_region, anodic_region = None, None

    if mask_c.any():
        slopes_c, E_c = slope_smooth[mask_c], E[mask_c]
        mean_slope_c = np.nanmean(slopes_c)
        good_mask = np.abs(slopes_c - mean_slope_c) < slope_tol * abs(mean_slope_c)
        if np.any(good_mask):
            cathodic_region = (E_c[good_mask][0], E_c[good_mask][-1])

    if mask_a.any():
        slopes_a, E_a = slope_smooth[mask_a], E[mask_a]
        mean_slope_a = np.nanmean(slopes_a)
        good_mask = np.abs(slopes_a - mean_slope_a) < slope_tol * abs(mean_slope_a)
        if np.any(good_mask):
            anodic_region = (E_a[good_mask][0], E_a[good_mask][-1])

    return cathodic_region, anodic_region

# ---- Upload ----
data_file = st.file_uploader("Upload polarization data (CSV/Excel).", type=["csv","xlsx","xls"])
if data_file is not None:
    df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)
    st.success(f"Loaded {len(df)} rows.")
    st.dataframe(df.head(8))

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
    E, i_meas = E_raw[idx], i_meas[idx]

    # Detect Ecorr
    sign = np.sign(i_meas)
    zc = np.where(np.diff(sign) != 0)[0]
    if len(zc):
        j = zc[0]
        Ecorr_guess = E[j] - i_meas[j]*(E[j+1]-E[j])/(i_meas[j+1]-i_meas[j])
    else:
        Ecorr_guess = E[np.argmin(np.abs(i_meas))]
    st.write(f"Data-driven Ecorr ≈ **{Ecorr_guess:.3f} V**")

    # Fit
    log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ru_guess = -6,0.5,-8,0.5,-4,0
    x0 = np.array([log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr_guess, Ru_guess])
    bounds_lo = [-12,0.05,-12,0.05,-6,E.min()-1,0]
    bounds_hi = [-2,0.99,-3,0.99,-2,E.max()+1,1e3]

    def residuals(x):
        pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
                "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}
        i_model = simulate_curve(E, pars)
        eps = 1e-15
        return np.log10(np.abs(i_model)+eps) - np.log10(np.abs(i_meas)+eps)

    res = least_squares(residuals, x0, bounds=(bounds_lo,bounds_hi),
                        loss="soft_l1", f_scale=0.2, max_nfev=2000)
    x = res.x
    pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
            "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}

    # Results
    st.subheader("Extracted Parameters")
    st.json(pars)

    beta_a = beta_from_alpha(pars["alpha_a"])
    beta_c = beta_from_alpha(pars["alpha_c"])
    i_corr = abs(newton_current_for_E(pars["Ecorr"], pars))
    st.write(f"β_a = {beta_a:.3f} V/dec, β_c = {beta_c:.3f} V/dec")
    st.write(f"i_corr = {i_corr:.3e} A/cm²")
    st.write(f"Fitted Ecorr = **{pars['Ecorr']:.3f} V**, Data-driven = {Ecorr_guess:.3f} V")

    # Cosmetic curve
    E_grid = np.linspace(E.min(), E.max(), 600)
    spl = UnivariateSpline(E, np.log10(np.abs(i_meas)+1e-12), s=0.001)
    i_smooth = 10**spl(E_grid)

    # Detect polarization active regions automatically
    cath_region, anod_region = detect_tafel_regions(E, i_meas)

    # Plot
    fig, ax = plt.subplots(figsize=(7,5))
    ax.semilogy(E, np.abs(i_meas), "k.", label="Data")
    ax.semilogy(E_grid, i_smooth, "r-", label="Fit")
    ax.axvline(Ecorr_guess, color="b", linestyle="--", label="Ecorr")

    if cath_region:
        ax.axvspan(cath_region[0], cath_region[1], color="blue", alpha=0.1, label="Cathodic active")
    if anod_region:
        ax.axvspan(anod_region[0], anod_region[1], color="red", alpha=0.1, label="Anodic active")

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("|i| (A/cm²)")
    ax.grid(True, which="both")
    ax.legend(loc="lower right")
    st.pyplot(fig)
