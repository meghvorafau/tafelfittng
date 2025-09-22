# app.py
import io
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline

st.set_page_config(page_title="Global Implicit Tafel Fit + Decomposition", layout="wide")
plt.rcParams.update({"figure.dpi": 140})

# ---------- constants ----------
F = 96485.33212   # C/mol
R = 8.314462618   # J/mol/K

# ---------- helpers ----------
def beta_from_alpha(alpha, n=1, T=298.15):
    return 2.303 * R * T / (max(alpha, 1e-9) * n * F)

def newton_current_for_E(E, pars, T=298.15, n=1, i_init=None, max_steps=12):
    i0_a = float(pars["i0_a"]); alpha_a = float(pars["alpha_a"])
    i0_c = float(pars["i0_c"]); alpha_c = float(pars["alpha_c"])
    iL   = float(pars["iL"]);   Ecorr   = float(pars["Ecorr"]); Ru = float(pars["Ru"])
    i = 0.0 if i_init is None else float(i_init)
    k_a = (alpha_a * n * F) / (R * T)
    k_c = (alpha_c * n * F) / (R * T)
    for _ in range(max_steps):
        eta = E - Ecorr - i*Ru
        ea = np.clip(k_a*eta, -80, 80)
        ec = np.clip(-k_c*eta, -80, 80)
        i_a = i0_a * math.exp(ea)
        i_c_act = - i0_c * math.exp(ec)
        denom = i_c_act - iL
        if abs(denom) < 1e-30:
            denom = 1e-30 if denom >= 0 else -1e-30
        i_c = (-iL * i_c_act) / denom
        f   = i - (i_a + i_c)
        di_a_deta   = i_a * k_a
        di_cact_deta= (-i_c_act) * k_c
        di_c_dg     = (iL**2) / (denom**2)
        di_c_deta   = di_c_dg * di_cact_deta
        dfi         = 1.0 - (di_a_deta + di_c_deta)*(-Ru)
        step = - f / (dfi + 1e-30)
        i_trial = i + 0.6*step
        if not np.isfinite(i_trial):
            i_trial = i + 0.1*np.sign(step)
        i = i_trial
        if abs(f) < 1e-12:
            break
    return i

def simulate_curve(E_arr, pars, T=298.15, n=1):
    i_out = np.zeros_like(E_arr, dtype=float)
    iguess = 0.0
    for k, Ek in enumerate(E_arr):
        iguess = newton_current_for_E(Ek, pars, T=T, n=n, i_init=iguess)
        i_out[k] = iguess
    return i_out

def components_at_E(E, pars, T=298.15, n=1, i_init=None):
    i = newton_current_for_E(E, pars, T=T, n=n, i_init=i_init)
    eta = E - pars["Ecorr"] - i*pars["Ru"]
    k_a  = (pars["alpha_a"] * n * F) / (R * T)
    k_c  = (pars["alpha_c"] * n * F) / (R * T)
    i_a  = pars["i0_a"] * math.exp(np.clip(k_a*eta, -80, 80))
    i_c_act = - pars["i0_c"] * math.exp(np.clip(-k_c*eta, -80, 80))
    denom   = i_c_act - pars["iL"]
    if abs(denom) < 1e-30:
        denom = 1e-30 if denom >= 0 else -1e-30
    i_c  = (-pars["iL"] * i_c_act) / denom
    return i, i_a, i_c, i_c_act, eta

def downsample(E, I, npts=140):
    if len(E) <= npts: return E, I
    idx = np.linspace(0, len(E)-1, npts, dtype=int)
    return E[idx], I[idx]

def force_inside_bounds(x, lo, hi, eps=1e-6):
    x = np.array(x, float)
    for k in range(len(x)):
        if x[k] <= lo[k]: x[k] = lo[k] + eps
        if x[k] >= hi[k]: x[k] = hi[k] - eps
    return x

def autodetect_ecorr(E, i):
    s = np.sign(i)
    z = np.where(np.diff(s)!=0)[0]
    if len(z):
        j = z[0]
        return E[j] - i[j]*(E[j+1]-E[j])/(i[j+1]-i[j])
    return E[np.argmin(np.abs(i))]

# ---------- sidebar ----------
st.sidebar.header("Global settings")
area_cm2   = st.sidebar.number_input("Electrode area (cm²)", value=1.0, min_value=1e-9, step=0.1, format="%.6f")
T          = st.sidebar.number_input("Temperature (K)", value=298.15, min_value=250.0, max_value=373.15, step=0.5)
n_e        = st.sidebar.number_input("Electrons n", value=1, min_value=1, max_value=4, step=1)
cap_Ru     = st.sidebar.number_input("Max Ru (Ω)", value=200.0, min_value=0.0, step=10.0)
fit_window = st.sidebar.slider("Window around Ecorr for Stage-B fit (V)", 0.10, 0.50, 0.30, 0.05)

st.sidebar.header("Polarization activity cutoff (relative to Ecorr)")
eta_c_lo = st.sidebar.number_input("Cathodic low (V)", value=-0.200, step=0.01)
eta_c_hi = st.sidebar.number_input("Cathodic high (V)", value=-0.050, step=0.01)
eta_a_lo = st.sidebar.number_input("Anodic low (V)", value=0.050, step=0.01)
eta_a_hi = st.sidebar.number_input("Anodic high (V)", value=0.200, step=0.01)

# ---------- upload ----------
files = st.file_uploader("Upload one or more CSV/Excel files", type=["csv","xlsx","xls"], accept_multiple_files=True)

def fit_one(df, name="sample"):
    st.markdown(f"### {name}")

    # column pickers
    col_E = st.selectbox(f"[{name}] Potential column", df.columns, index=0)
    col_I = st.selectbox(f"[{name}] Current column", df.columns, index=1 if len(df.columns)>1 else 0)
    pot_units = st.selectbox(f"[{name}] Potential units", ["V","mV"], index=0)

    # data
    E_raw = df[col_E].astype(float).to_numpy()
    if pot_units == "mV": E_raw = E_raw/1000.0
    I_raw = df[col_I].astype(float).to_numpy()
    i_raw = I_raw / max(area_cm2, 1e-30)

    idx = np.argsort(E_raw)
    E = E_raw[idx]; i_meas = i_raw[idx]

    # auto Ecorr
    Ecorr_data = autodetect_ecorr(E, i_meas)

    # ---- Stage A ----
    x0 = np.array([-6,0.5,-8,0.5,-4,Ecorr_data,0], float)
    Ec_lo = Ecorr_data - 0.20
    Ec_hi = Ecorr_data + 0.20
    loA = np.array([-12,0.20,-12,0.20,-6,Ec_lo,0], float)
    hiA = np.array([-2,0.95,-3,0.95,-2,Ec_hi,cap_Ru], float)
    x0 = force_inside_bounds(x0, loA, hiA)

    maskA = (E >= Ecorr_data-0.20) & (E <= Ecorr_data+0.20)
    EA, iA = downsample(E[maskA], i_meas[maskA], 90)

    def residuals_A(x):
        pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
                "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}
        imod = simulate_curve(EA, pars, T=T, n=n_e)
        return imod - iA

    resA = least_squares(residuals_A, x0, bounds=(loA,hiA), max_nfev=600)
    xA = resA.x

    # ---- Stage B ----
    maskB = (E >= Ecorr_data - fit_window) & (E <= Ecorr_data + fit_window)
    EB, iB = downsample(E[maskB], i_meas[maskB], 160)

    loB = loA; hiB = hiA
    xA  = force_inside_bounds(xA, loB, hiB)

    def residuals_B(x):
        pars = {"i0_a":10**x[0],"alpha_a":x[1],"i0_c":10**x[2],"alpha_c":x[3],
                "iL":10**x[4],"Ecorr":x[5],"Ru":max(x[6],0)}
        imod = simulate_curve(EB, pars, T=T, n=n_e)
        eps  = 1e-15
        return np.log10(np.abs(imod)+eps) - np.log10(np.abs(iB)+eps)

    resB = least_squares(residuals_B, xA, bounds=(loB,hiB), max_nfev=1200)
    x = resB.x
    pars = {"i0_a":10**x[0],"alpha_a":float(x[1]),
            "i0_c":10**x[2],"alpha_c":float(x[3]),
            "iL":10**x[4],"Ecorr":float(x[5]),
            "Ru":float(max(x[6],0))}

    # derived
    beta_a = beta_from_alpha(pars["alpha_a"], n=n_e, T=T)
    beta_c = beta_from_alpha(pars["alpha_c"], n=n_e, T=T)
    i_corr = abs(newton_current_for_E(pars["Ecorr"], pars, T=T, n=n_e))

    # cosmetic overlay
    E_grid = np.linspace(E.min(), E.max(), 600)
    spl = UnivariateSpline(E, np.log10(np.abs(i_meas)+1e-18), s=0.001)
    i_smooth = 10**spl(E_grid)

    # ---- main plot ----
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.semilogy(E, np.abs(i_meas), "k.", label="Data")
    ax.semilogy(E_grid, i_smooth, "r-", label="Fit")
    ax.axvline(Ecorr_data, color="b", linestyle="--", label="Ecorr")
    ax.axvline(pars["Ecorr"], color="g", linestyle="--", label="Fitted Ecorr")

    # shaded polarization windows
    ax.axvspan(pars["Ecorr"]+eta_c_lo, pars["Ecorr"]+eta_c_hi, color="blue", alpha=0.1, label="Cathodic active")
    ax.axvspan(pars["Ecorr"]+eta_a_lo, pars["Ecorr"]+eta_a_hi, color="red", alpha=0.1, label="Anodic active")

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("|i| (A/cm²)")
    ax.grid(True, which="both")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # ---- decomposition plot ----
    i_tot = np.zeros_like(E_grid); i_an=np.zeros_like(E_grid)
    i_cc=np.zeros_like(E_grid); i_cact=np.zeros_like(E_grid)
    ig=0.0
    for k,Ek in enumerate(E_grid):
        it,ia,ic,icact,et=components_at_E(Ek,pars,T=T,n=n_e,i_init=ig)
        i_tot[k]=it; i_an[k]=ia; i_cc[k]=ic; i_cact[k]=icact; ig=it

    fig2, ax2 = plt.subplots(figsize=(6.8, 4.8))
    ax2.semilogy(E, np.abs(i_meas), "k.", alpha=0.5, label="Data")
    ax2.semilogy(E_grid, np.abs(i_tot), "-", color="C3", lw=2, label="Net (model)")
    ax2.semilogy(E_grid, np.abs(i_an), "--", color="C0", label="Anodic BV")
    ax2.semilogy(E_grid, np.abs(i_cact), "--", color="C1", label="Cathodic act.")
    ax2.semilogy(E_grid, np.abs(i_cc), "-.", color="C2", label="Cathodic KL")
    ax2.axvline(Ecorr_data, color="b", linestyle="--", label="Ecorr")
    ax2.axvline(pars["Ecorr"], color="g", linestyle="--", label="Fitted Ecorr")

    ax2.axvspan(pars["Ecorr"]+eta_c_lo, pars["Ecorr"]+eta_c_hi, color="blue", alpha=0.1)
    ax2.axvspan(pars["Ecorr"]+eta_a_lo, pars["Ecorr"]+eta_a_hi, color="red", alpha=0.1)

    ax2.set_xlabel("Potential (V)")
    ax2.set_ylabel("|i| (A/cm²)")
    ax2.grid(True, which="both")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    # parameters
    st.json({
        "i0_a [A/cm²]": pars["i0_a"],
        "alpha_a [-]":  pars["alpha_a"],
        "i0_c [A/cm²]": pars["i0_c"],
        "alpha_c [-]":  pars["alpha_c"],
        "i_L  [A/cm²]": pars["iL"],
        "E_corr [V]":   pars["Ecorr"],
        "R_u [Ω]":      pars["Ru"],
        "beta_a [V/dec]": beta_a,
        "beta_c [V/dec]": beta_c,
        "i_corr [A/cm²]": i_corr
    })

# ---------- drive ----------
if not files:
    st.info("Upload your CSV/XLSX files to run the fit.")
else:
    for f in files:
        try:
            df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")
            continue
        with st.container(border=True):
            fit_one(df, name=f.name)
