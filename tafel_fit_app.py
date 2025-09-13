import io
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

st.set_page_config(page_title="Global Implicit Tafel Fit (BV + Koutecky–Levich + Ru)", layout="wide")

# Constants
F = 96485.33212  # C/mol
R = 8.314462618  # J/mol/K

st.title("Global Implicit Tafel Fit")
st.caption("Single implicit model: Butler–Volmer (anodic + cathodic) + Koutecky–Levich diffusion (ORR) + Ohmic drop Ru. "
           "Solve point-wise with Newton; fit globally.")

# -------------------- Physics --------------------
def beta_from_alpha(alpha, n=1, T=298.15):
    # β (V/dec) = 2.303 RT / (α n F)
    return 2.303 * R * T / (max(alpha, 1e-6) * n * F)

def newton_current_for_E(E, pars, T=298.15, n=1, i_init=None):
    """
    Solve implicit equation for current density i at given potential E:
      η = E − E_corr − i*Ru
      i_a = i0_a * exp( +α_a nF η / RT )
      i_c,act = − i0_c * exp( −α_c nF η / RT )      (cathodic < 0)
      i_c,eff via Koutecky–Levich: 1/i_c = 1/i_c,act + 1/(−iL)
      i_total = i_a + i_c,eff
      Find i such that f(i) = i − (i_a + i_c,eff) = 0
    Uses damped Newton iterations with analytic df/di via chain rule.
    """
    i0_a = pars["i0_a"]
    alpha_a = pars["alpha_a"]
    i0_c = pars["i0_c"]
    alpha_c = pars["alpha_c"]
    iL = pars["iL"]
    Ecorr = pars["Ecorr"]
    Ru = pars["Ru"]

    # Initial guess
    i = 0.0 if i_init is None else float(i_init)

    # Precompute factors
    k_a = (alpha_a * n * F) / (R * T)
    k_c = (alpha_c * n * F) / (R * T)

    # Damped Newton
    for _ in range(80):
        eta = E - Ecorr - i * Ru

        # Activation currents
        i_a = i0_a * math.exp(k_a * eta)                # anodic, positive
        i_c_act = - i0_c * math.exp(-k_c * eta)         # cathodic, negative

        # Koutecky–Levich combination for cathodic with diffusion limit (-iL plateau)
        denom = (i_c_act - iL)
        if abs(denom) < 1e-30:
            denom = -1e-30 if denom < 0 else 1e-30
        i_c = (i_c_act * (-iL)) / denom

        f = i - (i_a + i_c)

        # df/di = 1 - (di_a/dη + di_c/dη)*dη/di
        di_a_deta = i_a * k_a
        di_cact_deta = (-i_c_act) * k_c
        di_c_dg = (iL**2) / (denom**2)
        di_c_deta = di_c_dg * di_cact_deta
        di_sum_deta = di_a_deta + di_c_deta
        d_eta_di = -Ru
        dsum_di = di_sum_deta * d_eta_di
        dfi = 1.0 - dsum_di

        # Newton step
        step = - f / (dfi + 1e-30)

        # Damping
        lam = 1.0
        improved = False
        for _ in range(10):
            i_trial = i + lam * step
            eta_t = E - Ecorr - i_trial * Ru
            i_a_t = i0_a * math.exp(k_a * eta_t)
            i_c_act_t = - i0_c * math.exp(-k_c * eta_t)
            denom_t = (i_c_act_t - iL)
            if abs(denom_t) < 1e-30:
                denom_t = -1e-30 if denom_t < 0 else 1e-30
            i_c_t = (i_c_act_t * (-iL)) / denom_t
            f_t = i_trial - (i_a_t + i_c_t)
            if abs(f_t) < abs(f):
                i = i_trial
                improved = True
                break
            lam *= 0.5
        if not improved:
            i = i + 0.1 * step

        if abs(f) < 1e-12:
            break

    return i

def simulate_curve(E_arr, pars, T=298.15, n=1):
    i_out = np.zeros_like(E_arr, dtype=float)
    i_guess = 0.0
    for k, E in enumerate(E_arr):
        i_guess = newton_current_for_E(E, pars, T=T, n=n, i_init=i_guess)
        i_out[k] = i_guess
    return i_out

# -------------------- Data --------------------
data_file = st.file_uploader("Upload polarization data (CSV/Excel).", type=["csv","xlsx","xls"])

df = None
if data_file is not None:
    try:
        if data_file.name.lower().endswith(".csv"):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_excel(data_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

if df is not None:
    st.success(f"Loaded {len(df)} rows.")
    st.dataframe(df.head(8))

    col_E = st.selectbox("Potential column", list(df.columns))
    col_I = st.selectbox("Current column", list(df.columns))

    pot_units = st.selectbox("Potential units", ["V","mV"], index=0)
    cur_units = st.selectbox("Current units", ["A","mA","uA","nA"], index=1)
    area_mode = st.radio("Area handling", ["Single value", "From a column"], horizontal=True)
    if area_mode == "Single value":
        area_val = st.number_input("Electrode area (cm²)", value=1.0, min_value=1e-9, format="%.6f")
        A = float(area_val)
        area_arr = np.full(len(df), A, dtype=float)
    else:
        col_A = st.selectbox("Area column", list(df.columns))
        area_arr = df[col_A].astype(float).to_numpy()
        A = float(np.nanmedian(area_arr))

    E_raw = df[col_E].astype(float).to_numpy()
    if pot_units == "mV":
        E_raw = E_raw/1000.0
    I_raw = df[col_I].astype(float).to_numpy()
    cur_factor = {"A":1.0,"mA":1e-3,"uA":1e-6,"nA":1e-9}[cur_units]
    I = I_raw * cur_factor
    with np.errstate(divide="ignore", invalid="ignore"):
        i_meas = I / np.where(area_arr<=0, np.nan, area_arr)

    # Sort
    idx = np.argsort(E_raw)
    E = E_raw[idx]
    i_meas = i_meas[idx]

    st.subheader("Global parameters & initial guesses")
    col1, col2, col3 = st.columns(3)
    with col1:
        T = st.number_input("Temperature (K)", value=298.15, min_value=250.0, max_value=373.15, step=0.5)
        n = st.number_input("Electrons n", value=1, min_value=1, max_value=4, step=1)
        Ecorr_guess = float(np.nanmedian(E))
        Ecorr = st.number_input("E_corr initial (V)", value=Ecorr_guess, step=0.01, format="%.4f")
    with col2:
        log_i0a = st.slider("log10(i0_a) [A/cm²]", -12.0, -2.0, -6.0, 0.5)
        alpha_a = st.number_input("α_a (anodic)", value=0.5, min_value=0.05, max_value=0.99, step=0.01)
        log_i0c = st.slider("log10(i0_c) [A/cm²] (ORR act.)", -12.0, -3.0, -8.0, 0.5)
    with col3:
        alpha_c = st.number_input("α_c (cathodic)", value=0.5, min_value=0.05, max_value=0.99, step=0.01)
        log_iL = st.slider("log10(i_L) [A/cm²]", -6.0, -2.0, -4.0, 0.5)
        Ru_guess = st.number_input("R_u initial (Ω)", value=0.0, min_value=0.0, step=0.1)

    # Bounds
    bounds_lo = np.array([-12, 0.05, -12, 0.05, -6, np.min(E)-1.0, 0.0], float)
    bounds_hi = np.array([ -2, 0.99,  -3, 0.99, -2, np.max(E)+1.0, 1e6], float)
    x0 = np.array([log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr, Ru_guess], float)

    st.subheader("Fitting window")
    Emin, Emax = float(np.nanmin(E)), float(np.nanmax(E))
    fit_lo, fit_hi = st.slider("E range (V) to include", min_value=Emin, max_value=Emax, value=(Emin, Emax), step=0.01)
    mask = (E >= fit_lo) & (E <= fit_hi) & np.isfinite(i_meas)
    E_fit = E[mask]
    i_fit = i_meas[mask]

    def residuals(x):
        log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr, Ru = x
        pars = {
            "i0_a": 10**log_i0a,
            "alpha_a": alpha_a,
            "i0_c": 10**log_i0c,
            "alpha_c": alpha_c,
            "iL": 10**log_iL,
            "Ecorr": Ecorr,
            "Ru": max(Ru, 0.0),
        }
        i_model = simulate_curve(E_fit, pars, T=T, n=n)
        eps = 1e-15
        r = np.log10(np.clip(np.abs(i_model), eps, None)) - np.log10(np.clip(np.abs(i_fit), eps, None))
        sign_pen = 0.2 * (np.sign(i_model) - np.sign(i_fit))**2
        return r + sign_pen

    if st.button("Run global fit", type="primary"):
        try:
            res = least_squares(residuals, x0, bounds=(bounds_lo, bounds_hi), max_nfev=6000, verbose=0)
            x = res.x
            log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr, Ru = x
            pars = {
                "i0_a": 10**log_i0a,
                "alpha_a": float(alpha_a),
                "i0_c": 10**log_i0c,
                "alpha_c": float(alpha_c),
                "iL": 10**log_iL,
                "Ecorr": float(Ecorr),
                "Ru": float(max(Ru, 0.0)),
            }
            st.success("Fit completed.")
            st.json(pars)

            # Derived
            beta_a = beta_from_alpha(pars["alpha_a"], n=n, T=T)
            beta_c = beta_from_alpha(pars["alpha_c"], n=n, T=T)
            st.write(f"β_a ≈ **{beta_a:.3f} V/dec**,  β_c ≈ **{beta_c:.3f} V/dec**")

            i_corr = abs(newton_current_for_E(pars["Ecorr"], pars, T=T, n=n))
            st.write(f"Estimated i_corr ≈ **{i_corr:.3e} A/cm²**")

            # Rp (Ru→0) numerical
            def i_at_eta(eta):
                ptmp = dict(pars); ptmp["Ru"] = 0.0
                E0 = pars["Ecorr"] + eta
                return newton_current_for_E(E0, ptmp, T=T, n=n)
            deta = 1e-4
            di_deta = (i_at_eta(deta) - i_at_eta(-deta)) / (2*deta)
            Rp = 1.0 / max(di_deta, 1e-30)
            st.write(f"Polarization resistance Rp (near Ecorr, Ru→0) ≈ **{Rp:.2e} Ω·cm²**")

            # Corrosion rate
            st.subheader("Corrosion rate (optional)")
            colr1, colr2 = st.columns(2)
            with colr1:
                EW = st.number_input("Equivalent weight (g/equiv)", value=27.92, help="E.g., iron ≈ 27.92 g/equiv")
            with colr2:
                rho = st.number_input("Density (g/cm³)", value=7.87, help="E.g., steel ≈ 7.87 g/cm³")
            K = 3.27e-3
            corr_rate = K * i_corr * EW / max(rho, 1e-9)
            st.write(f"Corrosion rate ≈ **{corr_rate:.3f} mm/year**")

            # Deconvolution
            E_grid = np.linspace(E.min(), E.max(), 600)
            i_tot = np.zeros_like(E_grid)
            i_an, i_cc, i_cc_act, eta_arr = [], [], [], []
            for Ek in E_grid:
                i_k = newton_current_for_E(Ek, pars, T=T, n=n)
                eta_k = Ek - pars["Ecorr"] - i_k*pars["Ru"]
                k_a = (pars["alpha_a"] * n * F) / (R * T)
                k_c = (pars["alpha_c"] * n * F) / (R * T)
                i_a = pars["i0_a"] * math.exp(k_a * eta_k)
                i_c_act = - pars["i0_c"] * math.exp(-k_c * eta_k)
                denom = (i_c_act - pars["iL"])
                denom = denom if abs(denom) > 1e-30 else (1e-30 if denom > 0 else -1e-30)
                i_c = (i_c_act * (-pars["iL"])) / denom

                i_tot[len(i_an)] = i_k
                i_an.append(i_a)
                i_cc.append(i_c)
                i_cc_act.append(i_c_act)
                eta_arr.append(eta_k)

            i_an = np.array(i_an); i_cc = np.array(i_cc); i_cc_act = np.array(i_cc_act); eta_arr = np.array(eta_arr)

            st.subheader("Semi-log plot")
            fig, ax = plt.subplots()
            ax.semilogy(E, np.abs(i_meas), ".", label="Data")
            ax.semilogy(E_grid, np.abs(i_tot), "-", label="Global fit")
            ax.semilogy(E_grid, np.abs(i_an), "--", label="Anodic BV")
            ax.semilogy(E_grid, np.abs(i_cc), "--", label="ORR (KL)")
            ax.set_xlabel("E (V)")
            ax.set_ylabel("|i| (A/cm²)")
            ax.grid(True, which="both")
            ax.legend()
            st.pyplot(fig)

            # Exports
            out = {
                "parameters": pars,
                "derived": {
                    "beta_a_V_per_dec": beta_a,
                    "beta_c_V_per_dec": beta_c,
                    "i_corr_Acm2": i_corr,
                    "Rp_Ohm_cm2": Rp
                }
            }
            st.download_button("Download parameters (JSON)", data=json.dumps(out, indent=2).encode("utf-8"),
                               file_name="global_tafel_params.json", mime="application/json")

            df_out = pd.DataFrame({
                "E_V": E_grid,
                "i_total_Acm2": i_tot,
                "i_anodic_Acm2": i_an,
                "i_cathodic_KL_Acm2": i_cc,
                "i_cathodic_activation_Acm2": i_cc_act,
                "eta_V": eta_arr
            })
            st.download_button("Download deconvoluted curve (CSV)",
                               data=df_out.to_csv(index=False).encode("utf-8"),
                               file_name="global_tafel_deconvoluted.csv",
                               mime="text/csv")

        except Exception as e:
            st.error(f"Fit failed: {e}")

    # Preview
    with st.expander("Preview (data only)"):
        fig2, ax2 = plt.subplots()
        ax2.semilogy(E, np.abs(i_meas), ".", label="Data")
        ax2.set_xlabel("E (V)")
        ax2.set_ylabel("|i| (A/cm²)")
        ax2.grid(True, which="both")
        st.pyplot(fig2)

else:
    st.info("Upload a CSV/Excel file and map the potential/current columns.")

st.markdown('---')
st.caption("η = E − E_corr − iRu; i_c via Koutecky–Levich; solved with damped Newton; global least-squares fit in log-space.")
