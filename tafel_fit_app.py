
import io
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

st.set_page_config(page_title="Multi‑Process Tafel Fitting", layout="wide")

# --------------- Helpers ---------------

F = 96485.33212  # C/mol
R = 8.314462618  # J/mol/K

REF_OFFSETS = {
    "SHE (0 mV)": 0.0,
    "Ag/AgCl sat. KCl (+197 mV)": 0.197,
    "Ag/AgCl 3M KCl (+210 mV)": 0.210,
    "SCE (+241 mV)": 0.241
}

def nernst_rev(reaction: str, pH: float, T: float = 298.15):
    """
    Returns reversible potential vs SHE (V) for HER or ORR at given pH (neutral/acid assumption)
    HER: Erev = -0.0591 * pH  (25C), adjust with T
    ORR: Erev = 1.229 - 0.0591 * pH  (25C)
    """
    # Temperature correction: 2.303*R*T/F * pH term
    slope = 2.303*R*T/F  # ~0.0591 V at 298K
    if reaction.lower() == "her":
        return -slope * pH
    if reaction.lower() == "orr":
        return 1.229 - slope * pH
    return 0.0

def taf_elog(i, i0, beta):
    # E - Erev = beta * log10(|i|/i0)  ; beta signed (V/dec). i is signed.
    # Rearranged to compute overpotential from current; used only for diagnostics.
    return beta * (np.log10(np.abs(i)/i0 + 1e-30))

def butler_volmer_current(E, Erev, i0, alpha, n=1, T=298.15):
    eta = E - Erev
    term_a = np.exp( (alpha*n*F*eta)/(R*T) )
    term_c = np.exp( (-(1.0-alpha)*n*F*eta)/(R*T) )
    return i0*(term_a - term_c)  # signed current density (A/cm^2)

def cathodic_tafel_current(E, Erev, i0, beta, n=1):
    # For large cathodic overpotential: i = -i0 * 10^{ |eta|/|beta| }
    eta = E - Erev
    return - i0 * 10.0**( -eta/abs(beta) )

def anodic_tafel_current(E, Erev, i0, beta, n=1):
    eta = E - Erev
    return + i0 * 10.0**( +eta/abs(beta) )

def orr_mixed_current(E, Erev, i0, beta, iL, mode="tafel"):
    """
    Mixed activation-diffusion cathodic branch.
    Using a common approximation (see van Ede & Angst 2024):
    Combine activation current ic_act with diffusion limit iL via 1/ic = 1/ic_act + 1/(-iL)
    """
    ic_act = - i0 * 10.0**( -(E - Erev)/abs(beta) )  # cathodic (negative)
    # Mix with diffusion limit (negative): i = (iL * ic_act) / (iL + |ic_act|)
    # keep sign negative
    return ( -iL * ic_act ) / ( iL + np.abs(ic_act) + 1e-30 )

def ir_correct(E_meas, I, Rs):
    return E_meas - I*Rs

def stern_geary_beta(beta_a, beta_c):
    # Stern–Geary constant B (V) with β in V/dec
    return (beta_a * beta_c) / (2.303*(beta_a + beta_c) + 1e-30)

def icorr_from_Rp(Rp, beta_a, beta_c, area_cm2=1.0):
    B = stern_geary_beta(beta_a, beta_c)
    # Rp typically in ohm*cm^2 if corrected for area
    return B / (Rp + 1e-30)

def corrosion_rate_mm_per_yr(icorr_Acm2, EW_g, rho_gcm3):
    # CR (mm/y) = (K * icorr * EW) / (rho)
    # Using K = 3.27e-3 when icorr in A/cm^2, EW in g/equiv, rho g/cm^3
    # (Derived from Faraday; classic ASTM expression)
    K = 3.27e-3
    return (K * icorr_Acm2 * EW) / (rho_gcm3 + 1e-30)

# --------------- UI ---------------

st.title("Multi‑Process Tafel Fitting (activation + diffusion)")
st.caption("Fit HER, ORR, and anodic dissolution concurrently, extract β, i₀, i_L, E_corr, i_corr, corrosion rate, and export results.")

with st.expander("How it works (models & references)", expanded=False):
    st.markdown("""
- We model each partial reaction with Tafel-like kinetics (activation control) and optionally mix the ORR with a diffusion limit to capture plateaus.  
- The measured current is the algebraic sum of anodic and cathodic partial currents. We optionally apply **ohmic-drop correction** (E_true = E_meas − I·Rₛ) inside the fit loop.  
- Key references that motivate this deconvolution style and parameterization:  
  - Flitt & Schweinsberg (2005) *A guide to polarisation curve interpretation* — deconstruction of curves into anodic/cathodic components and handling mixed control/passivation ideas.  
  - van Ede & Angst (2024) *Tafel slopes and exchange current densities of ORR/HER on steel* — practical fitting windows and notes on scan direction, film effects, and mixed control for ORR.
""")

data_file = st.file_uploader("Upload polarisation data (CSV or Excel).", type=["csv","xlsx","xls"])

default_cols = {"E (V)": None, "I (A)": None, "Area (cm²)": None}
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
    st.success(f"Loaded data with {len(df)} rows.")
    st.dataframe(df.head(10))

    # Column mapping
    st.subheader("Map columns")
    col_E = st.selectbox("Potential column", list(df.columns))
    col_I = st.selectbox("Current column", list(df.columns))
    area_mode = st.radio("Area handling", ["Single value", "From a column"], horizontal=True)
    if area_mode == "Single value":
        area_val = st.number_input("Electrode area (cm²)", value=1.0, min_value=1e-6, step=0.1, format="%.6f")
        area_series = pd.Series([area_val]*len(df))
    else:
        col_A = st.selectbox("Area column", list(df.columns))
        area_series = df[col_A].astype(float)

    # Units
    st.subheader("Units & reference electrode")
    pot_units = st.selectbox("Potential units", ["V", "mV"])
    cur_units = st.selectbox("Current units", ["A", "mA", "uA", "nA"])
    ref_name = st.selectbox("Reference electrode", list(REF_OFFSETS.keys()))
    E_offset = REF_OFFSETS[ref_name]
    to_SHE = st.checkbox("Convert to SHE scale using the selected reference offset", value=True)

    # Compute E (V vs SHE) and i (A/cm^2)
    E_raw = df[col_E].astype(float).to_numpy()
    if pot_units == "mV":
        E_raw = E_raw/1000.0
    I_raw = df[col_I].astype(float).to_numpy()

    cur_factor = {"A":1.0,"mA":1e-3,"uA":1e-6,"nA":1e-9}[cur_units]
    I = I_raw*cur_factor
    A_cm2 = area_series.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        i = I / np.where(A_cm2<=0, np.nan, A_cm2)

    E = E_raw.copy()
    if to_SHE:
        E = E_raw + E_offset

    # Sort by E for nicer plots
    order = np.argsort(E)
    E = E[order]
    i = i[order]

    # IR drop correction inside fit loop; but we can offer a quick external preview too
    st.subheader("Ohmic drop (optional)")
    Rs = st.number_input("Solution resistance Rₛ (Ω)", value=0.0, min_value=0.0, step=0.1)
    preview_ir = st.checkbox("Apply IR correction to preview plot (for visual aid only)", value=False)
    if preview_ir:
        E_preview = E - (i * A_cm2[order]) * Rs  # using total current density -> total current via area
    else:
        E_preview = E

    # Model selection
    st.subheader("Select partial reactions to include")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_anodic = st.checkbox("Anodic dissolution (activation)", value=True)
    with col2:
        use_her = st.checkbox("HER (activation)", value=True)
    with col3:
        use_orr = st.checkbox("ORR (activation + optional diffusion limit)", value=True)

    st.subheader("Environment & equilibria")
    T = st.number_input("Temperature (K)", value=298.15, min_value=250.0, max_value=373.15, step=0.5)
    pH = st.number_input("pH (for E_rev of HER/ORR)", value=7.0, step=0.1)

    Erev_HER = nernst_rev("her", pH, T)
    Erev_ORR = nernst_rev("orr", pH, T)

    st.write(f"E_rev(HER) ~ {Erev_HER:+.3f} V vs SHE;  E_rev(ORR) ~ {Erev_ORR:+.3f} V vs SHE")

    st.subheader("Initial guesses & bounds")

    # Initial guesses
    guess = {}
    bounds = {"lo":[], "hi":[]}
    params = []

    if use_anodic:
        st.markdown("**Anodic (active):** i = i0_a * 10^{(E − Erev_a)/β_a}")
        with st.expander("Anodic parameters", expanded=False):
            Erev_a = st.number_input("E_rev (anodic, V vs SHE)", value=float(np.nan_to_num(np.nanmedian(E), nan=0.0)))
            i0_a = float(10**st.slider("log10(i0_a) [A/cm²]", -12.0, -2.0, -6.0, 0.5))
            beta_a = st.number_input("β_a (V/dec)", value=0.12, min_value=0.01, max_value=0.5, step=0.01)
            lo_i0a, hi_i0a = 1e-12, 1e-2
            lo_ba, hi_ba = 0.02, 0.5
        guess.update(Erev_a=Erev_a, i0_a=i0_a, beta_a=beta_a)
        params += ["Erev_a","i0_a","beta_a"]
        bounds["lo"] += [Erev_a-0.5, lo_i0a, lo_ba]
        bounds["hi"] += [Erev_a+0.5, hi_i0a, hi_ba]

    if use_her:
        st.markdown("**HER (cathodic, activation):** i = − i0_h * 10^{−(E − Erev_h)/|β_h|}")
        with st.expander("HER parameters", expanded=False):
            i0_h = float(10**st.slider("log10(i0_h) [A/cm²]", -10.0, -3.0, -6.0, 0.5))
            beta_h = st.number_input("β_h (V/dec) (≈0.12 for α=0.5)", value=0.12, min_value=0.03, max_value=0.5, step=0.01)
        guess.update(i0_h=i0_h, beta_h=beta_h)
        params += ["i0_h","beta_h"]
        bounds["lo"] += [1e-10, 0.03]
        bounds["hi"] += [1e-3, 0.5]

    if use_orr:
        st.markdown("**ORR (cathodic, activation ± diffusion):** mixed‑control approximation with i_L")
        with st.expander("ORR parameters", expanded=False):
            i0_o = float(10**st.slider("log10(i0_o) [A/cm²]", -12.0, -6.0, -9.0, 0.5))
            beta_o = st.number_input("β_o (V/dec)", value=0.12, min_value=0.05, max_value=0.5, step=0.01)
            use_iL = st.checkbox("Include diffusion limiting current i_L (plateau)", value=True)
            iL = float(10**st.slider("log10(i_L) [A/cm²]", -6.0, -2.0, -4.0, 0.5)) if use_iL else 0.0
        guess.update(i0_o=i0_o, beta_o=beta_o, iL=iL)
        params += ["i0_o","beta_o","iL"]
        bounds["lo"] += [1e-12, 0.05, 0.0]
        bounds["hi"] += [1e-6, 0.5, 1e-2]

    # Ohmic resistance
    include_Rs = st.checkbox("Fit solution resistance Rₛ (Ω) inside the model", value=(Rs==0.0))
    if include_Rs:
        Rs_guess = st.number_input("Initial guess for Rₛ (Ω)", value=float(Rs))
        guess.update(Rs=Rs_guess)
        params += ["Rs"]
        bounds["lo"] += [0.0]
        bounds["hi"] += [1e6]

    # Fitting window
    st.subheader("Fitting window (by potential vs SHE)")
    E_min, E_max = float(np.nanmin(E_preview)), float(np.nanmax(E_preview))
    fit_lo, fit_hi = st.slider("Select E-range to fit (V vs SHE)", min_value=E_min, max_value=E_max, value=(E_min, E_max), step=0.01)

    mask = (E_preview>=fit_lo) & (E_preview<=fit_hi) & np.isfinite(i)
    E_fit = E[mask]
    i_fit = i[mask]

    if len(E_fit) < 10:
        st.warning("Select a wider range or check data mapping; fewer than 10 points in fit window.")

    # Model function
    def model_current(pvec, E_arr):
        loc = {k:v for k,v in zip(params, pvec)}
        # Determine Rs inside loop
        Rs_in = loc.get("Rs", Rs)
        # Build summed current from components
        I_comp = np.zeros_like(E_arr)
        # Use E_true = E_meas - I*Rs -> implicit; iterate (one fixed-point iteration works for moderate Rs)
        E_eff = E_arr.copy()  # initial
        for _ in range(3):
            I_comp = 0.0
            if use_anodic:
                I_comp = I_comp + anodic_tafel_current(E_eff, loc["Erev_a"], loc["i0_a"], loc["beta_a"])
            if use_her:
                I_comp = I_comp + cathodic_tafel_current(E_eff, Erev_HER, loc["i0_h"], loc["beta_h"])
            if use_orr:
                if loc["iL"] > 0:
                    I_comp = I_comp + orr_mixed_current(E_eff, Erev_ORR, loc["i0_o"], loc["beta_o"], loc["iL"])
                else:
                    I_comp = I_comp + cathodic_tafel_current(E_eff, Erev_ORR, loc["i0_o"], loc["beta_o"])
            # update E_eff with IR correction (use total current density -> current = i*A; but A cancels if Rs is total solution Rs)
            E_eff = E_arr - (I_comp * Rs_in)
        return I_comp

    # Residual: fit in log-domain to weight decades fairly
    def residuals(pvec):
        pred = model_current(pvec, E_fit)
        # avoid sign mixing at zero; use magnitude in log domain and keep sign by weighting both sides
        eps = 1e-15
        obs = np.clip(np.abs(i_fit), eps, None)
        mod = np.clip(np.abs(pred), eps, None)
        return np.log10(mod) - np.log10(obs)

    x0 = np.array([guess[k] for k in params], dtype=float)
    lo = np.array(bounds["lo"], dtype=float)
    hi = np.array(bounds["hi"], dtype=float)

    # Run fit
    if st.button("Run fit", type="primary"):
        try:
            res = least_squares(residuals, x0, bounds=(lo,hi), max_nfev=20000, verbose=0)
            popt = res.x
            loc = {k:float(v) for k,v in zip(params, popt)}
            st.success("Fit completed.")

            # Compute fitted curve over full range + components
            E_grid = np.linspace(E.min(), E.max(), 500)
            Rs_in = loc.get("Rs", Rs)

            # Solve implicit E_true with simple fixed-point iterations
            def comps_at(E_arr):
                E_eff = E_arr.copy()
                for _ in range(3):
                    Ia = np.zeros_like(E_arr)
                    Ih = np.zeros_like(E_arr)
                    Io = np.zeros_like(E_arr)
                    if use_anodic:
                        Ia = anodic_tafel_current(E_eff, loc["Erev_a"], loc["i0_a"], loc["beta_a"])
                    if use_her:
                        Ih = cathodic_tafel_current(E_eff, Erev_HER, loc["i0_h"], loc["beta_h"])
                    if use_orr:
                        if loc["iL"] > 0:
                            Io = orr_mixed_current(E_eff, Erev_ORR, loc["i0_o"], loc["beta_o"], loc["iL"])
                        else:
                            Io = cathodic_tafel_current(E_eff, Erev_ORR, loc["i0_o"], loc["beta_o"])
                    Itot = Ia + Ih + Io
                    E_eff = E_arr - Itot*Rs_in
                return Ia, Ih, Io, Itot, E_eff

            Ia, Ih, Io, Itot, E_eff = comps_at(E_grid)

            # Ecorr where Itot ~ 0 (interpolate)
            Ecorr = np.nan
            try:
                sign = np.sign(Itot)
                zc = np.where(np.diff(sign)!=0)[0]
                if len(zc) > 0:
                    i0_ = zc[0]
                    x1,x2 = E_grid[i0_], E_grid[i0_+1]
                    y1,y2 = Itot[i0_], Itot[i0_+1]
                    Ecorr = x1 - y1*(x2-x1)/(y2-y1 + 1e-30)
            except Exception:
                pass

            # i_corr at Ecorr (evaluate model)
            icorr = np.nan
            if not np.isnan(Ecorr):
                Ia_c, Ih_c, Io_c, Itot_c, _ = comps_at(np.array([Ecorr]))
                icorr = float(abs(Itot_c[0]))

            # Tables
            out = {"fitted_parameters": loc,
                   "derived": {
                       "Erev_HER_V_vs_SHE": Erev_HER,
                       "Erev_ORR_V_vs_SHE": Erev_ORR,
                       "Ecorr_V_vs_SHE": Ecorr,
                       "icorr_A_cm2": icorr
                   }}

            st.subheader("Fitted parameters")
            st.json(out)

            # Optional corrosion rate
            st.subheader("Corrosion rate (optional)")
            colr1, colr2 = st.columns(2)
            with colr1:
                EW = st.number_input("Equivalent weight (g/equiv)", value=27.92, help="E.g., iron ≈ 27.92 g/equiv")
                rho = st.number_input("Density (g/cm³)", value=7.87, help="E.g., iron/steel ≈ 7.87 g/cm³")
            if not np.isnan(icorr):
                cr = corrosion_rate_mm_per_yr(icorr, EW, rho)
                st.write(f"Estimated corrosion rate ≈ **{cr:.3f} mm/year**")

            # Plot
            st.subheader("Polarisation curve (semi‑log)")
            fig, ax = plt.subplots()
            ax.semilogy(E_preview, np.abs(i), ".", label="Data")
            ax.semilogy(E_grid, np.abs(Itot), "-", label="Fit")
            if use_anodic:
                ax.semilogy(E_grid, np.abs(Ia), "--", label="Anodic")
            if use_her:
                ax.semilogy(E_grid, np.abs(Ih), "--", label="HER")
            if use_orr:
                ax.semilogy(E_grid, np.abs(Io), "--", label="ORR")
            ax.set_xlabel("E (V vs SHE)")
            ax.set_ylabel("|i| (A/cm²)")
            ax.grid(True, which="both")
            ax.legend()
            st.pyplot(fig)

            # Export CSV
            st.subheader("Download results")
            df_out = pd.DataFrame({
                "E_V_vs_SHE": E_grid,
                "i_total_Acm2": Itot,
                "i_anodic_Acm2": Ia,
                "i_HER_Acm2": Ih,
                "i_ORR_Acm2": Io,
                "E_true_V_vs_SHE": E_eff
            })
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download deconvoluted curve CSV", data=csv, file_name="tafel_fit_deconvoluted.csv", mime="text/csv")

            json_bytes = json.dumps(out, indent=2).encode("utf-8")
            st.download_button("Download fitted parameters (JSON)", data=json_bytes, file_name="tafel_fit_params.json", mime="application/json")

        except Exception as e:
            st.error(f"Fit failed: {e}")

    # Always show a quick preview plot
    with st.expander("Quick preview (data only)", expanded=False):
        fig2, ax2 = plt.subplots()
        ax2.semilogy(E_preview, np.abs(i), ".", label="Data")
        ax2.set_xlabel("E (V vs SHE)")
        ax2.set_ylabel("|i| (A/cm²)")
        ax2.grid(True, which="both")
        st.pyplot(fig2)

else:
    st.info("Upload CSV/Excel with columns for potential and current.")

st.markdown("---")
st.caption("Models inspired by Flitt & Schweinsberg (2005) and van Ede & Angst (2024).")
