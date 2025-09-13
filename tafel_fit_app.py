"""
Streamlit application for multi‑process Tafel fitting
====================================================

This application provides a convenient interface for fitting electrochemical
polarisation data to a combination of Tafel and mixed activation/diffusion
models.  It is designed to handle multiple electrochemical processes
simultaneously—for example the reduction of hydrogen (HER), reduction of
oxygen (ORR) and dissolution of iron—and to extract the kinetic parameters
associated with each process.  The motivation and mathematical basis for
the fitting procedure are taken from the literature on polarisation curve
interpretation.  The total measured current at a given potential is the
sum of the individual contributions from each process.  For a purely
activation controlled reaction the relationship between overpotential and
current density is given by the classical Tafel equation
\(\eta = b\,\log_{10}(i/i_0)\)【228677896600433†L238-L248】.  For systems that show
diffusion limitation (for example oxygen reduction) the activation and
diffusion overpotentials are additive and an approximate expression for
the total current density is
\(i=\frac{i_L\,i_c}{i_L+i_c}\)【228677896600433†L268-L276】, where \(i_c\) is the activation
controlled current.  These relations form the basis of the model used in
this application.

When the script is executed with ``streamlit run tafel_app.py`` it presents
an interface that allows the user to:

* Upload an Excel file containing potential and current data.
* Choose which columns correspond to the potential and current.
* Specify the number of electrochemical processes to model.
* For each process, choose whether it is purely activation controlled or
  exhibits mixed activation/diffusion control and whether it is anodic
  (oxidation, positive current) or cathodic (reduction, negative current).
* Provide initial guesses and bounds for the kinetic parameters: the
  exchange current \(i_0\), Tafel slope \(\beta\) (in V per decade), the
  reversible potential \(E_\text{rev}\) and, where relevant, the limiting
  current \(i_L\).
* Fit the model to the data using non‑linear least squares and display the
  extracted parameters with 95 % confidence intervals.
* Visualise the measured data and the fitted contributions of the
  individual processes.

The app uses SciPy’s ``curve_fit`` routine to perform the optimisation.
Parameter bounds are enforced to keep the optimisation stable; by default
\(i_0\), \(\beta\) and \(i_L\) are constrained to be positive and \(E_\text{rev}\)
is allowed to vary within a reasonable range.
"""

import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit


def multi_process_model(E: np.ndarray,
                        params: np.ndarray,
                        types: List[str],
                        directions: List[str]) -> np.ndarray:
    """Compute the total current from multiple processes.

    Each process contributes either a purely activation controlled current
    or a mixed activation/diffusion controlled current.  The sign of the
    current is determined by the direction: cathodic contributions are
    negative and anodic contributions are positive.

    Parameters
    ----------
    E : np.ndarray
        Array of potentials (in volts).
    params : np.ndarray
        Flattened parameter vector.  For each process the parameters are
        ordered as ``[i0, beta, E_rev]`` for activation controlled processes
        and ``[i0, beta, E_rev, i_L]`` for diffusion limited processes.
    types : List[str]
        A list whose elements are either ``"Activation"`` or
        ``"Mixed"``, specifying the type for each process.
    directions : List[str]
        A list whose elements are either ``"Cathodic"`` or ``"Anodic"``,
        specifying the sign of the current for each process.

    Returns
    -------
    np.ndarray
        Total current as a function of potential.
    """
    i_total = np.zeros_like(E, dtype=float)
    idx = 0
    for proc_type, direction in zip(types, directions):
        i0 = params[idx]
        beta = params[idx + 1]
        E_rev = params[idx + 2]
        idx += 3

        # activation component
        # Using base 10: i_c = i0 * 10 ** ((E - E_rev) / beta)
        i_c = i0 * np.power(10.0, (E - E_rev) / beta)

        if proc_type == "Mixed":
            i_L = params[idx]
            idx += 1
            # Mixed activation/diffusion: logistic form i_L * i_c / (i_L + i_c)
            i_proc = (i_L * i_c) / (i_L + i_c)
        else:
            # Pure activation controlled process
            i_proc = i_c

        # Apply direction sign
        if direction == "Cathodic":
            i_proc = -i_proc
        i_total += i_proc

    return i_total


def build_parameter_vectors(types: List[str], directions: List[str],
                            guesses: List[Tuple[float, float, float, float]],
                            bounds: List[Tuple[Tuple[float, ...], Tuple[float, ...]]]) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Assemble the initial guess and bounds arrays for curve_fit.

    Parameters
    ----------
    types : list of str
        Process types ("Activation" or "Mixed").
    directions : list of str
        Process directions ("Cathodic" or "Anodic").  Not used here but
        kept for future extension.
    guesses : list of 4‑tuple
        For each process, a tuple of initial guesses (i0, beta, E_rev,
        i_L).  i_L is ignored for activation processes.
    bounds : list of 2‑tuple of tuples
        Lower and upper bounds for each parameter of each process.

    Returns
    -------
    init_params : np.ndarray
        Flattened initial guess vector.
    full_bounds : tuple of np.ndarray
        Tuple of lower and upper bound arrays.
    """
    init_params = []
    lower_bounds = []
    upper_bounds = []
    for proc_type, guess, bnd in zip(types, guesses, bounds):
        i0_g, beta_g, E_rev_g, i_L_g = guess
        (i0_min, beta_min, E_rev_min, i_L_min), (i0_max, beta_max, E_rev_max, i_L_max) = bnd
        # Append parameters for activation or mixed
        init_params += [i0_g, beta_g, E_rev_g]
        lower_bounds += [i0_min, beta_min, E_rev_min]
        upper_bounds += [i0_max, beta_max, E_rev_max]
        if proc_type == "Mixed":
            # Add limiting current
            init_params.append(i_L_g)
            lower_bounds.append(i_L_min)
            upper_bounds.append(i_L_max)
    return np.array(init_params), (np.array(lower_bounds), np.array(upper_bounds))


def main() -> None:
    st.set_page_config(page_title="Multi‑process Tafel Fitting", layout="wide")
    st.title("Multi‑process Tafel Fitting")
    st.markdown(
        """
        Use this tool to fit polarisation data to a combination of kinetic
        models.  The total current is modelled as the sum of contributions
        from each process, each of which can be purely activation controlled
        (Tafel) or mixed activation/diffusion controlled.  The fitting is
        performed using non‑linear least squares.

        **Kinetic model background:**

        - For an activation controlled process the overpotential \(\eta\) and
          current density \(i\) are related by the Tafel equation
          \(\eta = b\,\log_{10}(i/i_0)\)【228677896600433†L238-L248】, where \(i_0\) is the
          exchange current and \(b\) is the Tafel slope.
        - When diffusion limitation is present, the activation and diffusion
          overpotentials are additive.  An approximate expression for the
          total current density is given by
          \(i = \tfrac{i_L i_c}{i_L + i_c}\)【228677896600433†L268-L276】, where \(i_L\) is the limiting
          current and \(i_c\) is the activation controlled current from the
          Tafel relation.
        - The total measured current at a given potential is the algebraic sum
          of the individual process currents.
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Data input")
    uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    if uploaded_file is None:
        st.info("Please upload an Excel file to continue.")
        return

    # Read Excel file
    try:
        bytes_data = uploaded_file.read()
        excel_io = io.BytesIO(bytes_data)
        df = pd.read_excel(excel_io)
    except Exception as e:
        st.error(f"Failed to read Excel file: {e}")
        return

    if df.empty:
        st.error("The uploaded Excel file appears to be empty.")
        return

    cols = df.columns.tolist()
    st.sidebar.subheader("Column selection")
    pot_col = st.sidebar.selectbox("Potential (V) column", cols, index=0)
    cur_col = st.sidebar.selectbox("Current (A) column", cols, index=2 if len(cols) > 2 else 1)

    E_data = df[pot_col].to_numpy(dtype=float)
    I_data = df[cur_col].to_numpy(dtype=float)

    # Optionally filter out NaNs
    mask = np.isfinite(E_data) & np.isfinite(I_data)
    E_data = E_data[mask]
    I_data = I_data[mask]

    st.sidebar.subheader("Model configuration")
    max_processes = 5
    num_proc = st.sidebar.number_input("Number of processes", min_value=1, max_value=max_processes, value=2, step=1)

    # Initialise lists for process configuration
    types: List[str] = []
    directions: List[str] = []
    guesses: List[Tuple[float, float, float, float]] = []
    bounds: List[Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]] = []

    # Provide default guesses based on data extent
    E_min, E_max = float(np.min(E_data)), float(np.max(E_data))
    I_abs_max = float(np.max(np.abs(I_data))) if len(I_data) > 0 else 1e-6

    for i in range(int(num_proc)):
        st.sidebar.markdown(f"### Process {i + 1}")
        p_type = st.sidebar.selectbox(
            f"Type of process {i + 1}",
            ("Activation", "Mixed"),
            index=0,
            key=f"type_{i}",
        )
        direction = st.sidebar.selectbox(
            f"Direction of process {i + 1}",
            ("Cathodic", "Anodic"),
            index=0,
            key=f"dir_{i}",
        )
        # Default guesses depend on magnitude of current
        default_i0 = I_abs_max * 1e-1 if I_abs_max > 0 else 1e-6
        default_beta = 0.1
        default_E_rev = 0.0
        default_i_L = I_abs_max * 10.0 if I_abs_max > 0 else 1.0
        i0_g = st.sidebar.number_input(
            f"Initial guess i₀ for process {i + 1} (A)",
            min_value=1e-12,
            value=default_i0,
            format="%e",
            key=f"i0_g_{i}",
        )
        beta_g = st.sidebar.number_input(
            f"Initial guess β for process {i + 1} (V/decade)",
            min_value=1e-4,
            value=default_beta,
            format="%f",
            key=f"beta_g_{i}",
        )
        E_rev_g = st.sidebar.number_input(
            f"Initial guess E_rev for process {i + 1} (V)",
            min_value=E_min - 1.0,
            max_value=E_max + 1.0,
            value=default_E_rev,
            format="%f",
            key=f"Erev_g_{i}",
        )
        if p_type == "Mixed":
            i_L_g = st.sidebar.number_input(
                f"Initial guess i_L for process {i + 1} (A)",
                min_value=1e-12,
                value=default_i_L,
                format="%e",
                key=f"iL_g_{i}",
            )
        else:
            i_L_g = 1.0  # placeholder, not used

        # Parameter bounds per process
        i0_min = st.sidebar.number_input(
            f"Lower bound i₀ for process {i + 1}",
            min_value=0.0,
            value=1e-12,
            format="%e",
            key=f"i0_min_{i}",
        )
        i0_max = st.sidebar.number_input(
            f"Upper bound i₀ for process {i + 1}",
            min_value=1e-12,
            value=I_abs_max * 1e3 if I_abs_max > 0 else 1.0,
            format="%e",
            key=f"i0_max_{i}",
        )
        beta_min = st.sidebar.number_input(
            f"Lower bound β for process {i + 1} (V/decade)",
            min_value=1e-4,
            value=1e-4,
            format="%f",
            key=f"beta_min_{i}",
        )
        beta_max = st.sidebar.number_input(
            f"Upper bound β for process {i + 1} (V/decade)",
            min_value=1e-4,
            value=1.0,
            format="%f",
            key=f"beta_max_{i}",
        )
        E_rev_min = st.sidebar.number_input(
            f"Lower bound E_rev for process {i + 1} (V)",
            min_value=E_min - 5.0,
            max_value=E_max + 5.0,
            value=E_min - 0.5,
            format="%f",
            key=f"Erev_min_{i}",
        )
        E_rev_max = st.sidebar.number_input(
            f"Upper bound E_rev for process {i + 1} (V)",
            min_value=E_min - 5.0,
            max_value=E_max + 5.0,
            value=E_max + 0.5,
            format="%f",
            key=f"Erev_max_{i}",
        )
        i_L_min = st.sidebar.number_input(
            f"Lower bound i_L for process {i + 1}",
            min_value=0.0,
            value=1e-12,
            format="%e",
            key=f"iL_min_{i}",
        )
        i_L_max = st.sidebar.number_input(
            f"Upper bound i_L for process {i + 1}",
            min_value=1e-12,
            value=I_abs_max * 1e3 if I_abs_max > 0 else 1.0,
            format="%e",
            key=f"iL_max_{i}",
        )

        # Append configuration lists
        types.append(p_type)
        directions.append(direction)
        guesses.append((i0_g, beta_g, E_rev_g, i_L_g))
        bounds.append(((i0_min, beta_min, E_rev_min, i_L_min), (i0_max, beta_max, E_rev_max, i_L_max)))

    # Build parameter vector and bounds
    init_params, full_bounds = build_parameter_vectors(types, directions, guesses, bounds)

    # Fit button
    if st.sidebar.button("Fit model"):
        with st.spinner("Fitting model, please wait..."):
            try:
                # Perform curve fitting
                popt, pcov = curve_fit(
                    lambda E, *params: multi_process_model(E, np.array(params), types, directions),
                    E_data,
                    I_data,
                    p0=init_params,
                    bounds=full_bounds,
                    maxfev=1000000,
                )
                # Calculate standard deviations and 95% CIs
                perr = np.sqrt(np.diag(pcov))
                ci = 1.96 * perr
                # Unpack parameters per process
                result_rows = []
                idx = 0
                for i in range(int(num_proc)):
                    proc_type = types[i]
                    direction = directions[i]
                    name = f"Process {i + 1}"
                    i0 = popt[idx]
                    i0_err = ci[idx]
                    beta = popt[idx + 1]
                    beta_err = ci[idx + 1]
                    E_rev = popt[idx + 2]
                    E_rev_err = ci[idx + 2]
                    idx += 3
                    row = {
                        "Process": name,
                        "Type": proc_type,
                        "Direction": direction,
                        "i0 (A)": i0,
                        "i0 CI": i0_err,
                        "β (V/dec)": beta,
                        "β CI": beta_err,
                        "E_rev (V)": E_rev,
                        "E_rev CI": E_rev_err,
                    }
                    if proc_type == "Mixed":
                        i_L = popt[idx]
                        i_L_err = ci[idx]
                        idx += 1
                        row.update({"i_L (A)": i_L, "i_L CI": i_L_err})
                    result_rows.append(row)
                result_df = pd.DataFrame(result_rows)
            except Exception as e:
                st.error(f"Error during fitting: {e}")
                return
        st.success("Fitting complete!")
        st.subheader("Fitted parameters (±95 % CI)")
        st.dataframe(result_df)

        # Plot measured and fitted data
        st.subheader("Polarisation curve and fitted contributions")
        # Create a high resolution potential grid for smooth curves
        E_grid = np.linspace(E_data.min(), E_data.max(), 400)
        total_fit = multi_process_model(E_grid, popt, types, directions)
        # Compute each process contribution
        contributions = []
        idx = 0
        for i, (p_type, direction) in enumerate(zip(types, directions)):
            i0 = popt[idx]
            beta = popt[idx + 1]
            E_rev = popt[idx + 2]
            idx += 3
            i_c = i0 * np.power(10.0, (E_grid - E_rev) / beta)
            if p_type == "Mixed":
                i_L = popt[idx]
                idx += 1
                i_proc = (i_L * i_c) / (i_L + i_c)
            else:
                i_proc = i_c
            if direction == "Cathodic":
                i_proc = -i_proc
            contributions.append(i_proc)
        # Create Altair chart data
        import altair as alt
        chart_data = pd.DataFrame({
            "Potential (V)": np.concatenate([E_data, E_grid]),
            "Current (A)": np.concatenate([I_data, total_fit]),
            "Series": ["Measured"] * len(E_data) + ["Fitted total"] * len(E_grid),
        })
        for i, contrib in enumerate(contributions):
            chart_data = pd.concat([
                chart_data,
                pd.DataFrame({
                    "Potential (V)": E_grid,
                    "Current (A)": contrib,
                    "Series": [f"Process {i + 1}"] * len(E_grid),
                })
            ], ignore_index=True)
        # Plot lines
        line_chart = alt.Chart(chart_data).mark_line().encode(
            x="Potential (V):Q",
            y="Current (A):Q",
            color="Series:N",
        ).interactive()
        st.altair_chart(line_chart, use_container_width=True)


if __name__ == "__main__":
    main()
