from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd

from problem import WeldedBeamProblem
from optimizer import run_hybrid, top_solutions_per_objective, top_solutions_per_constraint
from optimizer import HAS_SMPSO
from plotting import pareto_3d_figure, pareto_2d_figure

st.set_page_config(page_title="Welded Beam Multi-Objective Optimization", layout="wide", initial_sidebar_state="collapsed")

st.title("Welded Beam Optimization: Hybrid SMPSO → NSGA-II")

# Simple getting-started guide for beginners
st.markdown(
    """
    Follow these steps:
    1) Set your design sizes (Height h, Length l, Thickness t, Width b) using the horizontal controls at the top.
    2) Optionally adjust population and generations (you can keep defaults).
    3) Choose how much you care about Cost, Weight, and Weld Area using the sliders.
    4) Click "Run Optimization" to see the Pareto front and recommended best design.
    """
)

st.header("Configuration")
st.caption("Use the horizontal controls below. Units: inches")

st.markdown("<div class='card'>", unsafe_allow_html=True)
# Row 1: Population and generations + availability
r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1])
with r1c1:
    pop_size = st.number_input(
        "Population Size",
        min_value=40,
        max_value=500,
        value=140,
        step=10,
        help="How many designs to keep each generation",
    )
with r1c2:
    smpso_gens = st.number_input(
        "SMPSO Generations",
        min_value=50,
        max_value=1000,
        value=150,
        step=10,
        help="Exploration stage iterations (ignored if SMPSO not available)",
    )
with r1c3:
    nsga2_gens = st.number_input(
        "NSGA-II Generations",
        min_value=50,
        max_value=1500,
        value=250,
        step=10,
        help="Refinement stage iterations",
    )
with r1c4:
    # Availability banner removed per user request to keep the UI clean
    pass
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Design Variable Bounds / Fixed Values")
st.caption("Set min/max for each variable or tick 'Fix' to lock a single value.")
st.markdown("<div class='card'>", unsafe_allow_html=True)

# Row 2: Variables in four columns for a horizontal layout
vh, vl, vt, vb = st.columns(4)
with vh:
    fix_h = st.checkbox("Fix Height h (weld)", value=False, help="Tick to set a single value; leave unticked to set a range")
    if fix_h:
        h_val = st.number_input("h value", min_value=0.125, max_value=5.0, value=0.5, step=0.025, format="%.3f", help="Height of the weld seam")
        h_min, h_max = h_val, h_val
    else:
        h_min = st.number_input("h min", min_value=0.125, max_value=5.0, value=0.125, step=0.025, format="%.3f", help="Minimum weld height")
        h_max = st.number_input("h max", min_value=h_min, max_value=5.0, value=5.0, step=0.025, format="%.3f", help="Maximum weld height")
with vl:
    fix_l = st.checkbox("Fix Length l (weld)", value=False, help="Tick to set a single value; leave unticked to set a range")
    if fix_l:
        l_val = st.number_input("l value", min_value=0.1, max_value=10.0, value=5.0, step=0.1, format="%.3f", help="Length of the weld seam")
        l_min, l_max = l_val, l_val
    else:
        l_min = st.number_input("l min", min_value=0.1, max_value=10.0, value=0.1, step=0.1, format="%.3f", help="Minimum weld length")
        l_max = st.number_input("l max", min_value=l_min, max_value=10.0, value=10.0, step=0.1, format="%.3f", help="Maximum weld length")
with vt:
    fix_t = st.checkbox("Fix Thickness t (bar)", value=False, help="Tick to set a single value; leave unticked to set a range")
    if fix_t:
        t_val = st.number_input("t value", min_value=0.1, max_value=10.0, value=5.0, step=0.1, format="%.3f", help="Thickness of the bar")
        t_min, t_max = t_val, t_val
    else:
        t_min = st.number_input("t min", min_value=0.1, max_value=10.0, value=0.1, step=0.1, format="%.3f", help="Minimum bar thickness")
        t_max = st.number_input("t max", min_value=t_min, max_value=10.0, value=10.0, step=0.1, format="%.3f", help="Maximum bar thickness")
with vb:
    fix_b = st.checkbox("Fix Width b (bar)", value=False, help="Tick to set a single value; leave unticked to set a range")
    if fix_b:
        b_val = st.number_input("b value", min_value=0.1, max_value=10.0, value=5.0, step=0.1, format="%.3f", help="Width of the bar")
        b_min, b_max = b_val, b_val
    else:
        b_min = st.number_input("b min", min_value=0.1, max_value=10.0, value=0.1, step=0.1, format="%.3f", help="Minimum bar width")
        b_max = st.number_input("b max", min_value=b_min, max_value=10.0, value=10.0, step=0.1, format="%.3f", help="Maximum bar width")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Best Solution Preferences")
st.caption("Choose how important each objective is (the app recommends one design based on these)")
st.markdown("<div class='card'>", unsafe_allow_html=True)
w1, w2, w3, w4 = st.columns([1, 1, 1, 0.8])
with w1:
    w_cost = st.slider("Weight: Cost", 0.0, 1.0, 0.33, 0.01)
with w2:
    w_weight = st.slider("Weight: Weight", 0.0, 1.0, 0.33, 0.01)
with w3:
    w_weld = st.slider("Weight: Weld Area", 0.0, 1.0, 0.34, 0.01)
with w4:
    run_btn = st.button("Run Optimization", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

obj_names = WeldedBeamProblem.objective_names()
var_names = WeldedBeamProblem.var_names()
constr_names = WeldedBeamProblem.constraint_names()

if run_btn:
    xl = np.array([h_min, l_min, t_min, b_min], dtype=float)
    xu = np.array([h_max, l_max, t_max, b_max], dtype=float)
    problem = WeldedBeamProblem(xl=xl, xu=xu)
    # Quick feasibility tip for beginners
    if h_min > b_max:
        st.warning("Your minimum weld height h is greater than your maximum bar width b. Feasible designs may not exist (constraint h ≤ b). Consider increasing b or reducing h.")

    with st.spinner("Running optimization (SMPSO → NSGA-II when available)..."):
        res = run_hybrid(problem, pop_size=int(pop_size), smpso_gens=int(smpso_gens), nsga2_gens=int(nsga2_gens), seed=None)

    F = res["F"]
    X = res["X"]
    G = res["G"]
    F_all = res["F_all"]
    G_all = res["G_all"]

    # Compute Top 5 (by weighted normalized objectives) to highlight on plots
    highlight_F_top5 = None
    top5_rows_cache = None
    if len(F_all) > 0:
        feas_mask_all = (G_all <= 0).all(axis=1)
        if feas_mask_all.any():
            baseF2, baseX2, baseG2 = F_all[feas_mask_all], res["X_all"][feas_mask_all], G_all[feas_mask_all]
        else:
            baseF2, baseX2, baseG2 = F_all, res["X_all"], G_all

        eps = 1e-12
        fmin2 = np.min(baseF2, axis=0)
        fmax2 = np.max(baseF2, axis=0)
        rng2 = np.maximum(fmax2 - fmin2, eps)
        Fn2 = (baseF2 - fmin2) / rng2
        wv = np.array([w_cost, w_weight, w_weld], dtype=float)
        if wv.sum() <= 0:
            wv = np.array([1/3, 1/3, 1/3])
        else:
            wv = wv / wv.sum()
        scores2 = Fn2 @ wv
        k = int(min(5, len(scores2)))
        idxs = np.argsort(scores2)[:k]
        highlight_F_top5 = baseF2[idxs]
        # prepare rows cache for later display
        rows = []
        for i in idxs:
            row = {**{var_names[j]: float(baseX2[i, j]) for j in range(len(var_names))},
                   **{obj_names[j]: float(baseF2[i, j]) for j in range(len(obj_names))},
                   "Feasible": bool(np.all(baseG2[i] <= 0))}
            rows.append(row)
        top5_rows_cache = rows

    feas_mask = (G_all <= 0).all(axis=1)
    n_feas = int(feas_mask.sum())
    n_all = int(len(G_all)) if G_all is not None else 0

    st.subheader("Results Summary")
    cols = st.columns(3)
    cols[0].metric("Feasible Solutions", f"{n_feas} / {n_all}")
    if len(F) > 0:
        mins = np.min(F, axis=0)
        cols[1].write(pd.DataFrame({"Objective": obj_names, "Min Value": mins}))
    else:
        cols[1].warning("No feasible solutions. Displaying entire final population below.")

    # Plot Pareto front 3D and 2D projections
    st.subheader("Pareto Front")
    fig3d = pareto_3d_figure(F, obj_names, highlight=highlight_F_top5)
    st.plotly_chart(fig3d, use_container_width=True)

    c2 = st.columns(3)
    c2[0].plotly_chart(pareto_2d_figure(F, 0, 1, obj_names, highlight=highlight_F_top5), use_container_width=True)
    c2[1].plotly_chart(pareto_2d_figure(F, 0, 2, obj_names, highlight=highlight_F_top5), use_container_width=True)
    c2[2].plotly_chart(pareto_2d_figure(F, 1, 2, obj_names, highlight=highlight_F_top5), use_container_width=True)

    # Top solutions per objective
    st.subheader("Top Solutions per Objective (Best / Worst / Average)")
    tops = top_solutions_per_objective(F, X, obj_names, var_names)

    for name in obj_names:
        st.markdown(f"### {name}")
        t = tops[name]
        if t["best"] is None:
            st.info("No solutions available.")
            continue
        df = pd.DataFrame([
            {"Type": "Best", name: t["best"]["obj"], **t["best"]["x"]},
            {"Type": "Worst", name: t["worst"]["obj"], **t["worst"]["x"]},
            {"Type": "Average", name: t["average"]["obj"], **t["average"]["x"]},
        ])
        st.dataframe(df, use_container_width=True)

    # Recommended best solution based on weighted normalized objectives
    st.subheader("Recommended Best Solution")
    if len(F_all) > 0:
        # Prefer feasible points if any
        feas_mask = (G_all <= 0).all(axis=1)
        if feas_mask.any():
            baseF, baseX, baseG = F_all[feas_mask], res["X_all"][feas_mask], G_all[feas_mask]
        else:
            baseF, baseX, baseG = F_all, res["X_all"], G_all

        # Normalize objectives to [0,1]
        eps = 1e-12
        fmin = np.min(baseF, axis=0)
        fmax = np.max(baseF, axis=0)
        rng = np.maximum(fmax - fmin, eps)
        Fn = (baseF - fmin) / rng

        # Weights
        w = np.array([w_cost, w_weight, w_weld], dtype=float)
        if w.sum() <= 0:
            w = np.array([1/3, 1/3, 1/3])
        else:
            w = w / w.sum()
        scores = Fn @ w
        idx = int(np.argmin(scores))

        best_vars = {var_names[i]: float(baseX[idx, i]) for i in range(len(var_names))}
        best_objs = {obj_names[i]: float(baseF[idx, i]) for i in range(len(obj_names))}
        feas = bool(np.all(baseG[idx] <= 0))

        c1, c2 = st.columns(2)
        c1.write("Decision Variables")
        c1.dataframe(pd.DataFrame([best_vars]))
        c2.write("Objectives")
        c2.dataframe(pd.DataFrame([best_objs]))
        st.caption(f"Feasible: {feas}")
    else:
        st.info("No solutions available to recommend.")

    # Top 5 solutions by weighted normalized objective score (prefer feasible)
    st.subheader("Top 5 Solutions")
    if top5_rows_cache is not None and len(top5_rows_cache) > 0:
        st.dataframe(pd.DataFrame(top5_rows_cache), use_container_width=True)
    else:
        st.info("No solutions to display.")

    with st.expander("What is being optimized? (Beginner Friendly)"):
        st.markdown(
            """
            - **Cost**: Approximates welding and material costs. Lower is better.
            - **Weight**: Proportional to the steel volume. Lower is better.
            - **Weld Area**: 2 × h × l. Smaller weld areas can reduce fabrication effort.
            - The algorithm searches for designs that balance these goals while meeting constraints.
            """
        )

    with st.expander("Constraints explained"):
        st.markdown(
            """
            - **Shear stress (τ ≤ 13,600 psi)** and **Normal stress (σ ≤ 30,000 psi)**: strength limits.
            - **End deflection (δ ≤ 0.25 in)**: stiffness limit.
            - **Buckling (Pc ≥ P)**: prevents column buckling under load.
            - **Geometry (h ≤ b)**: weld height should not exceed bar width.
            """
        )


    # Raw feasible table
    st.subheader("Feasible Solutions (Objectives)")
    if len(F) > 0:
        dfF = pd.DataFrame(F, columns=obj_names)
        st.dataframe(dfF, use_container_width=True, height=260)

    st.subheader("Decision Variables (matching rows above)")
    if len(X) > 0:
        dfX = pd.DataFrame(X, columns=var_names)
        st.dataframe(dfX, use_container_width=True, height=260)
else:
    st.info("Configure parameters in the sidebar and click 'Run Optimization'.")

# --- Top section UI flow cards (like the provided mockup) ---
st.markdown(
    """
    <style>
    .card {background: #f7f9fc; border: 1px solid #e6eaf2; padding: 18px; border-radius: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);} 
    .card h3 {margin-top: 0;}
    .muted {color: #5f6c7b; font-size: 0.92rem;}
    .good {color: #0f7b6c;}
    .warn {color: #a05a00;}
    </style>
    """,
    unsafe_allow_html=True,
)

colL, colR = st.columns(2)
with colL:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Beam Parameters</h3>", unsafe_allow_html=True)
    # Show a concise current design summary (using fixed value or midpoint of range)
    def mid(a: float, b: float) -> float:
        return float((a + b) / 2.0)
    h_cur = float(h_min) if 'fix_h' in locals() and (h_min == h_max) else mid(float(h_min), float(h_max))
    l_cur = float(l_min) if 'fix_l' in locals() and (l_min == l_max) else mid(float(l_min), float(l_max))
    t_cur = float(t_min) if 'fix_t' in locals() and (t_min == t_max) else mid(float(t_min), float(t_max))
    b_cur = float(b_min) if 'fix_b' in locals() and (b_min == b_max) else mid(float(b_min), float(b_max))

    st.markdown(f"""
    <div class='muted'>Current Design Summary:</div>
    • Height h: <b>{h_cur:.3f}</b> in<br/>
    • Length l: <b>{l_cur:.3f}</b> in<br/>
    • Thickness t: <b>{t_cur:.3f}</b> in<br/>
    • Width b: <b>{b_cur:.3f}</b> in
    """, unsafe_allow_html=True)

    if st.button("Quick Evaluate Design"):
        # Evaluate single design (no optimization) to quickly see feasibility and objectives
        ptmp = WeldedBeamProblem()
        x0 = np.array([h_cur, l_cur, t_cur, b_cur], dtype=float)
        out = ptmp.evaluate(x0, return_values_of=["F", "G"], return_as_dictionary=True)
        F0, G0 = out["F"], out["G"]
        feas0 = bool(np.all(G0 <= 0))
        st.write("Objectives (lower is better):")
        st.dataframe(
            pd.DataFrame([{
                obj_names[0]: float(F0[0]),
                obj_names[1]: float(F0[1]),
                obj_names[2]: float(F0[2]),
            }])
        )
        st.write("Constraints g(x) ≤ 0 (negative is satisfied):")
        st.dataframe(
            pd.DataFrame([{
                "tau-13600": float(G0[0]),
                "sigma-30000": float(G0[1]),
                "delta-0.25": float(G0[2]),
                "P-Pc": float(G0[3]),
                "h-b": float(G0[4]),
            }])
        )
        st.success("Feasible" if feas0 else "Not Feasible")
    st.markdown("</div>", unsafe_allow_html=True)

with colR:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Optimization Process</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        - **Phase 1: SMPSO** – explores diverse designs.
        - **Phase 2: NSGA-II** – refines the Pareto front using crossover/mutation and non-dominated sorting.

        **Objectives to minimize**
        - Cost (material + welding)
        - Weight (total beam weight)
        - Weld Area (total welding area)

        **Subject to constraints**
        - τ ≤ 13,600 psi (shear)


        
        - σ ≤ 30,000 psi (normal)      
        - δ ≤ 0.25 in (deflection)
        - P_c ≥ P (buckling)
        - h ≤ b (geometry)
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)









