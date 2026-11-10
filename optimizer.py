from __future__ import annotations
import numpy as np
from typing import Dict, Any

from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
SMPSO = None
HAS_SMPSO = False
try:
    # Common location in many 0.6.x releases
    from pymoo.algorithms.moo.smpso import SMPSO as _SMPSO  # type: ignore
    SMPSO = _SMPSO
    HAS_SMPSO = True
except Exception:
    try:
        # Some builds may place it under pso
        from pymoo.algorithms.moo.pso import SMPSO as _SMPSO  # type: ignore
        SMPSO = _SMPSO
        HAS_SMPSO = True
    except Exception:
        # SMPSO not available - will fallback to NSGA-II only
        HAS_SMPSO = False
from pymoo.termination import get_termination


def _feasible_mask(G: np.ndarray) -> np.ndarray:
    if G is None:
        return np.ones(0, dtype=bool)
    # Feasible if all constraints <= 0
    return np.all(G <= 0.0, axis=1)


def run_hybrid(problem, pop_size: int = 120, smpso_gens: int = 150, nsga2_gens: int = 200, seed: int | None = None) -> Dict[str, Any]:
    """
    Pipeline:
    1) Run SMPSO to explore/search.
    2) Seed NSGA-II with SMPSO population to refine the Pareto front.

    Returns a dict with keys: 'F', 'X', 'G', 'F_all', 'X_all', 'G_all', 'res_nsga2'.
    """
    # ---- Stage 1: SMPSO (if available) ----
    X_seed = None
    if HAS_SMPSO and SMPSO is not None:
        smpso = SMPSO(pop_size=pop_size, sampling=FloatRandomSampling())
        term_smpso = get_termination("n_gen", smpso_gens)
        res_smpso = minimize(
            problem,
            smpso,
            termination=term_smpso,
            seed=seed,
            save_history=False,
            verbose=False,
        )
        X_seed = res_smpso.pop.get("X") if hasattr(res_smpso, "pop") else getattr(res_smpso, "X", None)

    # ---- Stage 2: NSGA-II (seeded if SMPSO ran) ----
    nsga2 = NSGA2(pop_size=pop_size, sampling=(X_seed if X_seed is not None else FloatRandomSampling()))
    term_nsga2 = get_termination("n_gen", nsga2_gens)
    res_nsga2 = minimize(
        problem,
        nsga2,
        termination=term_nsga2,
        seed=seed,
        save_history=False,
        verbose=False,
    )

    # Extract final population
    pop = res_nsga2.pop if hasattr(res_nsga2, "pop") else Population.new(X=res_nsga2.X)
    F_all = pop.get("F")
    X_all = pop.get("X")
    G_all = pop.get("G")

    # Keep feasible points as Pareto set approximation for display
    feas_mask = _feasible_mask(G_all)
    if feas_mask.size == 0:
        F, X, G = F_all, X_all, G_all
    else:
        if np.any(feas_mask):
            F, X, G = F_all[feas_mask], X_all[feas_mask], G_all[feas_mask]
        else:
            # No feasible solutions: return entire population and let UI indicate infeasibility
            F, X, G = F_all, X_all, G_all

    return {
        "F": F,
        "X": X,
        "G": G,
        "F_all": F_all,
        "X_all": X_all,
        "G_all": G_all,
        "res_nsga2": res_nsga2,
    }


def top_solutions_per_objective(F: np.ndarray, X: np.ndarray, obj_names: list[str], var_names: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    For each objective, report best (min), worst (max), and average solution among provided points.
    Returns a dict per objective with entries for 'best', 'worst', 'average', each including objective value and variables.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if F is None or F.size == 0:
        for name in obj_names:
            results[name] = {
                "best": None,
                "worst": None,
                "average": None,
            }
        return results

    for j, name in enumerate(obj_names):
        vals = F[:, j]
        best_idx = int(np.argmin(vals))
        worst_idx = int(np.argmax(vals))
        avg_val = float(np.mean(vals))

        results[name] = {
            "best": {"obj": float(vals[best_idx]), "x": {vn: float(X[best_idx, i]) for i, vn in enumerate(var_names)}},
            "worst": {"obj": float(vals[worst_idx]), "x": {vn: float(X[worst_idx, i]) for i, vn in enumerate(var_names)}},
            "average": {"obj": avg_val, "x": {vn: float(np.mean(X[:, i])) for i, vn in enumerate(var_names)}},
        }
    return results


def top_solutions_per_constraint(G: np.ndarray, X: np.ndarray, constr_names: list[str], var_names: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    For each constraint g_k(x) <= 0, report:
    - best: minimal g (most satisfied)
    - worst: maximal g (largest violation)
    - average: mean g and the solution closest to the mean
    Returns a dict per constraint similar to objectives.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if G is None or G.size == 0:
        for name in constr_names:
            results[name] = {"best": None, "worst": None, "average": None}
        return results

    for j, name in enumerate(constr_names):
        vals = G[:, j]
        best_idx = int(np.argmin(vals))
        worst_idx = int(np.argmax(vals))
        avg_val = float(np.mean(vals))
        avg_idx = int(np.argmin(np.abs(vals - avg_val)))

        results[name] = {
            "best": {"g": float(vals[best_idx]), "x": {vn: float(X[best_idx, i]) for i, vn in enumerate(var_names)}},
            "worst": {"g": float(vals[worst_idx]), "x": {vn: float(X[worst_idx, i]) for i, vn in enumerate(var_names)}},
            "average": {"g": avg_val, "x": {vn: float(X[avg_idx, i]) for i, vn in enumerate(var_names)}},
        }
    return results
