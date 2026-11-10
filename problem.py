# Welded Beam Multi-Objective Optimization (Cost, Weight, Weld Area)
# Formulation adapted from the classical welded beam benchmark.
# Variables: x = [h, l, t, b]
#   h: weld thickness (in)
#   l: weld length (in)
#   t: beam thickness (in)
#   b: beam width (in)
# Objectives (to minimize):
#   f1: cost (approx welding + material cost proxy)
#   f2: weight (volume proxy)
#   f3: weld area (2 * h * l)
# Constraints:
#   - Shear stress tau <= 13,600 psi
#   - Normal stress sigma <= 30,000 psi
#   - End deflection delta <= 0.25 in
#   - Buckling P_c >= P (rewrite as P - P_c <= 0)
#   - Geometric: b - h >= 0
# Bounds:
#   h in [0.125, 5.0], l in [0.1, 10.0], t in [0.1, 10.0], b in [0.1, 10.0]

from __future__ import annotations
import numpy as np
from pymoo.core.problem import Problem


class WeldedBeamProblem(Problem):
    def __init__(self, xl: np.ndarray | None = None, xu: np.ndarray | None = None):
        xl_def = np.array([0.125, 0.1, 0.1, 0.1], dtype=float)
        xu_def = np.array([5.0, 10.0, 10.0, 10.0], dtype=float)
        self._xl = np.array(xl, dtype=float) if xl is not None else xl_def
        self._xu = np.array(xu, dtype=float) if xu is not None else xu_def

        super().__init__(
            n_var=4,
            n_obj=3,
            n_constr=5,
            xl=self._xl,
            xu=self._xu,
            elementwise_evaluation=False,
        )
        # Constants (imperial units)
        self.P = 6000.0  # lb
        self.L = 14.0  # in (overhang)
        self.E = 30e6  # psi
        self.G = 12e6  # psi
        self.tau_max = 13600.0  # psi
        self.sigma_max = 30000.0  # psi
        self.delta_max = 0.25  # in

    def _evaluate(self, x, out, *args, **kwargs):
        # Support both elementwise and vectorized evaluation across pymoo versions
        X = np.atleast_2d(x).astype(float)

        h, l, t, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        P = self.P
        L = self.L
        E = self.E
        Gs = self.G

        # Derived quantities for stresses (classical formulation)
        M = P * (L + l / 2.0)
        R = np.sqrt(l**2 / 4.0 + (h + t / 2.0) ** 2)
        # Polar moment approximation for the weld group
        J = 2.0 * (np.sqrt(2.0) * h * l * (l**2 / 12.0 + (h + t / 2.0) ** 2))

        # Avoid division by zero
        eps = 1e-12
        h_l = np.maximum(h * l, eps)
        J = np.maximum(J, eps)
        t2b = np.maximum(t**2 * b, eps)
        t3b = np.maximum(t**3 * b, eps)

        # Shear components and resultant
        tau_prime = P / (np.sqrt(2.0) * h_l)
        tau_double_prime = M * R / J
        tau = np.sqrt(
            tau_prime**2
            + 2.0 * tau_prime * tau_double_prime * (l / (2.0 * np.maximum(R, eps)))
            + tau_double_prime**2
        )

        # Normal stress on the bar cross-section
        sigma = 6.0 * P * L / t2b

        # End deflection of the bar
        delta = 4.0 * P * L**3 / (E * t3b)

        # Buckling load of the bar (column buckling approximation)
        Pc = (
            4.013 * E * np.sqrt(np.maximum(t**2 * b**6 / 36.0, eps)) / (L**2)
            * (1.0 - t / (2.0 * L) * np.sqrt(E / (4.0 * Gs)))
        )

        # Objectives
        cost = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
        weight = t * b * (14.0 + l)
        weld_area = 2.0 * h * l

        # Constraints g(x) <= 0
        g1 = tau - self.tau_max
        g2 = sigma - self.sigma_max
        g3 = delta - self.delta_max
        g4 = P - Pc
        g5 = h - b

        F = np.column_stack([cost, weight, weld_area])
        G = np.column_stack([g1, g2, g3, g4, g5])

        # Return shape according to input shape
        if np.ndim(x) == 1:
            out["F"] = F[0]
            out["G"] = G[0]
        else:
            out["F"] = F
            out["G"] = G

    @staticmethod
    def objective_names():
        return ["Cost", "Weight", "Weld Area"]

    @staticmethod
    def var_names():
        return ["h (weld)", "l (length)", "t (thickness)", "b (width)"]

    @staticmethod
    def constraint_names():
        return [
            "tau - 13600 <= 0",
            "sigma - 30000 <= 0",
            "delta - 0.25 <= 0",
            "P - Pc <= 0",
            "h - b <= 0",
        ]
