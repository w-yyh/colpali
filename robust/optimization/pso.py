"""PSO optimizer for continuous parameter search (maximization)."""
from __future__ import annotations
import numpy as np
from typing import Callable, List, Tuple
import pyswarms as ps


class PSOptimizer:
    """Maximize objective_fn over a bounded parameter space using PSO.

    Example:
        optimizer = PSOptimizer(bounds=[(1, 30), (0.1, 5)], n_particles=10, iters=20)
        best_params, best_score = optimizer.optimize(fn)
    """

    def __init__(self, bounds: List[Tuple[float, float]],
                 n_particles: int = 10, iters: int = 20,
                 options: dict | None = None):
        self.bounds = bounds
        self.n_particles = n_particles
        self.iters = iters
        self.options = options or {"c1": 0.5, "c2": 0.3, "w": 0.9}

    def optimize(self, objective_fn: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """Returns (best_params, best_score). Negate internally for minimization."""
        def _neg(particles): return np.array([-objective_fn(p) for p in particles])
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        opt = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, dimensions=len(self.bounds),
            options=self.options, bounds=(lb, ub))
        best_cost, best_pos = opt.optimize(_neg, iters=self.iters, verbose=False)
        return best_pos, float(-best_cost)
