# tests/test_pso.py
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import numpy as np

def test_pso_finds_near_optimum():
    from robust.optimization.pso import PSOptimizer
    optimizer = PSOptimizer(bounds=[(0, 10), (0, 10)], n_particles=8, iters=10)
    # Maximize -(x-3)^2 - (y-7)^2, optimum at (3,7) -> score near 0
    def objective(p): return -((p[0]-3)**2 + (p[1]-7)**2)
    best_params, best_score = optimizer.optimize(objective)
    assert best_score > -5.0, f"Expected score near 0, got {best_score}"
    assert all(0 <= v <= 10 for v in best_params)

def test_pso_types():
    from robust.optimization.pso import PSOptimizer
    opt = PSOptimizer(bounds=[(0, 5)], n_particles=4, iters=5)
    params, score = opt.optimize(lambda p: -p[0]**2)
    assert isinstance(params, np.ndarray)
    assert isinstance(score, float)
