"""Quick sanity checks of the numerical building blocks.

    python -m pytest tests          (or simply: python tests/test_physics.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from galaxy_formation import Config, GalaxyFormation          # noqa: E402
from galaxy_formation.cooling import cooling_function          # noqa: E402
from galaxy_formation.engine import Engine                     # noqa: E402
from galaxy_formation.rendering import rotation_to_z           # noqa: E402
from galaxy_formation.sph import kernel                        # noqa: E402


def test_barnes_hut_matches_direct_sum():
    """Barnes-Hut (theta = 0.7) vs exact O(N^2) forces: median relative error < 1 %."""
    rng = np.random.default_rng(0)
    pos = np.ascontiguousarray(rng.normal(size=(20000, 3)))
    mass = np.full(len(pos), 1.0 / len(pos))
    eng = Engine()
    acc = np.zeros_like(pos)
    eng.gravity(pos, mass, acc, theta=0.7, eps=0.01)
    exact = eng.gravity_direct(pos, mass, eps=0.01)
    err = np.linalg.norm(acc - exact, axis=1) / np.linalg.norm(exact, axis=1)
    assert np.median(err) < 1e-2


def test_kernel_is_normalised():
    """The cubic spline integrates to 1 over space."""
    h = 0.7
    r = np.linspace(0, 2 * h, 20001)
    w = np.array([kernel(x, h) for x in r])
    assert abs(np.trapz(4 * np.pi * r ** 2 * w, r) - 1) < 1e-4


def test_cooling_function():
    """No cooling below the 1e4 K floor, positive and finite above it."""
    lam = cooling_function(np.array([5e3, 2e4, 1e5, 1e6, 1e8]))
    assert lam[0] == 0
    assert np.all(np.isfinite(lam)) and np.all(lam[1:] > 0)


def test_rotation_to_z():
    J = np.array([0.3, -0.5, 0.8])
    J /= np.linalg.norm(J)
    R = rotation_to_z(J)
    assert np.allclose(R @ J, [0, 0, 1])
    assert np.allclose(R @ R.T, np.eye(3))


def test_initial_cloud():
    """Mass, baryon fraction, radius and spin of the initial cloud."""
    cfg = Config(n_gas=2000, n_dm=2000)
    sim = GalaxyFormation(cfg)
    assert np.isclose(sim.mass.sum() * 4e10, cfg.mass)
    assert np.isclose(sim.mass[: cfg.n_gas].sum() / sim.mass.sum(), cfg.f_baryon)
    assert np.linalg.norm(sim.pos, axis=1).max() * 3.5 <= cfg.r0 + 1e-9
    J = np.sum(sim.mass[:, None] * np.cross(sim.pos, sim.vel), axis=0)
    assert J[2] > 0 and abs(J[2]) > 5 * np.hypot(J[0], J[1])   # spinning about z


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok ", name)
