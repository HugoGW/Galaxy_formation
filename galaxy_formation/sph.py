"""Smoothed Particle Hydrodynamics (SPH) for the gas.

Each gas particle carries a mass m, a velocity v and a specific internal energy u. Fluid
quantities are kernel-weighted sums over the 32 nearest neighbours (Monaghan 1992, 2005;
Springel 2010 for a review):

    density      rho_i = sum_j m_j W(r_ij, h_i)
    pressure     P_i = (gamma - 1) rho_i u_i            (ideal monatomic gas, gamma = 5/3)
    momentum     dv_i/dt = -sum_j m_j [P_i/rho_i^2 grad W(h_i) + P_j/rho_j^2 grad W(h_j)
                                       + Pi_ij grad W_avg]
    energy       du_i/dt = sum_j m_j [P_i/rho_i^2 grad W(h_i) + Pi_ij/2 grad W_avg] . v_ij

W is the cubic-spline kernel and h_i is half the distance to the 32nd neighbour, floored at
half the gravitational softening (no hydro structure below the gravity resolution).

Shocks are captured with the Monaghan & Gingold (1983) artificial viscosity (alpha = 1,
beta = 2), active only for approaching pairs. It turns the kinetic energy of colliding
gas streams into heat, which radiative cooling then removes: the gas loses energy but keeps
its angular momentum, and settles into a rotating disk.
"""

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

N_NEIGHBOURS = 32
N_CANDIDATES = 48   # Verlet-list size: candidates kept between two full searches
ALPHA, BETA = 1.0, 2.0


@njit(cache=True, fastmath=True)
def kernel(r, h):
    """Cubic spline (M4) kernel with support 2h (Monaghan & Lattanzio 1985)."""
    q = r / h
    s = 1.0 / (np.pi * h ** 3)
    if q < 1.0:
        return s * (1 - 1.5 * q * q + 0.75 * q ** 3)
    if q < 2.0:
        return s * 0.25 * (2 - q) ** 3
    return 0.0


@njit(cache=True, fastmath=True)
def kernel_grad(r, h):
    """(1/r) dW/dr: multiply by the separation vector to get grad W."""
    q = r / h
    s = 1.0 / (np.pi * h ** 4)
    if q < 1.0:
        g = s * (-3 * q + 2.25 * q * q)
    elif q < 2.0:
        g = -s * 0.75 * (2 - q) ** 2
    else:
        return 0.0
    return g / max(r, 1e-30)


@njit(parallel=True, cache=True, fastmath=True)
def sph_forces(pos, vel, mass, nb, h, u, gamma, alpha, beta, rho, acc, dudt, dt_i):
    """Density, pressure + viscosity acceleration, du/dt and Courant time step of every gas
    particle. nb holds the neighbour indices (the particle itself included). Outputs: rho,
    acc, dudt, dt_i (written in place)."""
    n = nb.shape[0]
    for i in prange(n):
        s = 0.0
        for jj in range(nb.shape[1]):
            j = nb[i, jj]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            s += mass[j] * kernel(np.sqrt(dx * dx + dy * dy + dz * dz), h[i])
        rho[i] = s
    for i in prange(n):
        Pi = (gamma - 1) * rho[i] * u[i]
        ci = np.sqrt(gamma * (gamma - 1) * u[i])          # sound speed
        ax = ay = az = 0.0
        du = 0.0
        mu_max = 0.0
        for jj in range(nb.shape[1]):
            j = nb[i, jj]
            if j == i:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            gi = kernel_grad(r, h[i])
            gj = kernel_grad(r, h[j])
            Pj = (gamma - 1) * rho[j] * u[j]
            cj = np.sqrt(gamma * (gamma - 1) * u[j])
            vr = (vel[i, 0] - vel[j, 0]) * dx + (vel[i, 1] - vel[j, 1]) * dy + (vel[i, 2] - vel[j, 2]) * dz
            visc = 0.0
            if vr < 0:                                      # approaching pair: shock viscosity
                hb = 0.5 * (h[i] + h[j])
                mu = hb * vr / (r * r + 0.01 * hb * hb)
                mu_max = max(mu_max, -mu)
                visc = (-alpha * 0.5 * (ci + cj) * mu + beta * mu * mu) / (0.5 * (rho[i] + rho[j]))
            gb = 0.5 * (gi + gj)
            f = Pi / rho[i] ** 2 * gi + Pj / rho[j] ** 2 * gj + visc * gb
            ax -= mass[j] * f * dx
            ay -= mass[j] * f * dy
            az -= mass[j] * f * dz
            du += mass[j] * (Pi / rho[i] ** 2 * gi + 0.5 * visc * gb) * vr
        acc[i, 0], acc[i, 1], acc[i, 2] = ax, ay, az
        dudt[i] = du
        # Courant condition including the viscous signal speed (Monaghan 1992)
        dt_i[i] = 0.3 * h[i] / (ci + 1.2 * (alpha * ci + beta * mu_max) + 1e-30)


@njit(parallel=True, cache=True, fastmath=True)
def knn_from_candidates(pos, cand, k, nb, dist):
    """k nearest neighbours of each particle among its candidates (local indices, -1 = none).
    Returns False if some particle does not have enough candidates."""
    n, m = cand.shape
    ok = True
    for i in prange(n):
        d2 = np.empty(m)
        cnt = 0
        for jj in range(m):
            j = cand[i, jj]
            if j < 0:
                d2[jj] = np.inf
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d2[jj] = dx * dx + dy * dy + dz * dz
            cnt += 1
        if cnt < k:
            ok = False
            continue
        order = np.argsort(d2)
        for jj in range(k):
            nb[i, jj] = cand[i, order[jj]]
            dist[i, jj] = np.sqrt(d2[order[jj]])
    return ok


class NeighbourFinder:
    """32 nearest gas neighbours of every gas particle.

    A full search (k-d tree, 48 candidates) is done when `fresh` is True. Otherwise the 32
    nearest are re-selected among the candidates remembered from the last full search (a
    Verlet list), which is much faster while particles have moved little (sub-steps).
    Indices are global particle indices in the cache, so the list survives gas particles
    turning into stars between two searches."""

    def __init__(self, n_total):
        self.n_total = n_total
        self.cache = None

    def find(self, g, p, fresh):
        """g: global indices of gas particles, p: their positions. Returns (dist, nb) with nb in
        local (0..len(g)-1) indices."""
        k = N_NEIGHBOURS
        if not fresh and self.cache is not None:
            row = self.cache["row"][g]
            if (row >= 0).all():
                loc = np.full(self.n_total, -1, np.int64)
                loc[g] = np.arange(g.size)
                cand = loc[self.cache["cand"][row]]      # candidates that are still gas
                nb = np.empty((g.size, k), np.int64)
                d = np.empty((g.size, k))
                if knn_from_candidates(p, cand, k, nb, d):
                    return d, nb
        kq = min(N_CANDIDATES, g.size)
        d, nb = cKDTree(p).query(p, k=kq, workers=-1)
        row = np.full(self.n_total, -1, np.int64)
        row[g] = np.arange(g.size)
        self.cache = {"row": row, "cand": g[nb]}
        return d[:, :k], np.ascontiguousarray(nb[:, :k])
