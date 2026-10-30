"""The simulation: particle state, forces and time integration.

Three kinds of particles share the same arrays (see initial_conditions.py):
gas (SPH + gravity), stars (gravity only, born from gas) and dark matter (gravity only).

Time integration is a kick-drift-kick leapfrog (second order, symplectic for gravity) with
two levels of hierarchical time steps (in the spirit of GADGET-2, Springel 2005):

  * each particle has an admissible step: 0.3 sqrt(eps / |a|) for gravity, plus the Courant
    condition for gas;
  * the long step D is the 10th percentile of these steps. Particles that can afford D
    ("slow", ~90 % of them, mostly dark matter in the outer halo) are kicked once per D;
  * gas and "fast" particles do sub-steps inside D. During sub-steps only their gravity is
    recomputed, on a tree that is refitted rather than rebuilt.

Each particle therefore respects its own accuracy criterion, at a fraction of the cost of
a single global step set by the fastest particle.
"""

import numpy as np

from . import cooling, star_formation
from .engine import Engine
from .initial_conditions import DM, GAS, STAR, make_cloud
from .sph import ALPHA, BETA, NeighbourFinder, sph_forces
from .units import K_B, KPC, M_H, MSUN, MU_IONISED, MYR, RHO_CGS, U_CGS, X_H


class GalaxyFormation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = Engine()
        self.rng = np.random.default_rng(cfg.seed)
        self.pos, self.vel, self.mass, self.kind = make_cloud(cfg, self.rng)
        n = len(self.mass)
        self.birth = np.full(n, np.inf)                 # birth time of star particles

        # internal energy: T = (gamma - 1) mu m_H u / k_B
        self.gamma = 5.0 / 3.0
        self.u_per_K = K_B / ((self.gamma - 1) * MU_IONISED * M_H) / U_CGS
        self.u = np.full(n, cfg.t_gas * self.u_per_K)
        self.u_floor = cooling.T_FLOOR * self.u_per_K
        self.no_cool_until = np.full(n, -np.inf)        # supernova bubbles: no cooling until then

        self.eps = cfg.eps / KPC
        self.rho_sf = cfg.n_sf * M_H / X_H / RHO_CGS    # star-formation threshold (code units)
        self.t = 0.0
        self.history = []                               # (t in Myr, SFR in Msun/yr, stellar fraction)

        self.neighbours = NeighbourFinder(n)
        self.acc_grav = np.zeros_like(self.pos)
        self.acc_sph = np.zeros_like(self.pos)
        self.dt_courant = np.full(n, np.inf)
        self.acc = self.accelerations()

    # -- physics -------------------------------------------------------------------------

    def temperature(self, idx):
        return self.u[idx] / self.u_per_K

    def accelerations(self, active=None, rebuild=True, sph=True, fresh_nb=True):
        """Gravity (Barnes-Hut) of the active particles (all by default) plus SPH forces on
        the gas. Also updates the admissible time step of every particle (self.dt_ind)."""
        self.engine.gravity(self.pos, self.mass, self.acc_grav, self.cfg.theta, self.eps,
                            active, rebuild)
        if sph:
            self._hydro(fresh_nb)
        acc = self.acc_grav + self.acc_sph
        a_mag = np.sqrt(np.einsum("ij,ij->i", acc, acc))
        self.dt_ind = np.minimum(0.3 * np.sqrt(self.eps / np.maximum(a_mag, 1e-30)), self.dt_courant)
        self.dt_grav = 0.3 * np.sqrt(self.eps / max(a_mag.max(), 1e-30))
        return acc

    def _hydro(self, fresh_nb):
        self.acc_sph[:] = 0.0
        self.dt_courant[:] = np.inf
        self.dt_gas = np.inf
        self.rho_gas = np.zeros(0)
        g = np.flatnonzero(self.kind == GAS)
        if g.size <= 40:
            return
        p = np.ascontiguousarray(self.pos[g])
        d, nb = self.neighbours.find(g, p, fresh_nb)
        h = np.maximum(0.5 * d[:, -1], 0.5 * self.eps)     # smoothing length >= eps / 2
        rho = np.empty(g.size)
        a = np.empty((g.size, 3))
        dudt = np.empty(g.size)
        dt_i = np.empty(g.size)
        sph_forces(p, np.ascontiguousarray(self.vel[g]), np.ascontiguousarray(self.mass[g]),
                   nb, h, np.ascontiguousarray(self.u[g]), self.gamma, ALPHA, BETA,
                   rho, a, dudt, dt_i)
        self.acc_sph[g] = a
        self.dt_courant[g] = dt_i
        self.dt_gas = dt_i.min()
        self.gas_idx, self.rho_gas, self.dudt = g, rho, dudt

    def _thermodynamics(self, dt):
        """Compression / shock heating (explicit), then radiative cooling."""
        if self.rho_gas.size == 0:
            return
        g = self.gas_idx
        self.u[g] = np.maximum(self.u[g] + dt * self.dudt, self.u_floor)
        self.u[g] = cooling.cool(self.u[g], self.temperature(g), self.rho_gas, dt,
                                 self.no_cool_until[g] > self.t, self.u_floor)

    def _stars(self, dt):
        if self.rho_gas.size == 0:
            return
        new = star_formation.form_stars(self, dt)
        star_formation.supernova_feedback(self, new)

    # -- integration ---------------------------------------------------------------------

    def step(self, dt_max):
        """One long step (at most dt_max) of the hierarchical KDK leapfrog. Returns its length."""
        dt_min = min(self.dt_gas, self.dt_grav, dt_max)
        D = min(dt_max, np.percentile(self.dt_ind, 10))
        if D <= 1.5 * dt_min:
            # nothing to gain from sub-stepping: classic global step
            self._thermodynamics(dt_min)
            self.vel += 0.5 * dt_min * self.acc
            self.pos += dt_min * self.vel
            self.acc = self.accelerations()
            self.vel += 0.5 * dt_min * self.acc
            self.t += dt_min
            self._stars(dt_min)
            return dt_min

        fast = (self.dt_ind < D) | (self.kind == GAS)
        slow = ~fast
        self.vel[slow] += 0.5 * D * self.acc[slow]          # opening kick of slow particles
        tau = 0.0
        first = True
        while tau < D * (1 - 1e-9):
            dt = min(self.dt_ind[fast].min(), D - tau)
            self._thermodynamics(dt)
            self.vel[fast] += 0.5 * dt * self.acc[fast]
            self.pos += dt * self.vel                       # everybody drifts
            # gas neighbours: full search on the first sub-step, Verlet list afterwards
            self.acc = self.accelerations(active=fast, rebuild=False, fresh_nb=first)
            first = False
            self.vel[fast] += 0.5 * dt * self.acc[fast]
            tau += dt
            self.t += dt
            self._stars(dt)
        # end of the long step: new tree, gravity of the slow particles, closing kick
        self.acc = self.accelerations(active=slow, rebuild=True, sph=False)
        self.vel[slow] += 0.5 * D * self.acc[slow]
        return D

    def advance_to(self, t_end):
        """Integrates up to t_end (code units) and records the star-formation rate."""
        m0 = self.stellar_mass()
        t0 = self.t
        while self.t < t_end - 1e-9:
            self.step(t_end - self.t)
        dm = self.stellar_mass() - m0
        sfr = dm * MSUN / ((self.t - t0) * MYR * 1e6) if self.t > t0 else 0.0
        baryons = self.mass[self.kind != DM].sum()
        self.history.append((self.t * MYR, sfr, self.stellar_mass() / baryons))

    def stellar_mass(self):
        return self.mass[self.kind == STAR].sum()
