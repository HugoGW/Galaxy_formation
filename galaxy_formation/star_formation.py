"""Star formation and supernova feedback (sub-grid models).

Individual stars are far below the resolution (one particle weighs ~10^5-10^6 Msun), so a
gas particle turns *as a whole* into a "star particle" representing a stellar population.

Star formation (Katz 1992; Springel & Hernquist 2003). Gas that is dense
(n_H > n_sf ~ 1 atom/cm^3, the density of a galactic disk) and cold (T < 3e4 K) forms stars
at the rate

    d rho_star / dt = eps_ff rho / t_ff,     t_ff = sqrt(3 pi / (32 G rho)),

with eps_ff ~ 0.5-1 % per free-fall time, as observed in molecular clouds and consistent
with the Kennicutt-Schmidt law Sigma_SFR ~ Sigma_gas^1.4 (Schmidt 1959; Kennicutt 1998;
Krumholz & Tan 2007). It is applied stochastically: during dt a dense particle becomes a
star with probability p = 1 - exp(-eps_ff dt / t_ff). Star particles feel only gravity.

Supernova feedback ("blast-wave" model, Stinson et al. 2006). A massive-star population
releases ~1e51 erg per supernova, i.e. ~1e49 erg per solar mass formed for a standard IMF.
A fraction fb_eff of this energy is given as heat to the fb_ngb nearest gas particles of
each new star, whose cooling is then switched off for fb_myr Myr (the lifetime of a
supernova bubble). Heated gas expands and stops forming stars: star formation regulates
itself and the disk keeps gas for billions of years.
"""

import numpy as np
from scipy.spatial import cKDTree

from .initial_conditions import GAS, STAR
from .units import MSUN, MSUN_G, MYR, U_CGS

T_SF_MAX = 3e4          # only cold gas forms stars (K)
E_SN_PER_MSUN = 1e49    # supernova energy per solar mass of stars formed (erg)


def form_stars(sim, dt):
    """Turns dense, cold gas into stars; returns the global indices of the new stars."""
    g = sim.gas_idx
    rho = sim.rho_gas
    dense = (rho > sim.rho_sf) & (sim.temperature(g) < T_SF_MAX)
    if not np.any(dense):
        return g[:0]
    t_ff = np.sqrt(3 * np.pi / (32 * rho[dense]))      # G = 1
    p = 1 - np.exp(-sim.cfg.sf_eff * dt / t_ff)
    new = g[dense][sim.rng.uniform(size=p.size) < p]
    sim.kind[new] = STAR
    sim.birth[new] = sim.t
    return new


def supernova_feedback(sim, new):
    """Heats the gas neighbours of the new star particles and blocks their cooling."""
    cfg = sim.cfg
    if cfg.fb_eff <= 0 or new.size == 0:
        return
    g = np.flatnonzero(sim.kind == GAS)
    if g.size < 10:
        return
    k = min(cfg.fb_ngb, g.size)
    _, nb = cKDTree(sim.pos[g]).query(sim.pos[new], k=k, workers=-1)
    targets = g[np.asarray(nb).reshape(len(new), k)]                     # (n_new, k)
    e_per_g = cfg.fb_eff * E_SN_PER_MSUN / MSUN_G                        # erg per gram of stars formed
    E = e_per_g * sim.mass[new] * MSUN * MSUN_G                          # erg per new star
    du = E[:, None] / k / (sim.mass[targets] * MSUN * MSUN_G) / U_CGS     # code units
    np.add.at(sim.u, targets.ravel(), du.ravel())
    sim.no_cool_until[targets.ravel()] = sim.t + cfg.fb_myr / MYR
