"""Initial conditions: a cold, slowly rotating, randomly filled cloud of gas and dark matter.

The cloud is a sphere of radius R0 in which gas (fraction f_b of the mass) and dark matter
are mixed and placed at random. Two density profiles are available:

    extended  rho ~ 1/r   ->  M(<r) ~ r^2   (default)
    uniform   rho = cst   ->  M(<r) ~ r^3

With the extended profile, the centre is much denser than the outskirts: its free-fall time
t_ff ~ 1/sqrt(G rho) is short, so it collapses first while the outer shells, which carry most
of the angular momentum, rain down later. The galaxy is built from the inside out.

Rotation is set by the dimensionless spin parameter lambda (Peebles 1969; Bullock et al. 2001),
typically 0.035-0.05 for dark-matter halos formed in cosmological simulations. The random
positions are Poisson noise: their density fluctuations are the seeds of the first clumps.
"""

import numpy as np

from .units import KPC, MSUN

GAS, STAR, DM = 0, 1, 2      # particle kinds


def random_ball(n, R0, exponent, rng):
    """n points in a sphere of radius R0 with M(<r) ~ r^(1/exponent).
    exponent = 1/3 gives a uniform ball, exponent = 1/2 gives rho ~ 1/r."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * R0 * rng.uniform(0, 1, n)[:, None] ** exponent


def make_cloud(cfg, rng):
    """Returns pos, vel, mass, kind (code units). Gas comes first, then dark matter;
    gas particles later turn into stars in place, so baryons always come first."""
    ng, nd = cfg.n_gas, cfg.n_dm
    M_tot = cfg.mass / MSUN
    R0 = cfg.r0 / KPC
    exponent = 1 / 3 if cfg.profile == "uniform" else 1 / 2

    pos = np.ascontiguousarray(np.concatenate([random_ball(ng, R0, exponent, rng),
                                               random_ball(nd, R0, exponent, rng)]))
    mass = np.concatenate([np.full(ng, cfg.f_baryon * M_tot / ng),
                           np.full(nd, (1 - cfg.f_baryon) * M_tot / nd)])
    kind = np.concatenate([np.full(ng, GAS, np.int8), np.full(nd, DM, np.int8)])

    ez = np.array([0.0, 0.0, 1.0])
    r = np.linalg.norm(pos, axis=1)
    if cfg.profile == "uniform":
        # solid-body rotation. For a uniform sphere J = 2/5 M R^2 Omega and E = -3 G M^2 / (5 R);
        # lambda = J |E|^1/2 / (G M^5/2) (Peebles 1969) gives Omega below.
        Om = cfg.spin * np.sqrt(M_tot) / (0.4 * np.sqrt(0.6) * R0 ** 1.5)
        vel = np.cross(Om * ez, pos)
    else:
        # v_phi = kappa v_c(r) R_cyl / r with v_c = sqrt(G M(<r) / r) and M(<r) = M (r/R0)^2.
        # The Bullock spin lambda' = j / (sqrt(2) R V) of this field is ~ 0.47 kappa.
        kappa = cfg.spin / 0.47
        vc = np.sqrt(M_tot * (r / R0) ** 2 / np.maximum(r, 1e-6))
        phi_hat = np.cross(ez, pos)
        phi_hat /= np.linalg.norm(phi_hat, axis=1, keepdims=True) + 1e-30
        R_cyl = np.hypot(pos[:, 0], pos[:, 1])
        vel = (kappa * vc * R_cyl / np.maximum(r, 1e-6))[:, None] * phi_hat

    # nearly cold cloud: small random motions, 10 % of the virial velocity
    v_vir = np.sqrt(M_tot / R0)
    vel += 0.1 * v_vir * rng.normal(size=pos.shape) / np.sqrt(3)
    vel = np.ascontiguousarray(vel - np.average(vel, axis=0, weights=mass))  # no bulk motion
    return pos, vel, mass, kind
