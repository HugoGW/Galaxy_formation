"""Radiative cooling of a primordial (H + He, no metals) ionised gas.

Hot gas radiates and loses internal energy at the rate

    du/dt = -Lambda(T) n_H^2 / rho        (erg/g/s)

where Lambda(T) is the cooling function. We use the fits of Cen (1992) for collisional
excitation of H and He+ (the two peaks near 2e4 K and 1e5 K) plus thermal bremsstrahlung
(free-free emission, dominant above ~1e7 K); see also Katz, Weinberg & Hernquist (1996).

Below 1e4 K a primordial gas without molecules or metals barely cools, so the temperature
is floored there. This is the classical picture of White & Rees (1978) and
Fall & Efstathiou (1980): gas falling into a halo is shock-heated, cools if its cooling
time is shorter than the free-fall time, and sinks to the centre.
"""

import numpy as np

from .units import M_H, MYR, RHO_CGS, SEC_PER_MYR, U_CGS, X_H

T_FLOOR = 1e4


def cooling_function(T):
    """Lambda(T) in erg cm^3 / s (to be multiplied by n_H^2)."""
    T = np.maximum(T, 1.0)
    f = 1 / (1 + np.sqrt(T / 1e5))
    lam = (7.5e-19 * np.exp(-118348 / T) * f                     # H collisional excitation
           + 5.54e-17 * T ** -0.397 * np.exp(-473638 / T) * f   # He+ collisional excitation
           + 1.42e-27 * 1.1 * np.sqrt(T))                       # bremsstrahlung
    return np.where(T > T_FLOOR, lam, 0.0)


def cool(u, T, rho, dt, blocked, u_floor):
    """New internal energy after a step dt (code units). The energy removed in one step is
    bounded by the gap to the floor, so the explicit update cannot overshoot. Particles with
    blocked=True (inside a supernova bubble) do not cool."""
    rho_cgs = rho * RHO_CGS
    nH = X_H * rho_cgs / M_H
    dudt_cgs = cooling_function(T) * nH ** 2 / rho_cgs        # erg/g/s
    du = dudt_cgs * dt * MYR * SEC_PER_MYR / U_CGS             # code units
    du[blocked] = 0.0
    return np.maximum(u - du, u_floor)
