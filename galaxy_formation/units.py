"""Unit system.

The code works with G = 1. Physical units are fixed by choosing a length and a mass unit
(those of a Milky-Way-like disk); velocity and time units follow:

    length   1 unit = 3.5 kpc
    mass     1 unit = 4e10 Msun
    velocity 1 unit = sqrt(G M / L)  ~ 222 km/s
    time     1 unit = L / V          ~ 15.4 Myr

All command-line options are in physical units (kpc, Myr, Msun, K, atoms/cm^3) and are
converted with the constants below.
"""

import numpy as np

G_KPC = 4.30091e-6                    # gravitational constant in kpc (km/s)^2 / Msun
KPC = 3.5                             # length unit in kpc
MSUN = 4e10                           # mass unit in Msun
KMS = np.sqrt(G_KPC * MSUN / KPC)     # velocity unit in km/s (~222)
MYR = KPC / KMS * 977.79              # time unit in Myr (1 kpc/(km/s) = 977.79 Myr), ~15.4

# cgs constants
M_H = 1.6726e-24                      # hydrogen mass (g)
K_B = 1.381e-16                       # Boltzmann constant (erg/K)
MSUN_G = 1.989e33                     # solar mass (g)
KPC_CM = 3.086e21                     # kpc (cm)
YR_S = 3.156e7                        # year (s)

RHO_CGS = MSUN * MSUN_G / (KPC * KPC_CM) ** 3       # density unit (g/cm^3)
U_CGS = (KMS * 1e5) ** 2                            # specific energy unit (erg/g)
SEC_PER_MYR = 1e6 * YR_S                            # seconds in a Myr

X_H = 0.76                            # hydrogen mass fraction (primordial gas)
MU_IONISED = 0.6                      # mean molecular weight of an ionised primordial gas
