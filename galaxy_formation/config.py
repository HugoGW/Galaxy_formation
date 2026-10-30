"""Simulation parameters (physical units). main.py exposes each field as a command-line option."""

from dataclasses import dataclass


@dataclass
class Config:
    # particles
    n_gas: int = 5000            # gas particles (baryons)
    n_dm: int = 5000             # dark-matter particles
    seed: int = 3                # random seed

    # initial cloud
    mass: float = 2.8e11         # total mass (Msun)
    f_baryon: float = 0.16       # baryon (gas) fraction, cosmic value ~ 0.16 (Planck 2018)
    profile: str = "extended"    # "extended" (density ~ 1/r) or "uniform"
    r0: float = 100.0            # cloud radius (kpc)
    spin: float = 0.05           # spin parameter lambda (Bullock et al. 2001)
    t_gas: float = 1e4           # initial gas temperature (K)

    # star formation and feedback
    n_sf: float = 1.0            # star-formation density threshold (H atoms / cm^3)
    sf_eff: float = 0.005        # efficiency per free-fall time
    fb_eff: float = 0.4          # supernova energy coupled to the gas, in units of 1e49 erg / Msun formed
    fb_ngb: int = 16             # gas neighbours heated by each new star particle
    fb_myr: float = 30.0         # cooling switched off for this long after heating (Myr)

    # numerics
    eps: float = 0.4             # gravitational softening (kpc)
    theta: float = 0.7           # Barnes-Hut opening angle

    # output
    gyr: float = 5.0             # simulated time (Gyr)
    myr_per_frame: float = 5.0   # simulated time between two video frames (Myr)
    view: float = 40.0           # half-width of each view (kpc)
    size: int = 600              # size of each view (pixels, rounded down to a multiple of 4)
    fps: int = 30
    save: str = "galaxy_formation.mp4"


HELP = {
    "n_gas": "number of gas particles",
    "n_dm": "number of dark-matter particles",
    "seed": "random seed",
    "mass": "total mass of the cloud (Msun)",
    "f_baryon": "gas (baryon) mass fraction",
    "profile": "initial cloud: 'extended' (density ~ 1/r) or 'uniform'",
    "r0": "radius of the initial cloud (kpc)",
    "spin": "spin parameter lambda",
    "t_gas": "initial gas temperature (K)",
    "n_sf": "star-formation density threshold (atoms/cm^3)",
    "sf_eff": "star-formation efficiency per free-fall time",
    "fb_eff": "supernova efficiency (fraction of 1e49 erg per Msun formed)",
    "fb_ngb": "gas neighbours heated by each supernova",
    "fb_myr": "time without cooling for heated gas (Myr)",
    "eps": "gravitational softening (kpc)",
    "theta": "Barnes-Hut opening angle",
    "gyr": "simulated time (Gyr)",
    "myr_per_frame": "simulated Myr per video frame",
    "view": "half-width of each view (kpc)",
    "size": "size of each view in pixels",
    "fps": "video frames per second",
    "save": "output video file (.mp4)",
}
