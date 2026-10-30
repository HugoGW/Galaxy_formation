"""Formation of a disk galaxy from a cold, rotating cloud of gas and dark matter.

Modules
-------
units               code units (G = 1) and physical constants
config              simulation parameters
engine              C++ Barnes-Hut gravity and particle rendering (ctypes)
initial_conditions  the initial cloud
sph                 gas hydrodynamics (SPH)
cooling             radiative cooling of primordial gas
star_formation      star formation and supernova feedback
simulation          particle state and hierarchical leapfrog integration
rendering           colours, disk frame, camera and tone mapping
video               frame composition and ffmpeg output
"""

from .config import Config
from .simulation import GalaxyFormation
