"""Rendering: particle colours, disk-aligned frame of reference, camera and tone mapping.

Only baryons are drawn (dark matter is invisible). Each baryon falls into one of four colour
groups; each group is projected into its own count map, then all maps are tone-mapped
together:

    l_g = log(1 + K n_g / ref) / log(1 + K) + glow_g
    rgb = 1 - exp(-exposure * sum_g l_g color_g)

The logarithm brings out faint structures (spiral arms, tidal streams) next to the bright
centre; the glow is a blurred, 4x down-sampled copy of the image that mimics the
scattered light of a real photograph; the last line saturates smoothly to white.
`ref` is the 99.9th percentile of the image, smoothed in time to avoid flicker.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from scipy.ndimage import gaussian_filter

from .initial_conditions import DM, GAS, STAR
from .units import KPC, MYR

COLD_GAS, YOUNG_STARS, OLD_STARS, HOT_GAS = 0, 1, 2, 3
COLORS = np.array([[0.55, 0.50, 0.95],     # cold gas (< 1e5 K): pale blue-violet
                   [0.70, 0.85, 1.00],     # young stars (< 300 Myr): blue-white (O, B, A stars)
                   [1.00, 0.78, 0.45],     # old stars: yellow-orange (G, K, M stars)
                   [0.55, 0.22, 0.12]])    # hot halo gas (> 1e5 K): dark red
LABELS = ["cold gas", "young stars", "old stars", "hot gas"]
T_HOT = 1e5            # K
YOUNG_MYR = 300.0      # Myr


def font(size):
    return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)


def colour_groups(sim):
    """Colour group of every baryon (baryons are the first particles)."""
    nb = int(np.sum(sim.kind != DM))
    k = sim.kind[:nb]
    g = np.zeros(nb, np.uint8)
    g[(k == GAS) & (sim.temperature(np.arange(nb)) > T_HOT)] = HOT_GAS
    young = (k == STAR) & ((sim.t - sim.birth[:nb]) * MYR < YOUNG_MYR)
    g[young] = YOUNG_STARS
    g[(k == STAR) & ~young] = OLD_STARS
    return g


def disk_frame(sim):
    """Centre and spin axis of the forming disk: mass-weighted centre of the inner half of
    the baryons (re-centred once) and direction of their angular momentum."""
    b = sim.kind != DM
    pos, vel, m = sim.pos[b], sim.vel[b], sim.mass[b]
    c = np.average(pos, axis=0, weights=m)
    r = np.linalg.norm(pos - c, axis=1)
    near = r < np.percentile(r, 50)
    c = np.average(pos[near], axis=0, weights=m[near])
    J = np.sum(m[near, None] * np.cross(pos[near] - c, vel[near]), axis=0)
    return c, J / (np.linalg.norm(J) + 1e-30)


def rotation_to_z(J):
    """Rotation matrix taking the unit vector J onto the z axis (Rodrigues' formula)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(J, z)
    s, c = np.linalg.norm(v), J @ z
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / s ** 2


class DiskTracker:
    """Smoothed disk frame, so that the camera does not jitter from frame to frame."""

    def __init__(self):
        self.c = self.J = None

    def update(self, sim):
        c, J = disk_frame(sim)
        if self.c is None:
            self.c, self.J = c, J
        else:
            self.c = self.c + 0.2 * (c - self.c)
            self.J = self.J + 0.1 * (J - self.J)
            self.J /= np.linalg.norm(self.J)
        return self.c, rotation_to_z(self.J)


class Camera:
    """Orthographic camera looking along the direction given by (azim, elev) in degrees."""

    K = 150.0          # strength of the logarithmic compression
    GLOW = 0.5         # glow intensity
    EXPOSURE = 0.8

    def __init__(self, engine, size, half_width_kpc, azim, elev, colors=COLORS):
        self.engine = engine
        self.size = size
        self.half_width = half_width_kpc / KPC          # code units
        self.azim, self.elev = azim, elev
        self.colors = np.ascontiguousarray(colors, dtype=np.float32)
        self.ref = None
        self._font, self._small = font(max(12, size // 40)), font(max(10, size // 55))

    def matrix(self):
        """Rows: screen x axis, screen y axis, viewing direction."""
        a, e = np.radians(self.azim), np.radians(self.elev)
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        right = np.array([-np.sin(a), np.cos(a), 0.0])
        up = np.cross(d, right)
        return np.stack([right, up, d])

    def render(self, pos, group):
        """RGB uint8 image (size, size, 3) of the particles pos (one colour group each)."""
        S, ng = self.size, len(self.colors)
        img = self.engine.project(pos, group, ng, self.matrix(), S / (2 * self.half_width), S)
        small = img.reshape(ng, S // 4, 4, S // 4, 4).mean(axis=(2, 4))
        level = max(np.percentile(small.sum(axis=0), 99.9), 1e-3)
        self.ref = level if self.ref is None else 0.92 * self.ref + 0.08 * level
        glow = gaussian_filter(small / self.ref, sigma=(0, 3, 3))
        glow = (self.GLOW * np.log1p(self.K * glow) / np.log1p(self.K)).astype(np.float32)
        return self.engine.tonemap(img, glow, self.colors, 1 / self.ref, self.K, self.EXPOSURE)

    def annotate(self, rgb, t_myr, caption):
        """Adds the time, a caption and a scale bar (kpc)."""
        S = self.size
        im = Image.fromarray(rgb)
        dr = ImageDraw.Draw(im)
        m = S // 40
        white = (230, 230, 230)
        dr.text((m, m), f"t = {t_myr:.0f} Myr", fill=white, font=self._font)
        # scale bar: a round length close to 1/5 of the field
        width_kpc = 2 * self.half_width * KPC
        L = min([1, 2, 5, 10, 20, 50, 100, 200, 500], key=lambda x: abs(x - width_kpc / 5))
        px = L / width_kpc * S
        y = S - 2 * m
        dr.line([(m, y), (m + px, y)], fill=white, width=max(2, S // 400))
        dr.text((m + px / 2, y - m // 3), f"{L} kpc", fill=white, font=self._small, anchor="mb")
        dr.text((S // 2, S - 22), caption, fill=(150, 150, 150), font=font(14), anchor="mm")
        return im
