"""Python wrapper around the C++ engine (cpp/nbody.cpp).

The shared library is compiled automatically into build/ on first use, or whenever
nbody.cpp is newer than the library.
"""

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "cpp", "nbody.cpp")
LIB = os.path.join(ROOT, "build", "libnbody.so")


def _load():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        os.makedirs(os.path.dirname(LIB), exist_ok=True)
        cmd = ["g++", "-O3", "-march=native", "-ffast-math", "-fopenmp", "-std=c++17",
               "-shared", "-fPIC", SRC, "-o", LIB]
        print("Compiling:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(LIB)
    dbl = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    u8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    c_int, c_dbl, c_flt = ctypes.c_int, ctypes.c_double, ctypes.c_float
    lib.nbody_accel_active.argtypes = [c_int, dbl, dbl, dbl, c_dbl, c_dbl, u8, c_int]
    lib.nbody_accel_direct.argtypes = [c_int, dbl, dbl, dbl, c_dbl]
    lib.nbody_render.argtypes = [c_int, dbl, u8, c_int, dbl, dbl, c_dbl, c_int, c_int, f32]
    lib.nbody_tonemap.argtypes = [c_int, c_int, c_int, f32, f32, c_int, c_int, f32,
                                  c_flt, c_flt, c_flt, u8]
    lib.nbody_num_threads.restype = c_int
    return lib


class Engine:
    """Thin, typed interface to the compiled library."""

    def __init__(self):
        self.lib = _load()

    def gravity(self, pos, mass, acc, theta, eps, active=None, rebuild=True):
        """Barnes-Hut accelerations (G = 1) written into acc for the active particles.
        rebuild=False refits the previous tree instead of building a new one."""
        n = len(mass)
        act = np.ones(n, np.uint8) if active is None else np.ascontiguousarray(active, dtype=np.uint8)
        self.lib.nbody_accel_active(n, pos, mass, acc, theta, eps, act, int(rebuild))

    def gravity_direct(self, pos, mass, eps):
        """Exact O(N^2) accelerations (for tests)."""
        acc = np.zeros_like(pos)
        self.lib.nbody_accel_direct(len(mass), pos, mass, acc, eps)
        return acc

    def project(self, pos, group, n_groups, rot, scale, size):
        """Particle counts per pixel and per colour group, shape (n_groups, size, size)."""
        img = np.empty((n_groups, size, size), np.float32)
        self.lib.nbody_render(len(group), np.ascontiguousarray(pos), group, n_groups,
                              np.ascontiguousarray(rot), np.zeros(3), scale, size, size, img)
        return img

    def tonemap(self, img, glow, colors, inv_ref, K, exposure):
        """8-bit RGB image from the per-group count maps and their (4x smaller) glow maps."""
        ng, H, W = img.shape
        out = np.empty((H, W, 3), np.uint8)
        self.lib.nbody_tonemap(W, H, ng, img, np.ascontiguousarray(glow), glow.shape[2], glow.shape[1],
                               np.ascontiguousarray(colors, dtype=np.float32).ravel(),
                               inv_ref, K, exposure, out)
        return out
