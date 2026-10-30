"""Simulates the formation of a disk galaxy and writes the video.

Examples
--------
    python main.py                                    # 10 000 particles, ~10 min
    python main.py --n-gas 100000 --n-dm 100000 --save galaxy_N200000.mp4   # the README video
    python main.py --gyr 1 --size 400                 # quick look at the first billion years
"""

import argparse
import dataclasses
import time

from galaxy_formation import Config, GalaxyFormation
from galaxy_formation.config import HELP
from galaxy_formation.units import MYR
from galaxy_formation.video import FrameMaker, VideoWriter


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        kw = {"type": f.type if callable(f.type) else type(f.default), "default": f.default,
              "help": f"{HELP[f.name]} (default: %(default)s)"}
        if f.name == "profile":
            kw["choices"] = ["extended", "uniform"]
        p.add_argument("--" + f.name.replace("_", "-"), **kw)
    cfg = Config(**vars(p.parse_args(argv)))
    cfg.size -= cfg.size % 4
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    sim = GalaxyFormation(cfg)
    make_frame = FrameMaker(sim, cfg)
    video = VideoWriter(cfg.save, *make_frame.shape, cfg.fps)

    n_frames = int(round(cfg.gyr * 1000 / cfg.myr_per_frame))
    dt_frame = cfg.myr_per_frame / MYR
    t0 = time.time()
    for k in range(n_frames + 1):
        if k > 0:
            sim.advance_to(k * dt_frame)
        video.write(make_frame(sim))
        if k % 10 == 0:
            el = time.time() - t0
            fstar = sim.history[-1][2] if sim.history else 0.0
            print(f"\rframe {k}/{n_frames}  t = {sim.t * MYR:6.0f} Myr  stars {fstar:5.1%}  "
                  f"({el:5.0f} s, ~{el / (k + 1) * (n_frames - k):5.0f} s left)", end="", flush=True)
    video.close()
    print(f"\nVideo written: {cfg.save}")


if __name__ == "__main__":
    main()
