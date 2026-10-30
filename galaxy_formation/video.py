"""Video output: composition of a frame (face-on and edge-on views + header) and the
ffmpeg pipe that encodes the frames into an H.264 .mp4 file."""

import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .rendering import COLORS, HOT_GAS, LABELS, Camera, DiskTracker, colour_groups, font
from .units import MYR

HEADER = 60     # pixels above the views


class VideoWriter:
    """Streams raw RGB frames to ffmpeg."""

    def __init__(self, path, width, height, fps):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.asarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


class FrameMaker:
    """Face-on view (looking down the spin axis) and edge-on view of the forming disk, side by side."""

    def __init__(self, sim, cfg):
        S = cfg.size
        self.S = S
        self.myr_per_frame = cfg.myr_per_frame
        self.face = Camera(sim.engine, S, cfg.view, azim=-90, elev=90)
        self.edge = Camera(sim.engine, S, cfg.view, azim=-90, elev=0)
        self.tracker = DiskTracker()
        self.font, self.small = font(18), font(14)

    @property
    def shape(self):
        return 2 * self.S, self.S + HEADER

    def __call__(self, sim):
        S = self.S
        groups = colour_groups(sim)
        # the scene is rotated into the frame of the forming disk (smoothed in time)
        c, R = self.tracker.update(sim)
        pos = (sim.pos[:len(groups)] - c) @ R.T
        t = sim.t * MYR
        frame = Image.new("RGB", self.shape)
        frame.paste(self.face.annotate(self.face.render(pos, groups), t, "face-on"), (0, HEADER))
        frame.paste(self.edge.annotate(self.edge.render(pos, groups), t, "edge-on"), (S, HEADER))

        d = ImageDraw.Draw(frame)
        sfr, fstar = sim.history[-1][1:] if sim.history else (0.0, 0.0)
        d.text((10, 20), f"star formation rate: {sfr:6.1f} M☉/yr   —   gas already turned into stars: {fstar:5.1%}",
               fill=(200, 200, 200), font=self.small)
        d.text((2 * S - 10, 8), f"1 frame = {self.myr_per_frame:g} Myr", fill=(255, 224, 138),
               font=self.font, anchor="ra")
        x = 2 * S - 10
        for i in reversed(range(len(LABELS))):      # colour legend, right-aligned
            col = np.clip(COLORS[i] * (1.6 if i == HOT_GAS else 1.0), 0, 1)
            d.text((x, 34), LABELS[i], fill=tuple(int(255 * v) for v in col), font=self.small, anchor="ra")
            x -= d.textlength(LABELS[i], font=self.small) + 18
        return frame
