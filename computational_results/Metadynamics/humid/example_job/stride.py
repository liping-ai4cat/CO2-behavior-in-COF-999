#!/usr/bin/env python
"""Stride an ASE trajectory file.

Example
-------
python stride_traj.py meta.traj --stride 50      -> meta_50_stride.traj
python stride_traj.py meta.traj --stride 50 -o sub.traj
"""

import argparse
from pathlib import Path

from ase.io import Trajectory


def main():
    p = argparse.ArgumentParser(description="Write every Nth frame of a trajectory.")
    p.add_argument("input", type=Path, help="input trajectory (e.g. meta.traj)")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    p.add_argument("--start", type=int, default=0, help="first frame to keep")
    p.add_argument("--stop", type=int, default=None, help="last frame (exclusive)")
    p.add_argument("-o", "--output", type=Path, default=None, help="output filename")
    args = p.parse_args()

    if args.stride < 1:
        p.error("--stride must be >= 1")

    out = args.output or args.input.with_name(
        f"{args.input.stem}_{args.stride}_stride{args.input.suffix or '.traj'}"
    )

    src = Trajectory(str(args.input), "r")
    stop = len(src) if args.stop is None else args.stop
    kept = 0

    with Trajectory(str(out), "w") as dst:
        for i in range(args.start, min(stop, len(src)), args.stride):
            dst.write(src[i])
            kept += 1

    print(f"{args.input} ({len(src)} frames) -> {out} ({kept} frames), stride={args.stride}")


if __name__ == "__main__":
    main()
