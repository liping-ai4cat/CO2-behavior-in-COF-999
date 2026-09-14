#!/usr/bin/env python3
"""
Create element-averaged SOAP global descriptors per structure, using multiprocessing and batch streaming.

For each structure:
  - Compute SOAP per-atom.
  - Average by element in fixed order (C, H, N, O).
  - Concatenate the four averages → one 1-D descriptor per structure.

CSV columns:
  - natoms: integer (total number of atoms in the structure)
  - features: JSON list of floats (flattened concatenated descriptor)

Requirements:
  pip install dscribe==2.1.1 ase numpy pandas

Usage example:
  python make_soap_features.py \
      --traj "md_*.traj" \
      --rcut 6.0 --nmax 9 --lmax 3 \
      --batch-size 50000 \
      --num-workers 32 \
      --output active_learning.csv
"""

from __future__ import annotations
import argparse
import json
import ast
import gc
from pathlib import Path
from glob import glob
import multiprocessing as mp

import numpy as np
import pandas as pd
from ase.io import Trajectory, read
from ase import Atoms
from dscribe.descriptors import SOAP

# Fixed element order: C, H, N, O
ELEMENT_ORDER = [6, 1, 7, 8]

_PRINTED_DEBUG = False
_global_soap: SOAP  # placeholder for worker global

def load_image_paths(pattern: str) -> list[Path]:
    paths = sorted(glob(pattern))
    if not paths:
        p = Path(pattern)
        if p.exists():
            paths = [p]
    if not paths:
        raise FileNotFoundError(f"No files found for '{pattern}'")
    return [Path(p) for p in paths]

def iter_structures_from_paths(paths: list[Path]):
    for p in paths:
        if p.suffix.lower() == ".traj":
            with Trajectory(str(p)) as t:
                for i in range(len(t)):
                    yield t[i]
        else:
            try:
                imgs = read(str(p), index=":")
                if isinstance(imgs, list):
                    for a in imgs:
                        yield a
                else:
                    yield imgs
            except Exception:
                yield read(str(p))

def build_soap(species: list[int], rcut: float, nmax: int, lmax: int, periodic: bool = True) -> SOAP:
    return SOAP(
        species = sorted(set(species)),
        r_cut = rcut,
        n_max = nmax,
        l_max = lmax,
        periodic = periodic,
        sparse = False
    )

def per_structure_feature_species_concat(atoms: Atoms) -> np.ndarray:
    global _PRINTED_DEBUG, _global_soap
    soap = _global_soap
    D = soap.create(atoms)
    Z = atoms.numbers
    n_feat = D.shape[1]
    chunks = []
    for Zsym in ELEMENT_ORDER:
        mask = (Z == Zsym)
        if np.any(mask):
            chunks.append(D[mask].mean(axis=0))
        else:
            chunks.append(np.zeros(n_feat, dtype=D.dtype))
    feat_vec = np.concatenate(chunks, axis=0)
    expected_len = 4 * n_feat
    actual_len = int(feat_vec.size)
    if actual_len != expected_len:
        raise ValueError(f"SOAP feature length mismatch: expected {expected_len}, got {actual_len}")
    if not _PRINTED_DEBUG:
        print(f"[DEBUG] Single SOAP per-atom length (n_feat): {n_feat}")
        print(f"[DEBUG] Final concatenated descriptor length: {actual_len} (should be 4 * n_feat)")
        _PRINTED_DEBUG = True
    return feat_vec

def worker_init(soap_descriptor):
    """Initializer for each worker process."""
    global _global_soap
    _global_soap = soap_descriptor

def worker_compute(args):
    """Worker function: (idx, Atoms) → (idx, natoms, feat_vec)"""
    idx, atoms = args
    feat = per_structure_feature_species_concat(atoms)
    natoms = len(atoms)
    return idx, natoms, feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, required=True, help="Trajectory glob or file (or a directory).")
    ap.add_argument("--rcut", type=float, default=6.0, help="SOAP cutoff (Å).")
    ap.add_argument("--nmax", type=int, default=9, help="SOAP nmax.")
    ap.add_argument("--lmax", type=int, default=3, help="SOAP lmax.")
    ap.add_argument("--batch-size", type=int, default=50000, help="Number of structures per batch.")
    ap.add_argument("--limit", type=int, default=None, help="Optional: only use first N structures.")
    ap.add_argument("--num-workers", type=int, default=32, help="Number of worker processes.")
    ap.add_argument("--output", type=str, default="active_learning.csv", help="Output CSV filename.")
    args = ap.parse_args()

    paths = load_image_paths(args.traj)
    print(f"Found {len(paths)} files matching pattern.")

    # Build SOAP: sample species from first few structures
    species = set()
    sample_cnt = 0
    for p in paths:
        if sample_cnt >= 100:
            break
        if p.suffix.lower() == ".traj":
            with Trajectory(str(p)) as t:
                species.update(t[0].numbers.tolist())
        else:
            imgs = read(str(p), index=":")
            if isinstance(imgs, list):
                species.update(imgs[0].numbers.tolist())
            else:
                species.update(imgs.numbers.tolist())
        sample_cnt += 1

    soap_descriptor = build_soap(list(species), args.rcut, args.nmax, args.lmax, periodic=True)
    print(f"Built SOAP descriptor for species: {sorted(species)}")

    struct_iter = iter_structures_from_paths(paths)
    total = 0
    batch_size = args.batch_size
    out_csv = args.output

    # Write header later
    header_written = False

    while True:
        batch_list = []
        for i in range(batch_size):
            try:
                atoms = next(struct_iter)
                batch_list.append((total + i, atoms))
            except StopIteration:
                break
        if not batch_list:
            break

        print(f"Processing batch starting at index {total} (size {len(batch_list)})")

        # Use multiprocessing within batch
        with mp.Pool(
            processes = args.num_workers,
            initializer = worker_init,
            initargs = (soap_descriptor,)
        ) as pool:
            for idx, natoms, feat in pool.imap_unordered(worker_compute, batch_list, chunksize=100):
                # write row immediately
                row = {
                    "natoms": natoms,
                    "features": json.dumps(feat.tolist()),
                }
                # Append to CSV
                df_row = pd.DataFrame([row])
                df_row.to_csv(out_csv, mode='a', index=False, header=(not header_written))
                header_written = True
            pool.close()
            pool.join()

        total += len(batch_list)
        print(f"Completed total {total} structures so far.")
        # free memory
        batch_list.clear()
        gc.collect()

        if args.limit and total >= args.limit:
            break

    print(f"Finished processing {total} structures. Results in {out_csv}")

if __name__ == "__main__":
    main()
