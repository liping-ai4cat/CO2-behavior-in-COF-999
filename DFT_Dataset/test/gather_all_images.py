#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
from typing import List, Tuple

import numpy as np
from ase.io import read, write

# ========= User configurable section =========
Project_DIRS = '/scratch/lipingliu/double_Z/new_benchmarking_true_test'

# Key: Directory to search (recursively)
# Value: Output filename prefix (saved as value.traj)
ROOT_DIRS = {
   # f"{Project_DIRS}/the_lowest_energy_states_from_annealing_MD": "the_lowest_energy_states_from_annealing_MD",
   # f"{Project_DIRS}/Annealing_MD" : "Annealing_MD",
   # f"{Project_DIRS}/Physisorption_basin": "Physisorption_basin",
   # f"{Project_DIRS}/chemisorption": "chemisorption",
   # f"{Project_DIRS}/CO2_adsorption_dry_COF999": "CO2_adsorption_dry_COF999",
   # f"{Project_DIRS}/the_lowest_energy_states_from_annealing_MD+H2O": "the_lowest_energy_states_from_annealing_MD+H2O",
   # f"{Project_DIRS}/CO2_adsorption_humid_COF999": "CO2_adsorption_humid_COF999",
    #f"{Project_DIRS}/the_lowest_energy_states_from_annealing_MD_114": "114",
    f"{Project_DIRS}/the_lowest_energy_states_from_annealing_MD_113": "113",
    f"{Project_DIRS}/mutiple_H2O_MD":"mutiple_H2O_MD",
    f"{Project_DIRS}/mutiple_CO2_dry_META": "mutiple_CO2_dry_META",
    f"{Project_DIRS}/mutiple_CO2_humid_META": "mutiple_CO2_humid_META",

}

OUTCAR_NAME = "OUTCAR"

SKIP_FIRST_N = 0              # discard the first N frames -> index="6:"
MIN_DELTA_E = 0.0001           # eV, minimum energy spacing between any two frames
MAX_SELECT_PER_FILE = 100      # maximum selected frames per OUTCAR
RANDOM_SEED = 42              # random seed for reproducibility
# =============================================


def find_outcars(root_dir: str, filename: str = "OUTCAR") -> List[str]:
    """Recursively find all OUTCAR files in a specific root directory."""
    outcars = []
    for dirpath, _, files in os.walk(root_dir):
        if filename in files:
            outcars.append(os.path.join(dirpath, filename))
    return outcars


def extract_dft_energy(atoms) -> float:
    """
    Robustly extract the DFT energy (eV) from ASE Atoms/calc.
    Priority: atoms.get_potential_energy() -> atoms.info common keys -> calc.results
    """
    # 1) Standard: calc.results
    try:
        e = atoms.get_potential_energy()
        if e is not None and np.isfinite(e):
            return float(e)
    except Exception:
        pass

    # 2) Common info keys
    for k in ("energy", "free_energy", "E", "dft_energy"):
        if k in atoms.info:
            try:
                val = float(atoms.info[k])
                if np.isfinite(val):
                    return val
            except Exception:
                continue

    # 3) Fallback from calc.results
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        res = getattr(calc, "results", {})
        if "energy" in res:
            val = res["energy"]
            try:
                val = float(val)
                if np.isfinite(val):
                    return val
            except Exception:
                pass

    raise ValueError("Could not parse DFT energy from this frame.")


def select_diverse_by_energy(
    images: List, energies: np.ndarray, min_delta: float, max_n: int
) -> List[int]:
    """
    Farthest-point sampling (FPS) along the energy axis.
    """
    n = len(images)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # dynamic scaling of max_n
    if n < 600:
        scaled_max_n = max_n
    elif n <= 1200:
        scaled_max_n = int(1.5 * max_n)
    else:
        scaled_max_n = int(2 * max_n)

    order = np.argsort(energies)
    selected = {order[0], order[-1]} if n >= 2 else {order[0]}

    def nearest_dist(idx: int, selected_set: set) -> float:
        e = energies[idx]
        return min(abs(e - energies[j]) for j in selected_set)

    while len(selected) < min(scaled_max_n, n):
        best_idx, best_dist = None, -1.0
        for i in range(n):
            if i in selected:
                continue
            d = nearest_dist(i, selected)
            if d > best_dist:
                best_dist, best_idx = d, i

        if best_dist < min_delta:
            break
        selected.add(best_idx)

    sel_list = sorted(list(selected), key=lambda i: energies[i])
    final_sel = []
    for i in sel_list:
        e = energies[i]
        if all(abs(e - energies[j]) >= min_delta for j in final_sel):
            final_sel.append(i)
        if len(final_sel) >= scaled_max_n:
            break

    return final_sel

def read_outcar_frames(path: str, skip_first_n: int) -> List:
    """
    Read frames, skipping the first skip_first_n frames.
    Priority: OUTCAR -> Fallback: vasprun.xml (in same dir).
    """
    # Helper to standardize single Atoms object -> List[Atoms]
    def ensure_list(imgs):
        return [imgs] if not isinstance(imgs, list) else imgs

    # 1. Attempt to read the target file (usually OUTCAR)
    try:
        images = read(path, index=f"{skip_first_n}:")
        return ensure_list(images)

    # 2. If parsing fails (e.g. 'H/' error), try vasprun.xml
    except Exception as e_outcar:
        directory = os.path.dirname(path)
        xml_path = os.path.join(directory, "vasprun.xml")

        if os.path.exists(xml_path):
            try:
                print(f"[INFO] OUTCAR failed ({e_outcar}). Switching to: {xml_path}")
                images = read(xml_path, index=f"{skip_first_n}:")
                return ensure_list(images)
            except Exception as e_xml:
                print(f"[WARN] Failed to read both OUTCAR and vasprun.xml in {directory}")
                return []
        else:
            print(f"[WARN] Failed to read {path}: {e_outcar} (vasprun.xml not found)")
            return []

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=======================================================")
    print(f"[START] Processing {len(ROOT_DIRS)} root directories.")
    print("=======================================================\n")

    # Iterate over each dictionary item
    for root_path, output_name in ROOT_DIRS.items():
        print(f"--- Processing Root: {root_path} ---")
        
        # 1. Find OUTCARs in this specific root
        outcars = find_outcars(root_path, OUTCAR_NAME)
        if not outcars:
            print(f"[INFO] No OUTCAR found in {root_path}, skipping.")
            continue
        
        print(f"[INFO] Found {len(outcars)} OUTCAR files.")

        root_images_list = [] # Store images for this root only

        # 2. Process each OUTCAR
        for oc in outcars:
            imgs = read_outcar_frames(oc, SKIP_FIRST_N)
            if not imgs:
                continue

            # Extract energies
            energies = []
            valid_idx = []
            for i, at in enumerate(imgs):
                try:
                    e = extract_dft_energy(at)
                    energies.append(e)
                    valid_idx.append(i)
                except Exception:
                    pass

            if not energies:
                # print(f"[WARN] {oc} has no valid frames with energies, skipped.")
                continue

            energies = np.array(energies, dtype=float)
            imgs_valid = [imgs[i] for i in valid_idx]

            # Select diverse subset
            sel_idx_local = select_diverse_by_energy(
                imgs_valid, energies, MIN_DELTA_E, MAX_SELECT_PER_FILE
            )
            selected = [imgs_valid[i] for i in sel_idx_local]

            root_images_list.extend(selected)
            # Optional: print per-file stats to avoid clutter
            # print(f"[INFO] {os.path.basename(os.path.dirname(oc))}: selected {len(selected)} frames.")

        # 3. Save combined file for this root
        if not root_images_list:
            print(f"[WARN] No valid images collected for {output_name}.traj")
        else:
            output_filename = f"{output_name}.traj"
            write(output_filename, root_images_list)
            print(f"[DONE] Saved {len(root_images_list)} frames to -> {output_filename}\n")

    print("=======================================================")
    print("[ALL DONE] All directories processed.")

if __name__ == "__main__":
    main()
