#!/usr/bin/env python3
import numpy as np
import csv
import multiprocessing
from pathlib import Path
from ase.io import read
from ase.data import covalent_radii

# --- SETTINGS ---
R_SCALE = 1.2
REF_PATH = "/projects/caiw/lipingliu/COF-999/metadynamics_using_uma/fine-tuned-uma-s-v10-round2-SOAP-v3/CO2-single-COF-999-humid/plus_CO2.traj"
CSV_PATH = "/projects/caiw/lipingliu_tianyou/CO2-single-COF-999-humid/analysis_output/v7_all/job_results_updated.csv"


def get_partition_ids(csv_path):
    """
    Partitions Job IDs based on strict kinetic and thermodynamic performance thresholds:
    part_1: Spontaneous and fast channels (deltaG_new < 0 AND Ga_new < 70 kJ/mol)
    part_2: Residual non-favorable or sluggish amine channels
    """
    part_1 = []
    part_2 = []
    print(f"Reading JobIDs from {csv_path} using custom strict screening...")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Clean header names (strip whitespace)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        
        for row in reader:
            try:
                dg = float(row['deltaG_new'].strip())
                ga = float(row['Ga_new'].strip())
                jid = row['job_id'].strip()

                # Optimized Screening Logic: deltaG < 0 AND Ga < 70
                if dg <= 0 and ga <= 70:
                    part_1.append(jid)
                elif dg <= 0 and ga > 70:
                    part_2.append(jid)
                else:
                    continue
            except (ValueError, KeyError, AttributeError):
                continue
                
    print(f"--> Screening Complete. Part 1 (Active): {len(part_1)} jobs | Part 2: {len(part_2)} jobs")
    return part_1, part_2

# --- TOPOLOGY & GEOMETRY HELPERS ---

def mic_vec(cell, pbc, dv):
    if not np.any(pbc): return dv
    icell = np.linalg.inv(cell.T)
    s = icell @ dv
    s -= np.round(s * pbc)
    return cell.T @ s

def bonded_neighbors(atoms, i):
    positions = atoms.positions
    cell = atoms.cell.array
    pbc = atoms.pbc
    ri = covalent_radii[atoms[i].number]
    neighbors = []
    for j in range(len(atoms)):
        if i == j: continue
        rj = covalent_radii[atoms[j].number]
        dist = np.linalg.norm(mic_vec(cell, pbc, positions[j] - positions[i]))
        if dist <= R_SCALE * (ri + rj):
            neighbors.append(j)
    return neighbors

def classify_amine(atoms, idx_N):
    neighs = bonded_neighbors(atoms, idx_N)
    H_neighbors = [j for j in neighs if atoms.symbols[j] == "H"]
    nH = len(H_neighbors)
    if nH == 2:
        am_type = "N_primary"
    elif nH == 1:
        am_type = "N_secondary"
    elif nH == 0:
        heavy_neighbors = [j for j in neighs if atoms.symbols[j] != "H"]
        am_type = "N_tertiary" if len(heavy_neighbors) >= 3 else "N_other"
    else:
        am_type = "N_other"
    return am_type, H_neighbors, nH

def get_h_counts_and_types(atoms, find_N_type=False):
    positions = atoms.get_positions()
    cell = atoms.cell.array
    pbc = atoms.pbc
    symbols = atoms.get_chemical_symbols()
    no_indices = [i for i, s in enumerate(symbols) if s in ['N', 'O']]
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    counts = {}
    types = {}
    for i in no_indices:
        h_count = 0
        ri = covalent_radii[atoms[i].number]
        for j in h_indices:
            rj = covalent_radii[atoms[j].number]
            dist = np.linalg.norm(mic_vec(cell, pbc, positions[j] - positions[i]))
            if dist <= R_SCALE * (ri + rj):
                h_count += 1
        counts[i] = h_count
        if find_N_type:
            if symbols[i] == 'N':
                am_type, _, _ = classify_amine(atoms, i)
                types[i] = am_type
            else:
                c_indices = [idx for idx, s in enumerate(symbols) if s == 'C']
                is_co2 = False
                for c_idx in c_indices:
                    rc = covalent_radii[6]
                    dist_co = np.linalg.norm(mic_vec(cell, pbc, positions[c_idx] - positions[i]))
                    if dist_co <= R_SCALE * (ri + rc):
                        is_co2 = True
                        break
                types[i] = 'O_CO2' if is_co2 else 'O_H2O'
        else:
            types[i] = symbols[i]
    return counts, types

# --- WORKER FUNCTION ---

def process_trajectory(args):
    part_name, jobid, ref_h_counts, ref_types = args
    traj_path = Path(f"../all/{jobid}_sB.traj")
    if not traj_path.exists(): return None

    try:
        traj = read(traj_path, index=':')
        unique_labels = set(ref_types.values())
        stats = {"JobID": jobid, "Partition": part_name, "total_frames": len(traj)}
        for lbl in unique_labels:
            stats[f"{lbl}_plus_1"] = 0
            stats[f"{lbl}_minus_1"] = 0

        dist_records = []

        for f_idx, atoms in enumerate(traj):
            current_h_counts, _ = get_h_counts_and_types(atoms, find_N_type=False)

            plus_N_indices = []
            max_minus_N_indices = []
            minus_N_indices = []

            for idx, ref_h in ref_h_counts.items():
                label = ref_types[idx]
                curr_h = current_h_counts.get(idx, 0)
                diff = curr_h - ref_h

                if diff >= 1:
                    stats[f"{label}_plus_1"] += 1
                    if "N" in label: plus_N_indices.append(idx)
                elif diff <= -1:
                    stats[f"{label}_minus_1"] += 1
                    if "N" in label: minus_N_indices.append(idx)

            if plus_N_indices and minus_N_indices:
                cell = atoms.cell.array
                pbc = atoms.pbc
                pos = atoms.positions
                for p_idx in plus_N_indices:
                    for m_idx in minus_N_indices:
                        d_vec = mic_vec(cell, pbc, pos[p_idx] - pos[m_idx])
                        dist = np.linalg.norm(d_vec)
                        dist_records.append([jobid, f_idx, p_idx, m_idx, round(dist, 4)])

        return stats, dist_records
    except Exception as e:
        print(f"Error {jobid}: {e}")
        return None

# --- MAIN DRIVER ---

def main():
    PART_1_IDS, PART_2_IDS = get_partition_ids(CSV_PATH)

    print("Step 1: Analyzing Reference State...")
    ref_image = read(REF_PATH, index=1)
    ref_h_counts, ref_types = get_h_counts_and_types(ref_image, find_N_type=True)

    tasks = []
    partitions = {"part_1": PART_1_IDS, "part_2": PART_2_IDS}
    for part_name, job_list in partitions.items():
        for jobid in job_list:
            tasks.append((part_name, jobid, ref_h_counts, ref_types))

    print(f"Step 3: Running Multiprocessing ({multiprocessing.cpu_count()} cores)...")
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_trajectory, tasks)

    final_stats = []
    all_distances = []

    for res in results:
        if res:
            final_stats.append(res[0])
            all_distances.extend(res[1])

    if final_stats:
        header_stats = list(final_stats[0].keys())
        with open("protonation_by_amine_type.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_stats)
            writer.writeheader()
            writer.writerows(final_stats)

        header_dist = ["JobID", "Frame_Index", "N_plus_idx", "N_minus_idx", "Distance_Angstrom"]
        with open("Nplus-minus-distances.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header_dist)
            writer.writerows(all_distances)

        print(f"\nDone! Summary and distance files saved successfully.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
