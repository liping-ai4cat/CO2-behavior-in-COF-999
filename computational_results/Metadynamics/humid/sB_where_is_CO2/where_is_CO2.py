#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
from pathlib import Path
from ase.io import read
from ase.data import covalent_radii
import warnings

# Suppress pandas chained assignment warnings if necessary
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ================= CONFIGURATION =================
# Data file path
ENERGY_CSV = Path('../../../analysis_output/v7_all/job_results_updated.csv')

# Analysis Parameters
R_SCALE = 1.20
HBOND_HA_MAX = 3.0
HBOND_ANGLE_MIN = 110.0
OUTPUT_CSV = "species_analysis_sB.csv"

# ================= DATA LOADING & FILTERING =================
def load_and_filter_jobs(csv_path):
    """
    Reads the energy CSV and filters job IDs based on deltaG and Ga.
    """
    if not csv_path.exists():
        print(f"Error: Energy CSV not found at {csv_path.resolve()}")
        # Return empty lists so script doesn't crash immediately, but prints 0 jobs
        return [], []

    print(f"Reading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Standardize the ID column name
    # We need a column to use as the job ID. Check for common names.
    id_col = None
    # UPDATED: Added 'job_id' to match the specific CSV format provided
    for col in ['job_id', 'name', 'jobid', 'system', 'id']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        print("Error: Could not find a job ID column (checked 'job_id', 'name', 'jobid', 'system').")
        return [], []
    
    # 2. Ensure numeric columns for filtering
    for col in ['deltaG_new', 'Ga_new']:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing.")
            return [], []
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Apply Filters
    # Part 1: deltaG < 0 AND Ga < 140
    mask_p1 = (df['deltaG_new'] <= 0) & (df['Ga_new'] <= 70)
    part1_ids = df.loc[mask_p1, id_col].astype(str).tolist()

    # Part 2: deltaG < 0 AND Ga > 140
    mask_p2 = (df['deltaG_new'] <= 0) & (df['Ga_new'] > 70)
    part2_ids = df.loc[mask_p2, id_col].astype(str).tolist()

    print(f"Total entries: {len(df)}")
    print(f"Part 1 (dG < 0, Ga < 140): {len(part1_ids)} jobs found.")
    print(f"Part 2 (dG < 0, Ga > 140): {len(part2_ids)} jobs found.")

    return part1_ids, part2_ids

# Initialize the ID lists
PART_1_IDS, PART_2_IDS = load_and_filter_jobs(ENERGY_CSV)


# ================= HELPER FUNCTIONS =================

def mic_dist_vec(cell, pbc, r1, r2):
    """Calculates vector r2 - r1 and distance under PBC."""
    diff = r2 - r1
    if np.any(pbc):
        # Cartesian to Fractional
        icell = np.linalg.inv(cell.T)
        s = np.dot(icell, diff)
        # Apply PBC
        s -= np.round(s * pbc)
        # Fractional back to Cartesian
        diff = np.dot(cell.T, s)
    return np.linalg.norm(diff), diff

def get_angle(v1, v2):
    """Returns angle in degrees between two vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12: return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

def is_bonded(atoms, i, j, cell, pbc):
    """Checks if atoms i and j are bonded based on covalent radii."""
    cutoff = R_SCALE * (covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number])
    d, _ = mic_dist_vec(cell, pbc, atoms.positions[i], atoms.positions[j])
    return d <= cutoff

def get_bonded_neighbors(atoms, i, cell, pbc, elements=None):
    """Returns indices of atoms bonded to atom i."""
    return [j for j in range(len(atoms))
            if i != j
            and (not elements or atoms[j].symbol in elements)
            and is_bonded(atoms, i, j, cell, pbc)]

def identify_species(atoms, condition="dry"):
    """
    Identifies the chemical species of the CO2 moiety.
    """
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    # Handle cases with no cell defined
    if cell is None or not np.any(cell):
        cell = np.eye(3)*100; pbc = [False]*3

    syms = atoms.get_chemical_symbols()
    all_O = [i for i, s in enumerate(syms) if s == 'O']

    # --- 1. Define Oxygen Subsets based on Condition ---

    # Moiety is always the last 2 Oxygens
    if len(all_O) < 2:
        return "Error (Not enough O)", {"h3o_count": 0}

    moiety_O = all_O[-2:]

    # Define Water Oxygens based on condition
    if condition == "dry":
        water_O = []
    elif condition == "humid":
        # HUMID: User specified range [-66:-2]
        if len(all_O) >= 66:
            water_O = all_O[-66:-2]
        else:
            # Fallback if trajectory has fewer atoms than expected
            water_O = all_O[:-2]
    else:
        water_O = []

    # --- 2. SPECIES ID ---
    moiety_C = None
    N_idx = None
    species_label = "Free CO2"

    # Find the C bonded to the moiety Oxygens
    for c_idx in [i for i, s in enumerate(syms) if s == 'C']:
        if any(is_bonded(atoms, c_idx, o, cell, pbc) for o in moiety_O):
            moiety_C = c_idx
            break

    if moiety_C is not None:
        N_neighbors = get_bonded_neighbors(atoms, moiety_C, cell, pbc, elements=['N'])
        if N_neighbors:
            N_idx = N_neighbors[0]
            # Count H on Nitrogen
            nH_N = len(get_bonded_neighbors(atoms, N_idx, cell, pbc, elements=['H']))
            # Count H on Moiety Oxygens
            nH_O = sum(len(get_bonded_neighbors(atoms, o, cell, pbc, elements=['H'])) for o in moiety_O)

            if nH_N == 1 and nH_O == 1: species_label = "NHCOOH (Carbamic Acid)"
            elif nH_N == 1 and nH_O == 0: species_label = "NHCOO- (Carbamate)"
            elif nH_N == 0 and nH_O == 1: species_label = "NCOOH"
            elif nH_N == 0 and nH_O == 0: species_label = "NCOO (Carbamate Radical/Ion)"
            elif nH_N == 2 and nH_O == 0: species_label = "NH2COO (Zwitterion)"
            elif nH_N == 1 and nH_O == 2: species_label = "HOCOH+NH (Protonated)"
            else: species_label = f"Other ({nH_N}H_N, {nH_O}H_O)"
        else:
            species_label = "Unknown (C found, no N attached)"

    # --- 3. H3O+ COUNT ---
    num_h3o = 0
    # Loop runs only if water_O is not empty (i.e., humid case)
    for w in water_O:
        h_neighbors = get_bonded_neighbors(atoms, w, cell, pbc, elements=['H'])
        if len(h_neighbors) == 3:
            num_h3o += 1

    return species_label, {"h3o_count": num_h3o}

# ================= MAIN EXECUTION =================

def main():
    all_results = []

    # Organize jobs by partition for easy path finding
    partitions = {
        "part_1": PART_1_IDS,
        "part_2": PART_2_IDS
    }

    print(f"\nStarting analysis of sB trajectories...")

    for part_name, job_list in partitions.items():
        print(f"📂 Processing {part_name} ({len(job_list)} jobs)...")

        for jobid in job_list:
            # Construct path: ../{jobid}_sB.traj
            # NOTE: Adjust this path logic if your files are stored differently!
            traj_path = Path(f"../{jobid}_sB.traj")

            if not traj_path.exists():
                # Try simple fallback (sometimes jobid is just an index)
                # traj_path = Path(f"../{jobid}.traj") 
                if not traj_path.exists():
                    print(f"  ⚠️ File not found: {traj_path}")
                    continue

            try:
                # Read entire trajectory
                traj = read(traj_path, index=':')
                n_frames = len(traj)

                # Dictionary to count species occurrences in this trajectory
                species_counts = {}

                for i, atoms in enumerate(traj):
                    label, info = identify_species(atoms)
                    species_counts[label] = species_counts.get(label, 0) + 1

                # Determine MAJORITY species for this job
                if species_counts:
                    dominant_species = max(species_counts, key=species_counts.get)

                    # Add summary row
                    row = {
                        "jobid": jobid,
                        "partition": part_name,
                        "total_frames": n_frames,
                        "dominant_species": dominant_species,
                        "count_NHCOOH": species_counts.get("NHCOOH (Carbamic Acid)", 0),
                        "count_NHCOO-": species_counts.get("NHCOO- (Carbamate)", 0),
                        "count_NH2COO": species_counts.get("NH2COO (Zwitterion)", 0),
                        "count_NCOO-": species_counts.get("NCOO (Carbamate Radical/Ion)", 0),
                        "count_NCOOH": species_counts.get("NCOOH", 0),
                        "count_Other": sum(v for k,v in species_counts.items() if "Other" in k or "Unknown" in k or "Free" in k)
                    }
                    all_results.append(row)

            except Exception as e:
                print(f"  ❌ Error processing {jobid}: {e}")

    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Analysis complete. Data saved to {OUTPUT_CSV}")
        print(df.head())
        print(f"\nDistribution of Dominant Species:")
        print(df['dominant_species'].value_counts())
    else:
        print("\n❌ No data collected.")

if __name__ == "__main__":
    main()
