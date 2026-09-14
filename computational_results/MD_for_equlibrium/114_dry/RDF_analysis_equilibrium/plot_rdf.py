import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.geometry import get_distances
import multiprocessing
import os

# --- Worker Function (Runs on each core) ---
def process_chunk(chunk_args):
    """
    Reads a specific range of frames and calculates the histogram for a specific atom pair.
    """
    filename, start, stop, r_max, n_bins, atom_pair = chunk_args
    elem1, elem2 = atom_pair

    # Re-open file inside worker
    try:
        traj_chunk = read(filename, index=f'{start}:{stop}')
    except Exception as e:
        print(f"Error reading {filename} [{start}:{stop}]: {e}")
        return np.zeros(n_bins), 0.0, 0, 0

    radii = np.linspace(0, r_max, n_bins + 1)
    local_hist = np.zeros(n_bins)

    # Accumulators for normalization
    local_rho2_sum = 0.0  # Density of the "target" atom (atom2)
    local_n1_count_sum = 0 # Count of "reference" atom (atom1)
    valid_frames = 0

    is_homo_atomic = (elem1 == elem2)

    for atoms in traj_chunk:
        # 1. Identify indices for both elements
        idx1 = [a.index for a in atoms if a.symbol == elem1]
        idx2 = [a.index for a in atoms if a.symbol == elem2]

        n1 = len(idx1)
        n2 = len(idx2)

        # Skip if atoms missing
        if n1 == 0 or n2 == 0:
            continue
        if is_homo_atomic and n1 < 2:
            continue

        # 2. Geometry / Distance Calculation
        pos1 = atoms.positions[idx1]
        pos2 = atoms.positions[idx2]

        # get_distances args: p1, p2, cell, pbc. 
        _, dist_mat = get_distances(pos1, pos2, cell=atoms.cell, pbc=atoms.pbc)

        # 3. Filter Distances
        if is_homo_atomic:
            # Square matrix: We only want upper triangle (unique pairs, i < j)
            valid_dists = dist_mat[np.triu_indices(n1, k=1)]
        else:
            # Rectangular matrix: We want ALL distances between set 1 and set 2
            valid_dists = dist_mat.flatten()

        # 4. Histogram
        hist, _ = np.histogram(valid_dists, bins=radii)
        local_hist += hist

        # 5. Stats for Normalization
        vol = atoms.get_volume()
        local_rho2_sum += n2 / vol
        local_n1_count_sum += n1
        valid_frames += 1

    return local_hist, local_rho2_sum, local_n1_count_sum, valid_frames

# --- Main Driver ---
def compute_rdf_parallel(traj_files, atom_pair=('N', 'N'), r_max=10.0, n_bins=200, n_cores=None):
    """
    Calculates RDF for any pair of atoms across a LIST of trajectory files.
    Saves plot to .png and data to .csv.
    """
    elem1, elem2 = atom_pair

    # Ensure input is a list
    if isinstance(traj_files, str):
        traj_files = [traj_files]

    if n_cores is None:
        n_cores = multiprocessing.cpu_count()

    print(f"--- Starting Parallel RDF for {elem1}-{elem2} ---")
    print(f"Processing {len(traj_files)} file(s): {traj_files}")

    # 1. Prepare Chunks across all files
    all_ranges = []
    total_files_frames = 0

    for traj_file in traj_files:
        try:
            # Quickly get frame count without loading atoms
            t = Trajectory(traj_file, mode='r')
            n_frames = len(t)
            t.close()
            
            if n_frames == 0:
                print(f"Warning: {traj_file} is empty. Skipping.")
                continue
            
            total_files_frames += n_frames

            # Divide this file into chunks for the cores
            chunk_size = max(1, int(np.ceil(n_frames / n_cores)))

            for i in range(0, n_frames, chunk_size):
                stop = min(i + chunk_size, n_frames)
                all_ranges.append((traj_file, i, stop, r_max, n_bins, atom_pair))
                
        except Exception as e:
            print(f"Error accessing {traj_file}: {e}")

    print(f"Total frames detected: {total_files_frames}")
    print(f"Total parallel tasks: {len(all_ranges)}")

    if not all_ranges:
        print("No valid frames to process.")
        return

    # 2. Multiprocessing
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(process_chunk, all_ranges)

    print("Aggregating results...")

    # 3. Aggregate
    total_hist = np.zeros(n_bins)
    total_rho2 = 0.0
    total_n1 = 0
    total_frames = 0

    for res in results:
        h, rho2, n1, vf = res
        total_hist += h
        total_rho2 += rho2
        total_n1 += n1
        total_frames += vf

    if total_frames == 0:
        print(f"Error: No valid frames found containing both {elem1} and {elem2}.")
        return

    # 4. Normalization
    avg_rho2 = total_rho2 / total_frames
    avg_n1 = total_n1 / total_frames

    dr = r_max / n_bins
    radii = np.linspace(0, r_max, n_bins + 1)
    r_mid = 0.5 * (radii[1:] + radii[:-1])
    shell_volumes = 4.0 * np.pi * (r_mid ** 2) * dr

    if elem1 == elem2:
        norm_factor = total_frames * (avg_n1 * avg_rho2 * shell_volumes) / 2.0
    else:
        norm_factor = total_frames * (avg_n1 * avg_rho2 * shell_volumes)

    g_r = total_hist / norm_factor

    # 5. Export Data to CSV
    csv_filename = f'rdf_{elem1}{elem2}_averaged.csv'
    # Combine r and g(r) into columns
    data_to_save = np.column_stack((r_mid, g_r))
    np.savetxt(
        csv_filename, 
        data_to_save, 
        delimiter=',', 
        header=f'Distance_r_A,g_r_{elem1}{elem2}', 
        comments=''
    )
    print(f"Data saved to {csv_filename}")

    # 6. Plotting
    plt.figure(figsize=(4, 4.5), dpi=200)
    plt.plot(r_mid, g_r, color='navy', linewidth=2, label=f'{elem1}-{elem2} RDF')
    plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Distance $r$ ($\AA$)')
    plt.ylabel(f'$g_{{{elem1}{elem2}}}(r)$')

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_filename = f'rdf_{elem1}{elem2}_averaged.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() # Uncomment if running locally with a display

# --- Usage ---
if __name__ == "__main__":
    # Define your list of files
    my_files = ['../MD_UMA_final.traj']

    # Run analysis
    compute_rdf_parallel(my_files, atom_pair=('N', 'N'))
    compute_rdf_parallel(my_files, atom_pair=('C', 'C'))
    compute_rdf_parallel(my_files, atom_pair=('C', 'N'))
