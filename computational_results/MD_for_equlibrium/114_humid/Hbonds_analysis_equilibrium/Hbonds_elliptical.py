import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.geometry import get_distances
import multiprocessing
import os
from collections import defaultdict
import scipy.ndimage

# --- 1. Wernet Ellipse Parameters ---
def get_ellipse_coords(pair_key):
    a = 0.50
    b = 45.0
    centers = {
        #'O-O': 2.90, 'N-O': 3.05, 'O-N': 3.05,
        #'O-Cl': 3.25, 'N-Cl': 3.42, 
        #'N-N': 3.10,
        'Ow-Ow': 2.90,
        'N-O': 3.05,
        'O-N': 3.05,
    }
    if pair_key not in centers: return None, None
    r0 = centers[pair_key]
    r_min = r0 - a
    r_max = r0 + a
    r_values = np.linspace(r_min, r_max, 200)
    term = 1 - ((r_values - r0) / a)**2
    term[term < 0] = 0
    theta_values = 180.0 - b * np.sqrt(term)
    return r_values, theta_values

# --- 2. Worker Function ---
def process_chunk_geometry(chunk_args):
    filename, start, stop, stride = chunk_args
    try:
        # read index uses standard python slicing: start:stop:stride
        traj_chunk = read(filename, index=f'{start}:{stop}:{stride}')
    except Exception as e:
        print(f"Error reading chunk {start}:{stop}: {e}")
        return {}

    # Define Species
    donor_species = ['N', 'O']
    acceptor_species = ['N', 'O']
    h_symbol = 'H'

    chunk_results = defaultdict(list)

    for frame in traj_chunk:
        donors = [atom.index for atom in frame if atom.symbol in donor_species]
        acceptors = [atom.index for atom in frame if atom.symbol in acceptor_species]
        hydrogens = [atom.index for atom in frame if atom.symbol == h_symbol]

        if not donors or not hydrogens: continue

        _, d_h_dists = get_distances(frame.positions[donors], frame.positions[hydrogens],
                                     cell=frame.cell, pbc=frame.pbc)

        for d_i, d_idx in enumerate(donors):
            d_elem = frame[d_idx].symbol
            bonded_h_local_indices = np.where(d_h_dists[d_i] < 1.25)[0]
            if len(bonded_h_local_indices) == 0: continue

            for h_local_idx in bonded_h_local_indices:
                h_idx = hydrogens[h_local_idx]
                _, h_a_dists = get_distances(frame.positions[[h_idx]], frame.positions[acceptors],
                                             cell=frame.cell, pbc=frame.pbc)
                h_a_dists = h_a_dists.flatten()

                # Loose filter
                candidate_a_indices = np.where((h_a_dists < 5.0) & (h_a_dists > 1.4))[0]

                for a_local_idx in candidate_a_indices:
                    a_idx = acceptors[a_local_idx]
                    a_elem = frame[a_idx].symbol
                    if a_idx == d_idx: continue

                    r_XY = frame.get_distance(d_idx, a_idx, mic=True)
                    v_hd = frame.get_distance(h_idx, d_idx, mic=True, vector=True)
                    v_ha = frame.get_distance(h_idx, a_idx, mic=True, vector=True)
                    n_hd = np.linalg.norm(v_hd)
                    n_ha = np.linalg.norm(v_ha)

                    if n_hd == 0 or n_ha == 0: continue

                    dot = np.dot(v_hd, v_ha)
                    cos_theta = dot / (n_hd * n_ha)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta = np.degrees(np.arccos(cos_theta))

                    pair_key = f"{d_elem}-{a_elem}"
                    if r_XY < 4.5 and theta > 100:
                        chunk_results[pair_key].append((r_XY, theta))
    return chunk_results

# --- 3. Main Driver ---
def compute_geometry_parallel(traj_file, n_cores=None, stride=1, last_n_frames=10000):
    if n_cores is None:
        n_cores = multiprocessing.cpu_count()

    # Get total frames to find the "last 10,000" start point
    t = Trajectory(traj_file, mode='r')
    total_frames = len(t)
    t.close()

    # Determine the slice range
    start_frame = max(0, total_frames - last_n_frames)
    end_frame = total_frames
    actual_count = end_frame - start_frame

    print(f"--- Analysis: Last {actual_count} frames (Stride={stride}) on {n_cores} cores ---")

    # Divide the 10,000 frame window into chunks for multiprocessing
    ranges = []
    frames_per_core = int(np.ceil(actual_count / n_cores))
    
    for i in range(0, n_cores):
        chunk_start = start_frame + (i * frames_per_core)
        chunk_end = min(chunk_start + frames_per_core, end_frame)
        if chunk_start < end_frame:
            ranges.append((traj_file, chunk_start, chunk_end, stride))

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.map(process_chunk_geometry, ranges)

    print("Analysis complete. Aggregating results...")

    final_data = defaultdict(list)
    for res in results:
        for pair_key, values in res.items():
            final_data[pair_key].extend(values)

    # --- Plotting & Export Loop ---
    for pair_key, data in final_data.items():
        if len(data) < 10: # Lowered threshold slightly for testing
            print(f"Skipping {pair_key}: Not enough data points ({len(data)}).")
            continue

        data_arr = np.array(data)
        rs = data_arr[:, 0]
        thetas = data_arr[:, 1]

        x_range = [2.4, 4.0]
        y_range = [130, 180]
        bins = [100, 100]

        H, xedges, yedges = np.histogram2d(rs, thetas, bins=bins, range=[x_range, y_range])
        H_smoothed = scipy.ndimage.uniform_filter(H, size=3)

        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        X_grid, Y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')

        export_data = np.column_stack((X_grid.flatten(), Y_grid.flatten(), H_smoothed.flatten()))
        csv_name = f"HBond_Data_{pair_key}.csv"
        np.savetxt(csv_name, export_data, delimiter=",",
                   header="Distance_Angstrom,Angle_Degree,Smoothed_Density", comments='')
        print(f"Data saved: {csv_name}")

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(H_smoothed.T, interpolation='bilinear', origin='lower',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap='nipy_spectral', aspect='auto')

        plt.colorbar(im, ax=ax).set_label('Smoothed Density')

        ex, ey = get_ellipse_coords(pair_key)
        if ex is not None:
            ax.plot(ex, ey, color='black', linewidth=2, linestyle='-')

        ax.set_xlabel(f'{pair_key} Distance ($\AA$)')
        ax.set_ylabel(f'X - H - Y Angle (Degree)')
        ax.set_title(f'Smoothed Geometry: {pair_key} (Last 10k)')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"HBond_Smooth_{pair_key}.png")
        plt.close()

if __name__ == "__main__":
    # Stride set to 1, processing last 10,000 images
    compute_geometry_parallel('../MD_UMA_final.traj', stride=1, last_n_frames=10000)
