#!/usr/bin/env python3
"""
Amine physical-accessibility metric for COFs: probe-accessible surface area
(Shrake-Rupley) of amine groups to a CO2-sized spherical probe, from dry MD.

Headline metrics per material (dimensionless / transferable):
  f_acc               : <(1/N) Sum_i 1[group-SASA_i(t) > thr]>_t  (plan definition) [primary]
  <SASA_group/amine>  : time- & amine-averaged accessible area of the amine GROUP
                        (N + its bonded H) to a CO2 probe (A^2)
Also reported:
  <SASA_N/amine>      : N-only accessible area (N sits behind substituents; small)
  crowding            : mean # heavy framework atoms within CROWD_CUTOFF of the N
  phi                 : CO2-probe accessible void fraction of the cell (pore context)

I/O NOTE: the trajectory is read ONCE, sequentially, in the main process into
numpy arrays; CPU-only workers then receive in-memory array slices (no per-worker
disk access) to avoid saturating the shared filesystem with random frame reads.

CO2 probe r=1.65 A; Bondi vdW radii; full PBC via ase.neighborlist. Amine-N =
non-nitrile N (nitrile-C is 2-coordinated); subtype by H count on N.
"""
import os, sys, json, csv, time
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii, atomic_numbers

# ----------------------------- configuration --------------------------------
PROBE_R      = 1.65
N_SPHERE     = 256
N_LAST       = 10000
STRIDE       = 1
OCC_CUTOFF   = 7.0
CROWD_CUTOFF = 6.0
R_SCALE      = 1.20
FACC_THRESHOLDS = [0.0, 5.0, 10.0, 20.0]   # A^2 on amine-GROUP SASA
FACC_MAIN    = 5.0
PHI_NFRAMES  = 40
PHI_NMC      = 15000
N_BLOCKS     = 10
NUM_WORKERS  = int(os.environ.get('SLURM_CPUS_PER_TASK', min(30, os.cpu_count() or 8)))

BONDI = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Cl': 1.75, 'S': 1.80}

SYSTEMS = {
    'COF-998':  "/scratch/lipingliu/benckmarking_OMC_with_DFT/computational_workflow_for_COF_998_1000/equilibrium_MD/dry114/COF-998/COF-998/MD_UMA_final.traj",
    'COF-999':  "/scratch/lipingliu/benckmarking_OMC_with_DFT/mutilayers_all/fine-tune-v10-round2-SOAP/COF-999/balance_simulation_100ps/114/MD_UMA_final.traj",
    'COF-1000': "/scratch/lipingliu/benckmarking_OMC_with_DFT/computational_workflow_for_COF_998_1000/equilibrium_MD/dry114/COF-1000/MD_UMA_final.traj",
}

_MAXZ = 20
VDW_BY_Z = np.zeros(_MAXZ + 1)
for sym, r in BONDI.items():
    VDW_BY_Z[atomic_numbers[sym]] = r

def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
UNIT = fib_sphere(N_SPHERE)
SHIFTS27 = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])

def build_atoms(numbers, pos, cell, pbc=(True, True, True)):
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=pbc)

def classify_amines(atoms):
    syms = np.array(atoms.get_chemical_symbols())
    i, j, d = neighbor_list('ijd', atoms, cutoff=2.6, self_interaction=False)
    cut = R_SCALE * (covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]])
    m = d <= cut
    i, j = i[m], j[m]
    nbr = defaultdict(list)
    for a, b in zip(i.tolist(), j.tolist()):
        nbr[a].append(b)
    coord = {k: len(v) for k, v in nbr.items()}
    recs = []
    for n in np.where(syms == 'N')[0]:
        neigh = nbr[n]
        Hs = [b for b in neigh if syms[b] == 'H']
        if any(syms[b] == 'C' and coord.get(b, 0) == 2 for b in neigh):
            continue
        subtype = {2: 'primary', 1: 'secondary'}.get(len(Hs), 'tertiary')
        recs.append({'N': int(n), 'Hs': [int(h) for h in Hs], 'subtype': subtype})
    return recs

def sasa_interest(atoms, interest_idx, interest_R):
    numbers = atoms.numbers
    i, j, D = neighbor_list('ijD', atoms, cutoff=OCC_CUTOFF, self_interaction=False)
    order = np.argsort(i, kind='stable')
    i, j, D = i[order], j[order], D[order]
    lo = np.searchsorted(i, interest_idx, side='left')
    hi = np.searchsorted(i, interest_idx, side='right')
    Rj_all = VDW_BY_Z[numbers[j]] + PROBE_R
    dnorm = np.linalg.norm(D, axis=1)
    heavy_j = numbers[j] != 1
    sasa = np.empty(len(interest_idx))
    crowd = np.empty(len(interest_idx))
    for a in range(len(interest_idx)):
        s, e = lo[a], hi[a]
        Rt = interest_R[a]
        crowd[a] = int(np.count_nonzero((dnorm[s:e] < CROWD_CUTOFF) & heavy_j[s:e]))
        if e - s == 0:
            sasa[a] = 4.0 * np.pi * Rt * Rt
            continue
        Dt = D[s:e]; Rj = Rj_all[s:e]
        P = Rt * UNIT
        diff = P[:, None, :] - Dt[None, :, :]
        d2 = np.einsum('pmc,pmc->pm', diff, diff)
        occ = (d2 < (Rj * Rj)[None, :]).any(axis=1)
        sasa[a] = (1.0 - occ.mean()) * 4.0 * np.pi * Rt * Rt
    return sasa, crowd

def worker(args):
    (chunk_id, POS, CELL, numbers, pbc,
     interest_idx, interest_R, n_pos, group_pos) = args
    interest_idx = np.asarray(interest_idx)
    interest_R = np.asarray(interest_R)
    n_pos = np.asarray(n_pos)
    n_amine = len(group_pos)
    sasa_sum = np.zeros(len(interest_idx))
    crowd_sum = np.zeros(n_amine)
    acc_count = np.zeros(n_amine)
    mg, fa = [], []
    for f in range(POS.shape[0]):
        atoms = build_atoms(numbers, POS[f], CELL[f], pbc)
        sasa, crowd = sasa_interest(atoms, interest_idx, interest_R)
        sasa_sum += sasa
        crowd_sum += crowd[n_pos]
        group = np.array([sasa[g].sum() for g in group_pos])
        acc_count += (group > FACC_MAIN)
        mg.append(float(group.mean()))
        fa.append(float((group > FACC_MAIN).mean()))
    return (chunk_id, sasa_sum, crowd_sum, acc_count,
            np.array(mg), np.array(fa), POS.shape[0])

def void_fraction_arrays(POS, CELL, numbers, seed=1):
    rng = np.random.default_rng(seed)
    R = VDW_BY_Z[numbers] + PROBE_R
    ghostR = np.tile(R, len(SHIFTS27))
    phis = []
    for f in range(POS.shape[0]):
        cell = CELL[f]; pos = POS[f]
        ghosts = (pos[None, :, :] + (SHIFTS27 @ cell)[:, None, :]).reshape(-1, 3)
        tree = cKDTree(ghosts)
        pts = rng.random((PHI_NMC, 3)) @ cell
        dist, nn = tree.query(pts, k=4)
        overlap = (dist < ghostR[nn]).any(axis=1)
        phis.append(float((~overlap).mean()))
    return np.array(phis)

def block_se(x, nblocks=N_BLOCKS):
    x = np.asarray(x)
    if len(x) <= 1:
        return 0.0
    nb = min(nblocks, len(x))
    means = np.array([b.mean() for b in np.array_split(x, nb)])
    return float(means.std(ddof=1) / np.sqrt(nb))

def load_frames(path, start, stride):
    """Single-process sequential read of strided frames into numpy arrays."""
    traj = Trajectory(path)
    total = len(traj)
    idxs = list(range(start, total, stride))
    natoms = len(traj[idxs[0]])
    POS = np.empty((len(idxs), natoms, 3), dtype=np.float64)
    CELL = np.empty((len(idxs), 3, 3), dtype=np.float64)
    numbers = None
    for f, k in enumerate(idxs):
        a = traj[k]
        POS[f] = a.positions
        CELL[f] = np.asarray(a.cell)
        if numbers is None:
            numbers = a.numbers.copy()
    return POS, CELL, numbers

def run_system(name, path, outdir):
    t0 = time.time()
    traj = Trajectory(path); total = len(traj); traj = None
    start = max(0, total - N_LAST)
    POS, CELL, numbers = load_frames(path, start, STRIDE)
    nfr = POS.shape[0]
    t_load = time.time() - t0
    print(f"[{name}] loaded {nfr} frames in {t_load:.1f}s; computing SASA ...", flush=True)

    a0 = build_atoms(numbers, POS[0], CELL[0])
    recs = classify_amines(a0)
    n_amine = len(recs)

    interest = []; pos_of = {}
    def add(idx):
        if idx not in pos_of:
            pos_of[idx] = len(interest); interest.append(idx)
        return pos_of[idx]
    n_pos, group_pos = [], []
    for r in recs:
        np_ = add(r['N'])
        gp = [np_] + [add(h) for h in r['Hs']]
        n_pos.append(np_); group_pos.append(np.array(gp))
    interest_idx = np.array(interest)
    interest_R = VDW_BY_Z[numbers[interest_idx]] + PROBE_R
    pbc = (True, True, True)

    bounds = np.array_split(np.arange(nfr), NUM_WORKERS)
    tasks = [(int(b[0]), POS[b[0]:b[-1] + 1], CELL[b[0]:b[-1] + 1], numbers, pbc,
              interest_idx, interest_R, n_pos, group_pos) for b in bounds if len(b) > 0]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        res = list(ex.map(worker, tasks))
    res.sort(key=lambda r: r[0])

    sasa_sum = np.zeros(len(interest_idx)); crowd_sum = np.zeros(n_amine)
    acc_count = np.zeros(n_amine); tot = 0
    mg_series, fa_series = [], []
    for _, ss, cs, ac, mg, fa, nf in res:
        sasa_sum += ss; crowd_sum += cs; acc_count += ac; tot += nf
        mg_series.append(mg); fa_series.append(fa)
    mg_series = np.concatenate(mg_series); fa_series = np.concatenate(fa_series)

    per_interest = sasa_sum / tot
    per_amine_N = np.array([per_interest[p] for p in n_pos])
    per_amine_group = np.array([per_interest[g].sum() for g in group_pos])
    per_amine_crowd = crowd_sum / tot
    per_amine_accfrac = acc_count / tot

    facc = {thr: float((per_amine_group > thr).mean()) for thr in FACC_THRESHOLDS}
    subs = np.array([r['subtype'] for r in recs])
    per_sub = {}
    for s in ('primary', 'secondary', 'tertiary'):
        msk = subs == s
        per_sub[s] = {'n': int(msk.sum()),
                      'mean_group_SASA': float(per_amine_group[msk].mean()) if msk.any() else None,
                      'f_acc': float((per_amine_group[msk] > FACC_MAIN).mean()) if msk.any() else None}

    phi_sub = np.linspace(0, nfr - 1, min(PHI_NFRAMES, nfr), dtype=int)
    phis = void_fraction_arrays(POS[phi_sub], CELL[phi_sub], numbers)

    half = len(mg_series) // 2
    summary = {
        'material': name, 'n_frames': tot, 'n_amine': n_amine,
        'cell_vol_A3': float(a0.get_volume()),
        'f_acc_inst': float(fa_series.mean()), 'f_acc_SE': block_se(fa_series),
        'f_acc_main_thr': FACC_MAIN, 'f_acc_timeavg': facc,
        'mean_group_SASA_per_amine_A2': float(per_amine_group.mean()),
        'group_SASA_SE': block_se(mg_series),
        'mean_N_SASA_per_amine_A2': float(per_amine_N.mean()),
        'mean_crowding_6A': float(per_amine_crowd.mean()),
        'void_fraction_phi': float(phis.mean()),
        'phi_SE': float(phis.std(ddof=1) / np.sqrt(len(phis))),
        'stability_halfwindow': {'group_SASA_firsthalf': float(mg_series[:half].mean()),
                                 'group_SASA_secondhalf': float(mg_series[half:].mean()),
                                 'f_acc_firsthalf': float(fa_series[:half].mean()),
                                 'f_acc_secondhalf': float(fa_series[half:].mean())},
        'per_subtype': per_sub,
        'wall_seconds': round(time.time() - t0, 1),
        '_per_amine_group_sasa': per_amine_group.tolist(),
    }
    with open(os.path.join(outdir, f'{name}_per_amine.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N_atom_index', 'subtype', 'group_SASA_A2', 'N_SASA_A2',
                    f'accessible_time_frac(group>{FACC_MAIN})', 'crowding_6A'])
        for r, gs, ns, af, cr in zip(recs, per_amine_group, per_amine_N,
                                     per_amine_accfrac, per_amine_crowd):
            w.writerow([r['N'], r['subtype'], f'{gs:.3f}', f'{ns:.3f}', f'{af:.3f}', f'{cr:.2f}'])
    return summary

def make_plots(summaries, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [s['material'] for s in summaries]
    x = np.arange(len(names)); cols = ['#c44', '#48a', '#4a4']
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [([s['f_acc_inst'] for s in summaries], [s['f_acc_SE'] for s in summaries],
               f'f_acc  (group-SASA > {FACC_MAIN} $\\AA^2$)'),
              ([s['mean_group_SASA_per_amine_A2'] for s in summaries],
               [s['group_SASA_SE'] for s in summaries], r'$\langle$group-SASA/amine$\rangle$ ($\AA^2$)'),
              ([s['void_fraction_phi'] for s in summaries], [s['phi_SE'] for s in summaries],
               r'void fraction $\phi$ (CO$_2$ probe)')]
    for a, (v, se, lab) in zip(ax, panels):
        a.bar(x, v, yerr=se, capsize=4, color=cols)
        a.set_xticks(x); a.set_xticklabels(names); a.set_title(lab, fontsize=10)
        a.grid(axis='y', alpha=0.3)
    fig.suptitle('Amine physical accessibility to a CO$_2$-sized probe (dry MD, last 100 ps)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, 'accessibility_comparison.png'), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].boxplot([np.array(s['_per_amine_group_sasa']) for s in summaries], labels=names, showmeans=True)
    ax[0].set_ylabel(r'per-amine time-avg group-SASA ($\AA^2$)')
    ax[0].set_title('Per-amine accessibility distribution'); ax[0].grid(axis='y', alpha=0.3)
    subs = ['primary', 'secondary', 'tertiary']; w = 0.25
    for k, s in enumerate(summaries):
        vals = [(s['per_subtype'][su]['f_acc'] or 0.0) for su in subs]
        ax[1].bar(np.arange(len(subs)) + (k - 1) * w, vals, width=w, label=s['material'])
    ax[1].set_xticks(np.arange(len(subs))); ax[1].set_xticklabels(subs)
    ax[1].set_ylabel(f'f_acc (group-SASA > {FACC_MAIN} $\\AA^2$)')
    ax[1].set_title('Accessibility by amine subtype'); ax[1].legend(); ax[1].grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'accessibility_by_subtype.png'), dpi=200); plt.close(fig)

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    only = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(outdir, exist_ok=True)
    summaries = []
    for name, path in SYSTEMS.items():
        if only and name != only:
            continue
        s = run_system(name, path, outdir)
        summaries.append(s)
        print(f"[{name}] f_acc={s['f_acc_inst']:.3f}+/-{s['f_acc_SE']:.3f} | "
              f"<group-SASA/amine>={s['mean_group_SASA_per_amine_A2']:.2f}+/-{s['group_SASA_SE']:.2f} A^2 | "
              f"<N-SASA>={s['mean_N_SASA_per_amine_A2']:.2f} | crowd={s['mean_crowding_6A']:.2f} | "
              f"phi={s['void_fraction_phi']:.3f} | nfr={s['n_frames']} | {s['wall_seconds']}s", flush=True)
    with open(os.path.join(outdir, 'accessibility_summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    if len(summaries) > 1:
        make_plots(summaries, outdir)
        print("Saved plots.", flush=True)
    print("Saved accessibility_summary.json to", outdir, flush=True)

if __name__ == '__main__':
    main()
