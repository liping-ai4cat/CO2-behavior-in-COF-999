from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

# PLUMED (ASE wrapper)
from ase.calculators.plumed import Plumed
from ase import units
from ase.io import read
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
import numpy as np
from os.path import exists
import sys
import os  # Added for folder name extraction

# key input
C_idx = 1944 + 1  # the index of intersted C in gas phase CO2

# (1) Automatic N_idx generate based on current folder name
try:
    current_folder_name = os.path.basename(os.getcwd())
    N_idx = int(current_folder_name) + 1
    print(f"Set N_idx to {N_idx} based on folder name '{current_folder_name}'")
except ValueError:
    print("Warning: Current folder name is not an integer. Please check N_idx logic.")
    sys.exit(1)

# ---------------------------- Config ----------------------------
TASK = "omc"          # "omc" for inorganic clusters
MODEL_ID = "uma-s-1p1"
MODEL_ID_path = "/projects/caiw/lipingliu/COF-999/fine-tuning/fine-tuned-uma-s-v10-round2-SOAP-v3/inference_ckpt_18000.pt"
DEVICE = "cuda"       # "cuda" (GPU) or "cpu"
structure_dir = "/projects/caiw/lipingliu/COF-999/metadynamics_using_uma/fine-tuned-uma-s-v10-round2-SOAP-v3/CO2-single-COF-999-humid"

# MD / MTD settings
DT_FS = 1.0           # fs, MD integrator timestep
TEMP_K = 300.0        # K
FRIC_FS_INV = 0.01    # 1/fs (Langevin friction gamma)
N_STEPS = 2000000     # total MD steps
TRAJ_WRITE_INT = 25   # write atoms every N steps
TRAJ_FILE = "meta.traj"

# ---------------------------- Automatic Restart Logic ----------------------------
def check_traj(path):
    """Returns True if path exists and has > 0 frames, else False."""
    if exists(path):
        try:
            with Trajectory(path, 'r') as t:
                return len(t) > 0
        except Exception:
            return False
    return False

# Defaults (Start from scratch)
STRUCTURE = f'{structure_dir}/plus_CO2.traj'
restart = False       # Don't skip annealing
restart_flag = False  # Don't restart PLUMED (HILLS)

# Priority 1: Resume production (meta.traj)
if check_traj(TRAJ_FILE):
    STRUCTURE = TRAJ_FILE
    restart = True       # Skip annealing
    restart_flag = True  # Resume PLUMED
    print(f"Resuming production from {TRAJ_FILE}")

# Priority 2: Start fresh MetaD from heated structure (MD_round_heating.traj)
elif check_traj('./MD_round_heating.traj'):
    STRUCTURE = './MD_round_heating.traj'
    restart = True       # Skip annealing (already heated)
    restart_flag = False # New PLUMED run
    print(f"Starting fresh MetaD from heated structure {STRUCTURE}")

# Priority 3: Start from scratch
else:
    print(f"Starting from scratch: {STRUCTURE}")

PBC = True
# ---------------------------- Functions ----------------------------

def find_partners(atoms, center_idx, partner_element, max_dist):
    """
    Return 1-based indices of partner_element within max_dist of center_idx,
    accounting for Periodic Boundary Conditions (PBC).
    """
    c0 = center_idx - 1
    syms = np.array(atoms.get_chemical_symbols())
    partner_indices = np.where(syms == partner_element)[0]

    if len(partner_indices) == 0:
        raise RuntimeError(f"No atom of type {partner_element} found in structure.")

    distances = atoms.get_distances(c0, partner_indices, mic=True)

    out = []
    for idx, dist in zip(partner_indices, distances):
        if dist <= max_dist:
            out.append(int(idx) + 1)

    if not out:
        raise RuntimeError(f"No {partner_element} within {max_dist} Å of atom {center_idx}")

    return out

def find_Hs_near_N_and_Os_near_C(atoms, N_idx, C_idx, max_NH=1.25, max_CO=1.50):
    """Return (Hs_list, Os_list) with 1-based indices."""
    Hs = find_partners(atoms, N_idx, 'H', max_NH)
    Os = find_partners(atoms, C_idx, 'O', max_CO)
    return Hs, Os

# Anneal settings
ANNEAL_BASE_T = 295   # K
ANNEAL_MAX_T  = 300   # K
ANNEAL_DT_FS  = 1.0   # fs
ANNEAL_FRIC_FS_INV = 0.02
ANNEAL_TEMP_STEP = 1
ANNEAL_STEPS_PER_TEMP = 10

# PLUMED (WT-MTD) parameters
BIASFACTOR = 10
HEIGHT = 0.10        # eV
PACE   = 200         # deposit stride
HILLS_FILE = "HILLS"
COLVAR_FILE = "COLVAR"

def set_hydrogen_mass(atoms, new_mass=3.0):
    masses = atoms.get_masses()
    symbols = np.array(atoms.get_chemical_symbols())
    masses[symbols == 'H'] = new_mass
    atoms.set_masses(masses)
    return atoms

def anneal_md(
    atoms,
    base_temp=ANNEAL_BASE_T,
    max_temp=ANNEAL_MAX_T,
    temp_step=ANNEAL_TEMP_STEP,
    steps_per_temp=ANNEAL_STEPS_PER_TEMP,
    dt_fs=ANNEAL_DT_FS,
    friction_fs_inv=ANNEAL_FRIC_FS_INV,
    outprefix="MD_round",
):
    timestep = dt_fs * units.fs
    friction = friction_fs_inv / units.fs
    n_temp_steps = int((max_temp - base_temp) / temp_step) + 1

    traj = f"{outprefix}_heating.traj"
    log = f"{outprefix}_plusCO2.log"

    MaxwellBoltzmannDistribution(atoms, temperature_K=base_temp, force_temp=True)
    Stationary(atoms)
    ZeroRotation(atoms)

    for i in range(n_temp_steps):
        T = base_temp + i * temp_step
        dyn = Langevin(
            atoms, timestep=timestep, temperature_K=T, friction=friction,
            trajectory=traj, logfile=log, loginterval=10,
            append_trajectory=(i != 0)
        )
        dyn.run(steps_per_temp)

    hold_steps = int(1000 / dt_fs)  # 1 ps
    dyn = Langevin(
        atoms, timestep=timestep, temperature_K=max_temp, friction=friction,
        trajectory=traj, logfile=log, loginterval=10, append_trajectory=True
    )
    dyn.run(hold_steps)

    return atoms

def build_plumed_input(n_idx, c_idx, h_list, o_list, atoms, atoms_original, restart_flag=False):
    """
    Sets up PLUMED.
    Added restart_flag argument to append global RESTART and RESTART=YES when needed.
    """
    # --- 1. Calculate Current C-N Distance for Grid ---
    # (3) make d1_cur knows pbc conditions and uses atoms_original for consistency
    d1_cur = float(atoms_original.get_distance(n_idx-1, c_idx-1, mic=True)) 

    # --- 2. Define Grids ---
    GRID_MIN_D = 0.5
    GRID_MAX_D = max(d1_cur + 1.5, 5.5)

    GRID_MIN_CN = -5.0
    GRID_MAX_CN = 3.0

    print(f"Grid CV1 (Dist): {GRID_MIN_D} to {GRID_MAX_D}")
    print(f"Grid CV2 (Coord): {GRID_MIN_CN} to {GRID_MAX_CN}")

    # --- 3. Format Lists for PLUMED ---
    h_str = ",".join(map(str, h_list))
    o_str = ",".join(map(str, o_list))

    # metaD kernel settings
    SIGMA_D   = 0.2
    SIGMA_CN  = 0.15
    R0_CUT = 1.4

    lines = [
        "UNITS LENGTH=A ENERGY=eV TIME=fs"
    ]

    # --- CRITICAL: Global Restart ---
    # This tells PLUMED to append to existing output files (COLVAR)
    if restart_flag:
        lines.append("RESTART")

    # --- CV Definitions ---
    lines.extend([
        # --- CV1: Distance N-C ---
        f"d1: DISTANCE ATOMS={n_idx},{c_idx}",

        # --- CV2: Coordination Combination --
        f"cn1: COORDINATION GROUPA={n_idx} GROUPB={h_str} R_0={R0_CUT} NN=6 MM=12",
        f"cn2: COORDINATION GROUPA={o_str} GROUPB={h_str} R_0={R0_CUT} NN=6 MM=12",
        f"cv2: COMBINE ARG=cn1,cn2 COEFFICIENTS=1.0,-2.0 PERIODIC=NO",

        # --- Walls ---
        f"lw1: LOWER_WALLS ARG=d1 AT={GRID_MIN_D + 0.3:.3f} KAPPA=1000 EXP=4",
        f"uw1: UPPER_WALLS ARG=d1 AT={GRID_MAX_D - 0.3:.3f} KAPPA=1000 EXP=4",
        f"lw2: LOWER_WALLS ARG=cv2 AT={GRID_MIN_CN + 0.2:.3f} KAPPA=1000 EXP=4",
        f"uw2: UPPER_WALLS ARG=cv2 AT={GRID_MAX_CN - 0.2:.3f} KAPPA=1000 EXP=4",
    ])

    # --- Metadynamics Line construction ---
    metad_line = (
        "metad: METAD "
        f"ARG=d1,cv2 SIGMA={SIGMA_D},{SIGMA_CN} "
        f"HEIGHT={HEIGHT} PACE={PACE} BIASFACTOR={BIASFACTOR} FILE={HILLS_FILE} "
        f"GRID_MIN={GRID_MIN_D:.3f},{GRID_MIN_CN:.3f} "
        f"GRID_MAX={GRID_MAX_D:.3f},{GRID_MAX_CN:.3f} "
        f"GRID_BIN=200,300"
    )

    # --- Bias Restart ---
    # This tells PLUMED to read the HILLS file and restore the bias potential
    if restart_flag:
        metad_line += " RESTART=YES"

    lines.append(metad_line)
    lines.append(f"PRINT STRIDE=25 ARG=d1,cn1,cn2,cv2,metad.bias FILE={COLVAR_FILE}")

    return lines

def run_metadynamics(
    atoms,
    plumed_input,
    dt_fs=DT_FS,
    temperature_K=TEMP_K,
    friction_fs_inv=FRIC_FS_INV,
    n_steps=N_STEPS,
    traj_file=TRAJ_FILE,
    traj_interval=TRAJ_WRITE_INT,
    append_traj=False # ADDED: Flag to control file mode
):
    """Attach PLUMED to the current calculator and run WT-MTD with Langevin NVT."""
    kT = units.kB * temperature_K  # eV

    # Wrap existing calculator with PLUMED
    atoms.calc = Plumed(
        calc=atoms.calc,
        input=plumed_input,
        timestep=dt_fs,
        atoms=atoms,
        kT=kT,
    )

    # Langevin MD
    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction_fs_inv / units.fs,
    )

    # FIX: Use 'a' if restarting, 'w' if fresh
    file_mode = 'a' if append_traj else 'w'
    traj = Trajectory(traj_file, file_mode, atoms)
    dyn.attach(traj.write, interval=traj_interval)

    # Run production MD
    dyn.run(n_steps)

# ---------------------------- Main -----------------------------
if __name__ == "__main__":
    # Structure
    if restart:
        atoms = read(STRUCTURE, index=-1)
        # Try to read original structure for index mapping consistency
        try:
            atoms_original = read("./MD_round_heating.traj", index=-1)
        except:
            atoms_original = read(STRUCTURE, index=0)
    else:
        atoms = read(STRUCTURE, index=MMM)
        atoms_original = None
        # (2) Check atoms_original to make consistent CV ranges.
        # Ensure it is defined even if not restarting so build_plumed_input works.

    atoms.pbc = PBC
    atoms.wrap()
    set_hydrogen_mass(atoms)

    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    settings = InferenceSettings(
    tf32=True,
    activation_checkpointing=True,
    merge_mole=True,
    compile=True,
    external_graph_gen=False,
    internal_graph_gen_version=2,)

    # UMA -> FAIRChemCalculator
    predictor = load_predict_unit(MODEL_ID_path, device=DEVICE, workers=4, inference_settings=settings)
    calc = FAIRChemCalculator(predictor, task_name=TASK)
    atoms.calc = calc

    # Optional: gentle anneal / thermalization
    if restart:
        print('The MD is restarted (skipping anneal)')
    else:
        anneal_md(atoms)

    # automatically set up collective variables
    # Use atoms_original to ensure input CV range/indices are consistent
    if atoms_original:
        Hs, Os = find_Hs_near_N_and_Os_near_C(atoms_original, N_idx, C_idx)
    else:
        # Fallback (should typically be covered by atoms_original above)
        Hs, Os = find_Hs_near_N_and_Os_near_C(atoms, N_idx, C_idx)
        atoms_original = atoms
    atoms_original.pbc = PBC # Ensure original also has PBC set for distance check

    # Metadynamics production
    plumed_input = build_plumed_input(N_idx, C_idx, Hs, Os, atoms, atoms_original, restart_flag=restart_flag)

    # FIX: Pass restart status to run_metadynamics to control file writing mode
    run_metadynamics(atoms, plumed_input, append_traj=restart)
