#!/usr/bin/env python3
import numpy as np
import warnings

from ase import units, Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.filters import FrechetCellFilter
from ase.optimize.basin import BasinHopping

# FAIR-Chem / UMA
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    message="logm result may be inaccurate",
    category=RuntimeWarning,
)

# ---- Config ----
TASK = "omc" 
MODEL_ID_path = "/scratch/lipingliu/benckmarking_OMC_with_DFT/fine-tuning/fine-tune-v10-selection_SOAP_latent/round2_SOAP_with_omc_data/round2_SOAP-v3/inference_ckpt_18000.pt"
DEVICE = "cuda"
file_folder = '/scratch/lipingliu/benckmarking_OMC_with_DFT/mutilayers_all/fine-tune-v10-round2-SOAP/COF-999'

def relax_with_ocp(atoms: Atoms, fmax=0.05, steps=200,
                   round_number=0, outbasename="round", recalc=False) -> Atoms:
    """Cell + ionic relaxation."""
    obj = FrechetCellFilter(atoms)
    
    # Clean up logfile logic
    logfile = None if recalc else f"relax_{outbasename}{round_number}.log"
    opt = LBFGS(obj, logfile=logfile)
    
    opt.run(fmax=fmax, steps=steps)
    return atoms

def relax_with_basin(atoms: Atoms, fmax=0.05, steps=10,
                     round_number=0, outbasename="round") -> Atoms:
    """Basin hopping with local LBFGS optimization."""
    dyn = BasinHopping(
        atoms=atoms,
        temperature=300 * units.kB,
        dr=0.25,
        optimizer=LBFGS,
        fmax=fmax,
        logfile=f"basin_{outbasename}{round_number}.log",
        trajectory=f"basin_{outbasename}{round_number}.traj",
    )
    dyn.run(steps)
    
    # Get minimum
    Emin, best = dyn.get_minimum()
    print(f"[Basin] Lowest energy found in BH round {round_number}: {Emin:.6f} eV")
    
    # IMPORTANT: BasinHopping returns a copy without a calculator.
    # We must re-attach the calculator from the original atoms.
    best.calc = atoms.calc
    return best

def anneal_md(atoms: Atoms, round_number: int,
              base_temp=173, max_temp=300, temp_step=3,
              steps_per_temp=60, dt_fs=1.0, friction_fs_inv=0.02,
              outprefix="MD_round") -> Atoms:

    timestep = dt_fs * units.fs
    friction = friction_fs_inv / units.fs
    n_temp_steps = int((max_temp - base_temp) / temp_step) + 1

    trajfile = f"MD_UMA_final.traj"
    logfile = f"MD_UMA_final.log"

    # 1. Define Trajectory writer (Append mode)
    # Note: We do NOT set interval here. We set it when attaching to dyn.
    traj = Trajectory(trajfile, 'a', atoms)

    # 2. Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=base_temp, force_temp=True)

    # 3. Heating Phase
    for i in range(n_temp_steps):
        T = base_temp + i * temp_step
        dyn = Langevin(
            atoms,
            timestep,
            temperature_K=T,
            friction=friction,
            logfile=logfile,
            loginterval=20,
        )
        # Attach the writer explicitly
        dyn.attach(traj.write, interval=10)
        dyn.run(steps_per_temp)

    # 4. High-T Hold Phase
    dyn = Langevin(
        atoms,
        timestep,
        temperature_K=max_temp,
        friction=friction,
        logfile=logfile,
        loginterval=20,
    )
    dyn.attach(traj.write, interval=10)
    dyn.run(int(100000 / dt_fs))

    traj.close()
    return atoms

def get_lowest_energy_structure(atoms_list):
    """Return and print the structure with the lowest potential energy."""
    if not atoms_list:
        raise ValueError("atoms_list is empty")

    # NOTE: This assumes 'all_relaxed_atoms.traj' already contains calculated energies.
    # If not, this line will crash because there is no calculator attached yet.
    try:
        energies = [atoms.get_potential_energy() for atoms in atoms_list]
    except RuntimeError:
        print("Warning: Trajectory does not contain energies. Returning first frame.")
        return atoms_list[0]

    min_idx = int(np.argmin(energies))
    e_min = energies[min_idx]
    print(f"Lowest energy: {e_min:.6f} eV at index {min_idx}")
    return atoms_list[min_idx]


if __name__ == "__main__":
    # 1. Read input
    atoms_list = read(f"{file_folder}/114/AA/all_relaxed_atoms.traj", index=":")
    
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atoms = get_lowest_energy_structure(atoms_list)

    # 2. Setup Calculator Settings
    settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=True,
        merge_mole=True,
        compile=True,  # <--- CRITICAL FIX: Must be False for your model
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )

    # 3. Initialize Calculator
    predictor = load_predict_unit(MODEL_ID_path, device=DEVICE, workers=2, inference_settings=settings)
    calc = FAIRChemCalculator(predictor, task_name=TASK)
    atoms.calc = calc

    # 4. Run MD
    # <--- CRITICAL FIX: Added required argument 'round_number'
    atoms = anneal_md(atoms, round_number=0) 
    
    print("MD Finished Successfully.")
