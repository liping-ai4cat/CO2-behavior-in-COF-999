#!/usr/bin/env python3
import numpy as np
from copy import deepcopy  # optional; not required if using streaming writes only

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

import warnings
warnings.filterwarnings(
    "ignore",
    message="logm result may be inaccurate",
    category=RuntimeWarning,
)

# ---- Config ----
TASK = "omc"  # use "omol" ONLY for true molecules; use "oc20" for slabs/surfaces
MODEL_ID_path = "/scratch/lipingliu/benckmarking_OMC_with_DFT/fine-tuning/fine-tune-v10-selection_SOAP_latent/round2_SOAP_with_omc_data/round2_SOAP-v3/inference_ckpt_18000.pt"
DEVICE = "cuda"
file_folder = '/scratch/lipingliu/mutilayers_all/fine-tune-v10-round2-SOAP/COF-999/114/backup'


def relax_with_ocp(atoms: Atoms, fmax=0.05, steps=200,
                   round_number=0, outbasename="round", recalc=False) -> Atoms:
    """Cell + ionic relaxation with UMA (requires atoms.calc to be set)."""
    # For periodic systems, use FrechetCellFilter; for clusters you may want plain atoms.
    obj = FrechetCellFilter(atoms)

    if recalc:
        opt = LBFGS(obj, logfile=None)
    else:
        opt = LBFGS(obj, logfile=f"relax_{outbasename}{round_number}.log")

    opt.run(fmax=fmax, steps=steps)
    return atoms

def relax_with_basin(atoms: Atoms, fmax=0.05, steps=10,
                     round_number=0, outbasename="round") -> Atoms:
    """Basin hopping with local LBFGS optimization; return lowest-energy structure."""
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

    # Use BasinHopping.get_minimum() defined in the source you pasted
    Emin, best = dyn.get_minimum()   # best is an Atoms object
    print(f"[Basin] Lowest energy found in BH round {round_number}: {Emin:.6f} eV")

    # Restore UMA calculator for further use
    best.calc = atoms.calc

    return best

def anneal_md(atoms: Atoms, round_number: int,
              base_temp=173, max_temp=473, temp_step=3,
              steps_per_temp=50, dt_fs=1.0, friction_fs_inv=0.02,
              outprefix="MD_round") -> Atoms:
    """Heating–hold–cooling cycle under Langevin (NVT), then basin + OCP relax."""
    timestep = dt_fs * units.fs
    friction = friction_fs_inv / units.fs
    n_temp_steps = int((max_temp - base_temp) / temp_step) + 1

    traj = "uma_run.traj"
    log = f"{outprefix}{round_number}.log"

    # Initialize velocities once at base temperature
    MaxwellBoltzmannDistribution(atoms, temperature_K=base_temp, force_temp=True)

    # Heating
    for i in range(n_temp_steps):
        T = base_temp + i * temp_step
        dyn = Langevin(
            atoms,
            timestep,
            temperature_K=T,
            friction=friction,
            trajectory=traj,
            logfile=log,
            loginterval=10,
            append_trajectory=True,
        )
        dyn.run(steps_per_temp)

    # High-T hold (5 ps if dt_fs=1.0)
    dyn = Langevin(
        atoms,
        timestep,
        temperature_K=max_temp,
        friction=friction,
        trajectory=traj,
        logfile=log,
        loginterval=10,
        append_trajectory=True,
    )
    dyn.run(int(5000 / dt_fs))

    # Cooling
    for i in range(n_temp_steps):
        T = max_temp - i * temp_step
        dyn = Langevin(
            atoms,
            timestep,
            temperature_K=T,
            friction=friction,
            trajectory=traj,
            logfile=log,
            loginterval=10,
            append_trajectory=True,
        )
        dyn.run(steps_per_temp)

    # Basin hopping + relaxation after anneal
    #atoms = relax_with_basin(atoms,fmax=0.10,steps=10,round_number=round_number,outbasename="round")
    atoms = relax_with_ocp(
        atoms,
        fmax=0.03,
        steps=500,
        round_number=round_number,
        outbasename="round",
    )
    return atoms


def get_lowest_energy_structure(atoms_list):
    """Return and print the structure with the lowest potential energy."""
    if not atoms_list:
        raise ValueError("atoms_list is empty")

    energies = [atoms.get_potential_energy() for atoms in atoms_list]
    min_idx = int(np.argmin(energies))
    e_min = energies[min_idx]
    print(f"Lowest energy: {e_min:.6f} eV at index {min_idx}")
    return atoms_list[min_idx]


if __name__ == "__main__":
    # Read initial structures
    atoms_list = read(f"{file_folder}/staggered/all_relaxed_atoms.traj", index=":")

    # Sanity Check: Ensure atoms_list is iterable as a list of frames.
    # If read() returned a single Atoms object (not a list), wrap it in a list.
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    # UMA -> FAIRChemCalculator
    predictor = load_predict_unit(MODEL_ID_path, device=DEVICE, workers=1, inference_settings="turbo")
    calc = FAIRChemCalculator(predictor, task_name=TASK)

    # Now that we are sure it is a list, we can safely get the number of frames
    length = len(atoms_list)

    # Stream relaxed snapshots (re-relax each frame, then continue with annealing MD)
    with Trajectory("all_relaxed_atoms.traj", "w") as traj_relaxed:
        # Relax each frame and write it
        for i, atoms in enumerate(atoms_list):
            atoms.calc = calc
            atoms.pbc = True  # set False if this is really a non-periodic cluster
            atoms = relax_with_ocp(
                atoms,
                fmax=0.03,
                steps=500,
                round_number=i,
                outbasename="round",
                recalc=True,
            )
            traj_relaxed.write(atoms)
            atoms_list[i] = atoms  # keep the relaxed version in memory

        # Use the last relaxed structure as the starting point for annealing MD
        atoms = atoms_list[-1].copy()
        atoms.calc = calc
        atoms.pbc = True

        # Annealing rounds; index starting from length+1
        for r in range(length + 1, 41):
            atoms = anneal_md(atoms, round_number=r)
            traj_relaxed.write(atoms)
