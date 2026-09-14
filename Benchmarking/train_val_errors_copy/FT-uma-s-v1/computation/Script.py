#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read, write

# ---------------- Global Plot Settings ----------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 8,
    'pdf.fonttype': 42,
    'figure.figsize': (5, 4.5),
    'savefig.dpi': 300,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


# ---------------- User settings ----------------
project_folder = "/scratch/lipingliu/benckmarking_OMC_with_DFT/fine-tuning/fine-tune-v10-selection_SOAP_latent/round2_SOAP_only_COF-999"
TASK       = "omc"
MODEL_name = "uma-s"
MODEL_ID   = f"{MODEL_name}-1p1"
MODEL_ID_path = "/scratch/lipingliu/benckmarking_OMC_with_DFT/fine-tuning/fine-tune-v10-selection_SOAP_latent/round0_SOAP_new_calc_with_omc_data/inference_ckpt.pt"
DEVICE     = "cuda"
TRAJ_DIR   = f"{project_folder}/MMM"
RESULTS_DIR = os.getcwd()

# ---------------- Utilities ----------------
def ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def find_traj(root: str) -> List[Tuple[str, str]]:
    rootp = Path(root)
    out: List[Tuple[str, str]] = []
    for p in sorted(rootp.rglob("*.traj")):
        out.append((str(p), p.stem))
    return out


def extract_dft_energy(atoms) -> float:
    try:
        e = atoms.get_potential_energy()
        if e is not None and np.isfinite(e):
            return float(e)
    except Exception:
        pass
    for key in ("energy", "free_energy", "E", "dft_energy"):
        if key in atoms.info:
            try:
                val = float(atoms.info[key])
                if np.isfinite(val):
                    return val
            except Exception:
                continue
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        res = getattr(calc, "results", {})
        if "energy" in res and np.isfinite(res["energy"]):
            return float(res["energy"])
    raise ValueError("DFT energy not found")


def extract_dft_forces(atoms) -> Union[np.ndarray, None]:
    try:
        F = atoms.get_forces(apply_constraint=False)
        if F is not None and np.all(np.isfinite(F)):
            return np.asarray(F, float)
    except Exception:
        pass
    if "forces" in atoms.info:
        F = np.asarray(atoms.info["forces"], float)
        return F if np.all(np.isfinite(F)) else None
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        res = getattr(calc, "results", {})
        if "forces" in res:
            F = np.asarray(res["forces"], float)
            return F if np.all(np.isfinite(F)) else None
    return None


def extract_dft_stress_voigt(atoms) -> Union[np.ndarray, None]:
    try:
        vol = atoms.get_volume()
        if vol is None or not np.isfinite(vol) or vol <= 0:
            raise ValueError
    except Exception:
        vol = None

    if vol and vol > 0:
        try:
            s = atoms.get_stress(voigt=True)
            s = np.asarray(s, float).reshape(6)
            if np.all(np.isfinite(s)):
                return s
        except Exception:
            pass

    for k in ("stress", "virial"):
        if k in atoms.info:
            try:
                arr = np.asarray(atoms.info[k], float).reshape(-1)
                if arr.size == 6 and np.all(np.isfinite(arr)):
                    return arr
            except Exception:
                continue

    calc = getattr(atoms, "calc", None)
    if calc is not None:
        res = getattr(calc, "results", {})
        for k in ("stress", "virial"):
            if k in res:
                try:
                    arr = np.asarray(res[k], float).reshape(-1)
                    if arr.size == 6 and np.all(np.isfinite(arr)):
                        return arr
                except Exception:
                    continue
    return None
# -----------------------------------------------


# ---------------- Parity helpers ----------------
def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return dict(mae=np.nan, rmse=np.nan, me=np.nan, r2=np.nan, n=0)
    d = y_pred[m] - y_true[m]
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    me = float(np.mean(d))
    ss_res = float(np.sum((y_true[m] - y_pred[m])**2))
    ss_tot = float(np.sum((y_true[m] - np.mean(y_true[m]))**2))
    r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float("nan")
    return dict(mae=mae, rmse=rmse, me=me, r2=r2, n=int(np.sum(m)))


def _parity_xy(x, y, xlabel, ylabel, title, out_file):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)

    if not np.any(m):
        plt.figure(figsize=(5, 4.5), dpi=160)
        plt.title(f"{title}\nno valid data")
        plt.tight_layout()
        plt.savefig(out_file, bbox_inches="tight")
        plt.close()
        return dict(mae=np.nan, rmse=np.nan, r2=np.nan, me=np.nan, n=0)

    x = x[m]; y = y[m]
    stats = _metrics(x, y)

    xmin, xmax = float(np.min(np.r_[x, y])), float(np.max(np.r_[x, y]))
    pad = 0.02 * (xmax - xmin if xmax > xmin else 1.0)
    lo, hi = xmin - pad, xmax + pad

    plt.figure(figsize=(5, 4.5), dpi=160)
    plt.scatter(x, y, s=18)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nMAE={stats['mae']:.4g}, RMSE={stats['rmse']:.4g}, "
              f"ME={stats['me']:.4g}, R²={stats['r2']:.3f}, N={stats['n']}")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.tight_layout(); plt.savefig(out_file, bbox_inches="tight"); plt.close()

    return stats


def parity_and_metrics(df: pd.DataFrame, title: str, fig_out: str) -> dict:
    valid = df.dropna(subset=["dft_energy_eV"])
    if len(valid) == 0:
        plt.figure(figsize=(5, 4.5), dpi=160)
        plt.title(f"{title}\nno valid DFT energies")
        plt.tight_layout()
        plt.savefig(fig_out, bbox_inches="tight")
        plt.close()
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "me": np.nan, "n": 0}

    y_true = valid["dft_energy_eV"].values
    y_pred = valid["uma_energy_eV"].values
    diff = (y_pred - y_true)
    mae = float(np.mean(np.abs(diff))) * 1000.0
    rmse = float(np.sqrt(np.mean(diff**2))) * 1000.0
    me = float(np.mean(diff)) * 1000.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    n = int(len(valid))

    xmin = float(np.nanmin(np.r_[y_true, y_pred]))
    xmax = float(np.nanmax(np.r_[y_true, y_pred]))
    pad = 0.02 * (xmax - xmin if xmax > xmin else 1.0)
    lo, hi = xmin - pad, xmax + pad

    plt.figure(figsize=(5, 4.5), dpi=160)
    plt.scatter(y_true, y_pred, s=20)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("DFT energy (eV/atom)")
    plt.ylabel("UMA energy (eV/atom)")
    plt.title(f"{title}\nMAE={mae:.3f} meV, RMSE={rmse:.2f} meV, ME={me:.3f} meV, R²={r2:.2f}, N={n}")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.tight_layout(); plt.savefig(fig_out, bbox_inches="tight"); plt.close()

    return {"mae": mae, "rmse": rmse, "r2": r2, "me": me, "n": n}


def parity_forces(df: pd.DataFrame, prefix: str) -> dict:
    out = {}
    out["force_total"] = _parity_xy(
        df["total_force_dft_eVA"],
        df["total_force_uma_eVA"],
        xlabel="DFT total |F| (eV/Å)",
        ylabel="UMA total |F| (eV/Å)",
        title="Total force norm parity",
        out_file=f"{prefix}_forces_total.pdf",
    )
    out["force_rms"] = _parity_xy(
        df["rms_force_dft_eVA"],
        df["rms_force_uma_eVA"],
        xlabel="DFT RMS |F| per atom (eV/Å)",
        ylabel="UMA RMS |F| per atom (eV/Å)",
        title="Per-atom RMS force parity",
        out_file=f"{prefix}_forces_rms.pdf",
    )
    return out


def parity_force_components(flat_dft, flat_uma, title, out_file):
    x = np.asarray(flat_dft, float)
    y = np.asarray(flat_uma, float)
    m = np.isfinite(x) & np.isfinite(y)
    
    if not np.any(m):
        plt.figure(figsize=(5, 4.5), dpi=160)
        plt.title(f"{title}\nno valid data")
        plt.tight_layout(); plt.savefig(out_file, bbox_inches="tight"); plt.close()
        return dict(mae=np.nan, rmse=np.nan, me=np.nan, r2=np.nan, n=0)

    stats = _metrics(x, y)
    lo = float(np.min(np.r_[x[m], y[m]]))
    hi = float(np.max(np.r_[x[m], y[m]]))
    pad = 0.02 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad

    plt.figure(figsize=(5, 4.5), dpi=160)
    plt.scatter(x[m], y[m], s=6)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("DFT force components (eV/Å)")
    plt.ylabel("UMA force components (eV/Å)")
    plt.title(f"{title}\nMAE={stats['mae']:.4g}  RMSE={stats['rmse']:.4g}  ME={stats['me']:.4g}  R²={stats['r2']:.3f}  N={stats['n']}")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.tight_layout(); plt.savefig(out_file, bbox_inches="tight"); plt.close()

    return stats


def parity_stress_components(flat_dft, flat_uma, title, out_file, unit="eV/Å³"):
    x = np.asarray(flat_dft, float)
    y = np.asarray(flat_uma, float)
    m = np.isfinite(x) & np.isfinite(y)
    
    if not np.any(m):
        plt.figure(figsize=(5, 4.5), dpi=160)
        plt.title(f"{title}\nno valid data")
        plt.tight_layout(); plt.savefig(out_file, bbox_inches="tight"); plt.close()
        return dict(mae=np.nan, rmse=np.nan, me=np.nan, r2=np.nan, n=0)

    stats = _metrics(x, y)
    lo = float(np.min(np.r_[x[m], y[m]]))
    hi = float(np.max(np.r_[x[m], y[m]]))
    pad = 0.02 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad

    plt.figure(figsize=(5, 4.5), dpi=160)
    plt.scatter(x[m], y[m], s=6)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel(f"DFT stress components ({unit})")
    plt.ylabel(f"UMA stress components ({unit})")
    plt.title(f"{title}\nMAE={stats['mae']:.4g}  RMSE={stats['rmse']:.4g}  ME={stats['me']:.4g}  R²={stats['r2']:.3f}  N={stats['n']}")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.tight_layout(); plt.savefig(out_file, bbox_inches="tight"); plt.close()

    return stats
# -----------------------------------------------


# ---------------- Core compute ----------------
def compute_props(images, calc, out_traj) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, uma_frames = [], []
    flat_dft_forces, flat_uma_forces = [], []
    flat_dft_stress, flat_uma_stress = [], []

    EV_PER_A3_TO_GPA = 160.21766208

    for i, atoms in enumerate(images):
        nat = len(atoms)
        try:
            e_dft = extract_dft_energy(atoms) / nat
        except ValueError:
            e_dft = np.nan
        F_dft = extract_dft_forces(atoms)
        S_dft = extract_dft_stress_voigt(atoms)

        a = atoms.copy()
        a.calc = calc
        try:
            e_uma = float(a.get_potential_energy()) / nat
        except Exception:
            e_uma = np.nan
        try:
            F_uma = np.asarray(a.get_forces(apply_constraint=False), float)
        except Exception:
            F_uma = None
        try:
            vol = a.get_volume()
        except Exception:
            vol = None
        if vol is not None and np.isfinite(vol) and vol > 0:
            try:
                S_uma = np.asarray(a.get_stress(voigt=True), float).reshape(6)
                if not np.all(np.isfinite(S_uma)):
                    S_uma = None
            except Exception:
                S_uma = None
        else:
            S_uma = None

        a.info["E_uma_eV_per_atom"] = e_uma
        if F_uma is not None:
            a.set_array("F_uma_eVA", F_uma)
        if S_uma is not None:
            a.info["stress_uma_eV_perA3"] = S_uma.tolist()
        uma_frames.append(a)

        if F_dft is not None and F_uma is not None and F_dft.shape == F_uma.shape:
            flat_dft_forces.append(F_dft.ravel())
            flat_uma_forces.append(F_uma.ravel())
            force_mae = float(np.mean(np.abs(F_uma - F_dft)))
            totalF_dft = float(np.linalg.norm(F_dft.ravel()))
            totalF_uma = float(np.linalg.norm(F_uma.ravel()))
            rmsF_dft = float(np.sqrt(np.mean(F_dft**2)))
            rmsF_uma = float(np.sqrt(np.mean(F_uma**2)))
        else:
            force_mae = np.nan
            totalF_dft = float(np.linalg.norm(F_dft.ravel())) if F_dft is not None else np.nan
            totalF_uma = float(np.linalg.norm(F_uma.ravel())) if F_uma is not None else np.nan
            rmsF_dft = float(np.sqrt(np.mean(F_dft**2))) if F_dft is not None else np.nan
            rmsF_uma = float(np.sqrt(np.mean(F_uma**2))) if F_uma is not None else np.nan

        def stress_stats(S):
            if S is None:
                return np.nan, np.nan, np.nan
            S = np.asarray(S, float).reshape(6)
            fro = float(np.linalg.norm(S))
            trace = float(S[0] + S[1] + S[2])
            p_gpa = float(-trace / 3.0 * EV_PER_A3_TO_GPA)
            return fro, trace, p_gpa

        fro_dft, tr_dft, p_dft_gpa = stress_stats(S_dft)
        fro_uma, tr_uma, p_uma_gpa = stress_stats(S_uma)

        if S_dft is not None and S_uma is not None:
            flat_dft_stress.append(np.asarray(S_dft, float).ravel())
            flat_uma_stress.append(np.asarray(S_uma, float).ravel())

        rows.append({
            "frame": i,
            "dft_energy_eV": e_dft,
            "uma_energy_eV": e_uma,
            "delta_e_eV": (e_uma - e_dft) if np.isfinite(e_dft) else np.nan,
            "force_mae_eVA": force_mae,
            "total_force_dft_eVA": totalF_dft,
            "total_force_uma_eVA": totalF_uma,
            "rms_force_dft_eVA": rmsF_dft,
            "rms_force_uma_eVA": rmsF_uma,
            "stress_fro_dft_eV_A3": fro_dft,
            "stress_fro_uma_eV_A3": fro_uma,
            "stress_trace_dft_eV_A3": tr_dft,
            "stress_trace_uma_eV_A3": tr_uma,
            "pressure_dft_GPa": p_dft_gpa,
            "pressure_uma_GPa": p_uma_gpa,
        })

    if out_traj is not None:
        write(out_traj, uma_frames)
        print(f"Saved UMA-predicted structures: {out_traj}")

    df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    flat_dft_forces = np.concatenate(flat_dft_forces) if flat_dft_forces else np.array([], float)
    flat_uma_forces = np.concatenate(flat_uma_forces) if flat_uma_forces else np.array([], float)
    flat_dft_stress = np.concatenate(flat_dft_stress) if flat_dft_stress else np.array([], float)
    flat_uma_stress = np.concatenate(flat_uma_stress) if flat_uma_stress else np.array([], float)
    return df, flat_dft_forces, flat_uma_forces, flat_dft_stress, flat_uma_stress
# -----------------------------------------------


# ---------------- Main ----------------
def main():
    ensure_dir(RESULTS_DIR)

    predictor = load_predict_unit(MODEL_ID_path, device=DEVICE)
    calc = FAIRChemCalculator(predictor, task_name=TASK)

    trajs = find_traj(TRAJ_DIR)
    if not trajs:
        raise FileNotFoundError(f"No .traj files under {TRAJ_DIR}")

    all_rows: List[pd.DataFrame] = []
    metrics_rows: List[Dict[str, Union[str, float, int]]] = []
    
    # Lists for accumulating raw component errors for post-processing
    force_err_frames: List[pd.DataFrame] = []
    stress_err_frames: List[pd.DataFrame] = []

    # Lists for aggregating flat arrays for parity plots
    all_flat_dftF, all_flat_umaF = [], []
    all_flat_dftS, all_flat_umaS = [], []

    for traj_path, stem in trajs:
        out_dir = Path(RESULTS_DIR) / stem
        ensure_dir(out_dir)

        try:
            images = read(traj_path, index=":")
        except Exception as e:
            print(f"[WARN] Could not read {traj_path}: {e}")
            continue

        uma_out = out_dir / "uma.traj"
        df, f_dft_flat, f_uma_flat, s_dft_flat, s_uma_flat = compute_props(images, calc, uma_out)
        df.insert(0, "system", stem)

        # 1. Save component errors for this system
        # Calculate absolute errors
        if len(f_dft_flat) > 0 and len(f_uma_flat) > 0:
            abs_err_f = np.abs(f_uma_flat - f_dft_flat)
            force_err_frames.append(pd.DataFrame({
                "system": stem,
                "abs_err_eVA": abs_err_f
            }))

        if len(s_dft_flat) > 0 and len(s_uma_flat) > 0:
            abs_err_s = np.abs(s_uma_flat - s_dft_flat)
            stress_err_frames.append(pd.DataFrame({
                "system": stem,
                "abs_err_eVA3": abs_err_s
            }))

        # 2. Per-system tables and plots
        csv_out = out_dir / f"{stem}_energetics.csv"
        df.to_csv(csv_out, index=False)
        print(f"Saved: {csv_out}")

        fig_out = out_dir / f"{stem}_parity.pdf"
        mE = parity_and_metrics(df, f"{stem} | UMA ({MODEL_ID}, {TASK}) vs DFT", str(fig_out))
        print(f"Saved: {fig_out}  |  E-MAE={mE['mae']:.3f} meV  RMSE={mE['rmse']:.3f} meV  R2={mE['r2']:.3f}  N={mE['n']}")

        fplots_metrics = parity_forces(df, prefix=str(out_dir / stem))
        print(f"Saved: {out_dir / (stem + '_forces_total.pdf')}")
        print(f"Saved: {out_dir / (stem + '_forces_rms.pdf')}")

        fcomp_pdf = out_dir / f"{stem}_forces_components.pdf"
        mFcomp = parity_force_components(
            f_dft_flat, f_uma_flat,
            title="Force components parity",
            out_file=str(fcomp_pdf),
        )
        print(f"Saved: {fcomp_pdf}")

        scomp_pdf = out_dir / f"{stem}_stress_components.pdf"
        mScomp = parity_stress_components(
            s_dft_flat, s_uma_flat,
            title="Stress components parity",
            out_file=str(scomp_pdf),
            unit="eV/Å³",
        )
        print(f"Saved: {scomp_pdf}")

        all_rows.append(df)
        all_flat_dftF.append(f_dft_flat)
        all_flat_umaF.append(f_uma_flat)
        all_flat_dftS.append(s_dft_flat)
        all_flat_umaS.append(s_uma_flat)

        metrics_rows.append({
            "system": stem,
            "energy_mae_meV_per_atom": mE["mae"],
            "energy_rmse_meV_per_atom": mE["rmse"],
            "energy_me_meV_per_atom": mE["me"],
            "energy_r2": mE["r2"],
            "n_energy": mE["n"],
            "force_total_mae_eVA": fplots_metrics["force_total"]["mae"],
            "force_total_rmse_eVA": fplots_metrics["force_total"]["rmse"],
            "force_total_me_eVA": fplots_metrics["force_total"]["me"],
            "force_total_r2": fplots_metrics["force_total"]["r2"],
            "n_force_total": fplots_metrics["force_total"]["n"],
            "force_rms_mae_eVA": fplots_metrics["force_rms"]["mae"],
            "force_rms_rmse_eVA": fplots_metrics["force_rms"]["rmse"],
            "force_rms_me_eVA": fplots_metrics["force_rms"]["me"],
            "force_rms_r2": fplots_metrics["force_rms"]["r2"],
            "n_force_rms": fplots_metrics["force_rms"]["n"],
            "force_comp_mae_eVA": mFcomp["mae"],
            "force_comp_rmse_eVA": mFcomp["rmse"],
            "force_comp_me_eVA": mFcomp["me"],
            "force_comp_r2": mFcomp["r2"],
            "n_force_components": mFcomp["n"],
            "stress_comp_mae_eV_A3": mScomp["mae"],
            "stress_comp_rmse_eV_A3": mScomp["rmse"],
            "stress_comp_me_eV_A3": mScomp["me"],
            "stress_comp_r2": mScomp["r2"],
            "n_stress_components": mScomp["n"],
        })

    # 3. Combine and save everything
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_csv = Path(RESULTS_DIR) / "all_energetics.csv"
        all_df.to_csv(all_csv, index=False)
        print(f"Saved: {all_csv}")

        # Save component errors for post-processing (Violin plots)
        if force_err_frames:
            all_force_err_df = pd.concat(force_err_frames, ignore_index=True)
            force_err_csv = Path(RESULTS_DIR) / "all_force_component_errors.csv"
            all_force_err_df.to_csv(force_err_csv, index=False)
            print(f"Saved: {force_err_csv} (Rows: {len(all_force_err_df)})")
        
        if stress_err_frames:
            all_stress_err_df = pd.concat(stress_err_frames, ignore_index=True)
            stress_err_csv = Path(RESULTS_DIR) / "all_stress_component_errors.csv"
            all_stress_err_df.to_csv(stress_err_csv, index=False)
            print(f"Saved: {stress_err_csv} (Rows: {len(all_stress_err_df)})")

        fig_all = Path(RESULTS_DIR) / "all_parity.pdf"
        _ = parity_and_metrics(all_df, f"ALL | UMA ({MODEL_ID}, {TASK}) vs DFT", str(fig_all))
        print(f"Saved: {fig_all}")

        all_flat_dftF = np.concatenate(all_flat_dftF) if any(len(x) for x in all_flat_dftF) else np.array([], float)
        all_flat_umaF = np.concatenate(all_flat_umaF) if any(len(x) for x in all_flat_umaF) else np.array([], float)
        all_compF_pdf = Path(RESULTS_DIR) / "all_forces_components.pdf"
        _ = parity_force_components(
            all_flat_dftF, all_flat_umaF,
            title=f"ALL force components | UMA ({MODEL_ID}, {TASK}) vs DFT",
            out_file=str(all_compF_pdf),
        )
        print(f"Saved: {all_compF_pdf}")

        all_flat_dftS = np.concatenate(all_flat_dftS) if any(len(x) for x in all_flat_dftS) else np.array([], float)
        all_flat_umaS = np.concatenate(all_flat_umaS) if any(len(x) for x in all_flat_umaS) else np.array([], float)
        all_compS_pdf = Path(RESULTS_DIR) / "all_stress_components.pdf"
        _ = parity_stress_components(
            all_flat_dftS, all_flat_umaS,
            title=f"ALL stress components | UMA ({MODEL_ID}, {TASK}) vs DFT",
            out_file=str(all_compS_pdf),
            unit="eV/Å³",
        )
        print(f"Saved: {all_compS_pdf}")

        if metrics_rows:
            met_df = pd.DataFrame(metrics_rows)
            met_csv = Path(RESULTS_DIR) / "per_system_metrics.csv"
            met_df.to_csv(met_csv, index=False)
            print(f"Saved: {met_csv}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
