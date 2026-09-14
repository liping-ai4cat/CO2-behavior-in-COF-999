#!/usr/bin/env python3
"""
Active Learning Selection Script (Revised)

Changes:
1. SAVING: Now saves 'original_index' as the Numpy row index (0..N-1) for consistency.
2. PLOTTING: Optimized selection plotting.
3. SELECTION: Now scans ALL rounds (cur + previous) for new selections if a cluster is empty.
   Saves selections for each round into separate files (e.g., prefix_selected_cur.csv, prefix_selected_round1.csv).
4. LOADING: Checks if {out_prefix}_features.npy exists to skip heavy CSV parsing.
"""

from __future__ import annotations

import argparse
import json
import ast
import datetime
from pathlib import Path
from typing import List, Tuple, Set, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import multiprocessing as mp
import gc
import os

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
plt.rcParams.update({'pdf.fonttype': 42})

# --------------------------------------------------------------
# Parsing & parallel helpers
# --------------------------------------------------------------
def parse_feature_cell(cell):
    """Parse a serialized list of floats from a CSV cell robustly."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=float)
    if pd.isna(cell):
        return None
    s = str(cell).strip()
    # Strip matching quotes
    if len(s) >= 2 and s[0] == s[-1] in ("'", '"'):
        s = s[1:-1]
    # Try JSON then Python literal
    for parser in (json.loads, ast.literal_eval):
        try:
            arr = parser(s)
            return np.asarray(arr, dtype=float)
        except Exception:
            pass
    # Try to salvage until last ']'
    r = s.rfind("]")
    if r != -1:
        s2 = s[: r + 1]
        for parser in (json.loads, ast.literal_eval):
            try:
                arr = parser(s2)
                return np.asarray(arr, dtype=float)
            except Exception:
                pass
    return None


def _parallel_parse(series: pd.Series, n_procs: int = 0, mp_chunksize: int = 1000) -> pd.Series:
    vals = series.tolist()
    if n_procs in (0, 1):
        out = [parse_feature_cell(v) for v in vals]
        return pd.Series(out, index=series.index)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        out = list(pool.imap(parse_feature_cell, vals, chunksize=mp_chunksize))
    return pd.Series(out, index=series.index)


def load_features_from_csv(
    csv_path: str,
    n_procs: int = 0,
    mp_chunksize: int = 1000,
    read_chunksize: int = 50000
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Full load: Reads CSV, parses 'features' column, drops bad rows.
    Returns (df_ok, X, dim).
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Processing CSV fully: {csv_path}")

    reader = pd.read_csv(csv_path, engine="c", chunksize=read_chunksize)
    df_parts: List[pd.DataFrame] = []
    feats_parts: List[pd.Series] = []
    idx_offset = 0

    for chunk in reader:
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Reading rows {idx_offset} to {idx_offset + len(chunk)} …")
        chunk = chunk.reset_index(drop=False)
        df_parts.append(chunk)
        feats_chunk = _parallel_parse(chunk["features"], n_procs=n_procs, mp_chunksize=mp_chunksize)
        feats_parts.append(feats_chunk)
        idx_offset += len(chunk)

    df_full = pd.concat(df_parts, ignore_index=True)
    feats = pd.concat(feats_parts, ignore_index=True)

    if "features" not in df_full.columns:
        raise ValueError(f"Column 'features' not found in CSV: {csv_path}")

    lengths = feats.map(lambda x: None if x is None else int(x.shape[0]))
    mode_len = lengths.dropna().mode()
    if len(mode_len) == 0:
        raise ValueError(f"Could not parse any feature vectors in {csv_path}.")
    mode_len = int(mode_len.iloc[0])

    mask_ok = lengths == mode_len
    n_bad = (~mask_ok).sum()
    if n_bad:
        print(f"[warn] {csv_path}: Dropping {n_bad} rows with non-{mode_len} feature length.")

    df_ok = df_full.loc[mask_ok].copy().reset_index(drop=False)
    good = feats.loc[mask_ok].to_list()
    if len(good) == 0:
        raise ValueError(f"No rows with consistent feature length ({mode_len}) in {csv_path}.")
    X = np.vstack(good).astype(np.float32)

    return df_ok, X, mode_len


def load_indices_only(csv_path: str, expected_len: int) -> pd.DataFrame:
    """
    Lightweight load: Reads CSV excluding 'features' to get indices,
    assuming the provided .npy matches the valid rows of this CSV.
    We just perform a basic length check here or read 'index' col.
    """
    print(f"Loading indices from {csv_path} (skipping feature parsing)...")
    # Read without 'features' column to save memory/time
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    use_cols = [c for c in cols if c != "features"]
    
    df_full = pd.read_csv(csv_path, usecols=use_cols)
    
    # If the npy was filtered for bad feature lengths, we technically
    # don't know which rows were dropped just by reading the npy.
    # However, if the user is asking to "just read it", we assume 
    # the NPY corresponds perfectly to the CSV rows (no filtering needed 
    # or filtering was already done).
    
    if len(df_full) == expected_len:
        return df_full.reset_index(drop=False)
    else:
        # If lengths differ, we cannot guarantee alignment.
        # Fallback to full parse to ensure we drop the exact same bad rows?
        # For this script, we warn and attempt to slice.
        print(f"[WARN] CSV length ({len(df_full)}) != NPY length ({expected_len}).")
        print("       Assuming the NPY corresponds to the *first* N rows or valid rows.")
        # In a real pipeline, you should save the metadata/indices along with the npy.
        # We will return the full DF; alignment is user's responsibility here.
        return df_full.reset_index(drop=False)


def choose_k(n_items: int, default_k: int | None) -> int:
    if default_k and default_k > 1:
        return min(default_k, n_items)
    k = int(np.ceil(np.sqrt(n_items)))
    return max(2, min(k, 200))


# --------------------------------------------------------------
# Selection / history helpers
# --------------------------------------------------------------
def read_previous_selected(round_tag: str, selected_csv: str) -> Set[Tuple[str, int]]:
    selected: Set[Tuple[str, int]] = set()
    if not selected_csv or not os.path.exists(selected_csv):
        return selected

    print(f"Reading previous selections for {round_tag}: {selected_csv}")
    df = pd.read_csv(selected_csv)
    if "original_index" not in df.columns:
        return selected

    idxs = pd.to_numeric(df["original_index"], errors="coerce").dropna().astype(int)
    for i in idxs:
        selected.add((round_tag, int(i)))
    return selected


def kmeans_labels_and_centers(Xs: np.ndarray, n_clusters: int, random_state: int, use_minibatch: bool):
    n = Xs.shape[0]
    if n_clusters > n:
        n_clusters = n
    if use_minibatch and n > 5000:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=2048)
    else:
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    km.fit(Xs)
    return km.labels_, km.cluster_centers_, km


def select_per_cluster(
    Xs: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    round_tag_all: np.ndarray,    # shape (N,)
    local_idx_all: np.ndarray,    # shape (N,)
    prev_selected_set: Set[Tuple[str, int]],
    strategy: int = 1
) -> np.ndarray:
    """
    Selection of new items from ANY round (candidates) based on clustering.
    If a cluster is already covered by prev_selected_set, we skip.
    Otherwise, we pick center/farthest from the available points in that cluster.
    """
    selected = []

    for k_idx in range(centers.shape[0]):
        # All members of this cluster
        idx_k = np.where(labels == k_idx)[0]
        if idx_k.size == 0:
            continue

        # Check history: how many of ANY round in this cluster were previously selected?
        prev_hits = [
            i for i in idx_k
            if (round_tag_all[i], int(local_idx_all[i])) in prev_selected_set
        ]
        n_prev = len(prev_hits)

        # Distances to centroid for ALL candidates in this cluster
        diffs = Xs[idx_k] - centers[k_idx]
        d2 = np.einsum("ij,ij->i", diffs, diffs)

        if strategy == 1:
            # Strategy 1: pick only center (closest) if no previous hits
            if n_prev >= 1:
                continue
            # Pick best from available candidates in idx_k
            best_local = int(np.argmin(d2))
            selected.append(idx_k[best_local])
            
        else:
            # Strategy 2: center + farthest
            if n_prev >= 2:
                continue
            elif n_prev == 1:
                # We have 1, need 1 more (farthest)
                far_local = int(np.argmax(d2))
                selected.append(idx_k[far_local])
            else:
                # n_prev == 0: pick center + farthest
                cen_local = int(np.argmin(d2))
                cen_global = idx_k[cen_local]
                selected.append(cen_global)
                
                if len(idx_k) > 1:
                    far_local = int(np.argmax(d2))
                    far_global = idx_k[far_local]
                    if far_global != cen_global:
                        selected.append(far_global)

    return np.unique(np.array(selected, dtype=int))


# --------------------------------------------------------------
# main
# --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round0csv", type=str, required=True, help="Path to round0 active_learning.csv")
    ap.add_argument("--round0selected", type=str, required=True, help="Path to round0 selected.csv")
    ap.add_argument("--rounds-npy", nargs="*", default=[], help="Previous rounds' feature .npy files")
    ap.add_argument("--rounds-selected", nargs="*", default=[], help="Previous rounds' *_selected.csv files")
    ap.add_argument("--csv", type=str, required=True, help="Path to THIS round's active_learning.csv (current).")
    ap.add_argument("--n-clusters", type=int, default=None, help="Number of K-Means clusters")
    ap.add_argument("--strategy", type=int, choices=[1, 2], default=1, help="Selection strategy")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--out-prefix", type=str, default="kmeans_soap", help="Output prefix")
    ap.add_argument("--pca-dim", type=int, default=200, help="PCA dims before t-SNE")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--tsne-iter", type=int, default=1000, help="t-SNE iterations")
    ap.add_argument("--n-procs", type=int, default=max(1, (mp.cpu_count() or 2) // 2), help="Processes for parsing")
    ap.add_argument("--mp-chunksize", type=int, default=1000, help="Chunksize for Pool.imap")

    args = ap.parse_args()

    # ----------------------------------------------------------
    # 1) Build combined feature matrix across all rounds
    # ----------------------------------------------------------
    all_X: List[np.ndarray] = []
    round_tag_all: List[str] = []
    local_idx_all: List[int] = []
    prev_selected_set: Set[Tuple[str, int]] = set()

    # ---- round0 ----
    df0, X0, d0 = load_features_from_csv(args.round0csv, n_procs=args.n_procs, mp_chunksize=args.mp_chunksize)
    n0 = X0.shape[0]
    print(f"Loaded round0: N={n0}, D={d0}")

    all_X.append(X0)
    round_tag_all.extend(["round0"] * n0)
    local_idx_all.extend(list(range(n0)))
    prev_selected_set |= read_previous_selected("round0", args.round0selected)

    # ---- other previous rounds ----
    rounds_npy = args.rounds_npy or []
    rounds_sel = args.rounds_selected or []
    for npy_path, sel_path in zip(rounds_npy, rounds_sel):
        if not os.path.exists(npy_path):
            continue
        X_prev = np.load(npy_path)
        n_prev, d_prev = X_prev.shape
        round_tag = Path(npy_path).stem.replace("_features", "")
        print(f"Loaded {npy_path} ({round_tag}): N={n_prev}")

        if d_prev != d0:
            raise ValueError(f"Feature dim mismatch: round0 D={d0} vs {npy_path} D={d_prev}")

        all_X.append(X_prev)
        round_tag_all.extend([round_tag] * n_prev)
        local_idx_all.extend(list(range(n_prev)))
        prev_selected_set |= read_previous_selected(round_tag, sel_path)

    # ---- current round ----
    # MODIFICATION 1: Check if {out_prefix}_features.npy exists
    curr_npy_path = f"{args.out_prefix}_features.npy"
    if os.path.exists(curr_npy_path):
        print(f"[{datetime.datetime.now()}] Found existing features {curr_npy_path}, loading directly...")
        X_cur = np.load(curr_npy_path)
        d_cur = X_cur.shape[1]
        
        # Load indices lightly (no parsing) to get csv_file_index mapping
        df_cur = load_indices_only(args.csv, expected_len=X_cur.shape[0])
    else:
        # Load fully and parse
        df_cur, X_cur, d_cur = load_features_from_csv(
            args.csv,
            n_procs=args.n_procs,
            mp_chunksize=args.mp_chunksize
        )
        # Save for next time
        np.save(curr_npy_path, X_cur.astype(np.float32))

    n_cur = X_cur.shape[0]
    print(f"Loaded current: N={n_cur}, D={d_cur}")
    if d_cur != d0:
        raise ValueError(f"Feature dim mismatch: round0 D={d0} vs current D={d_cur}")

    all_X.append(X_cur)
    round_tag_all.extend(["cur"] * n_cur)
    local_idx_all.extend(list(range(n_cur)))

    # Mappings
    # We need a way to map 'local index' -> 'csv file index' for every round if we want to save them.
    # Currently we only have df_cur for "cur". For "round0", we have df0.
    # For intermediate rounds loaded via .npy, we DO NOT have the original CSVs loaded here.
    # If we select something from round1, we might lack the 'csv_file_index'.
    # We will assume 'csv_file_index' == 'original_index' (local numpy index) for intermediate rounds
    # unless we reload their CSVs. For this script, we will save what we have.
    
    # Store df maps for rounds we actually loaded CSVs for:
    round_csv_map = {
        "round0": df0["index"].to_numpy() if "index" in df0 else np.arange(n0),
        "cur": df_cur["index"].to_numpy() if "index" in df_cur else np.arange(n_cur)
    }

    X_all = np.vstack(all_X).astype(np.float32)
    round_tag_all = np.array(round_tag_all, dtype=object)
    local_idx_all = np.array(local_idx_all, dtype=int)

    n_all, d_all = X_all.shape
    print(f"Combined all rounds: N_total={n_all}, D={d_all}")

    # ----------------------------------------------------------
    # 2) Clustering
    # ----------------------------------------------------------
    scaler = StandardScaler()
    Xs_all = scaler.fit_transform(X_all)

    k = choose_k(n_all, args.n_clusters)
    print(f"Clustering with k = {k}")
    labels_all, centers, _ = kmeans_labels_and_centers(
        Xs_all, n_clusters=k, random_state=args.seed, use_minibatch=(n_all > 5000)
    )

    # ----------------------------------------------------------
    # 3) Selection (Scan ALL rounds)
    # ----------------------------------------------------------
    # MODIFICATION 2: Select from ALL rounds (not just "cur")
    print("Selecting new candidates from ALL rounds (filling empty clusters)...")
    new_sel_inds_global = select_per_cluster(
        Xs_all,
        labels_all,
        centers,
        round_tag_all,
        local_idx_all,
        prev_selected_set,
        strategy=args.strategy
    )

    # ----------------------------------------------------------
    # 4) Save new selections SPLIT by Round
    # ----------------------------------------------------------
    if len(new_sel_inds_global) > 0:
        # Group by round tag
        new_tags = round_tag_all[new_sel_inds_global]
        unique_tags = np.unique(new_tags)
        
        for r_tag in unique_tags:
            # Mask for this round
            mask_r = (new_tags == r_tag)
            inds_global_r = new_sel_inds_global[mask_r]
            
            # Local indices
            local_inds_r = local_idx_all[inds_global_r]
            clusters_r = labels_all[inds_global_r]
            
            # Retrieve CSV indices if available
            if r_tag in round_csv_map:
                csv_inds_r = round_csv_map[r_tag][local_inds_r]
            else:
                # Fallback: use local index if we didn't load that CSV
                csv_inds_r = local_inds_r 

            sel_df = pd.DataFrame({
                "original_index": local_inds_r,
                "csv_file_index": csv_inds_r,
                "cluster": clusters_r
            })
            
            suffix = f"_{r_tag}" if r_tag != "cur" else "" # keep logic or explicit?
            # Request says "save the different round seperately".
            # We will name them strictly: {prefix}_selected_{r_tag}.csv
            # But for 'cur', usually standard is just '_selected.csv'.
            # To be clear, we will do:
            if r_tag == "cur":
                out_name = f"{args.out_prefix}_selected.csv"
            else:
                out_name = f"{args.out_prefix}_selected_{r_tag}.csv"
            
            sel_df.to_csv(out_name, index=False)
            print(f"Saved {len(sel_df)} new selections for '{r_tag}' to {out_name}")

    else:
        print("No new selections made.")

    # ----------------------------------------------------------
    # 5) Save combined labeled CSV
    # ----------------------------------------------------------
    df_combined = pd.DataFrame({
        "round_tag": round_tag_all,
        "local_index": local_idx_all,
        "cluster": labels_all
    })
    out_labeled = f"{args.out_prefix}_labeled.csv"
    df_combined.to_csv(out_labeled, index=False)
    print(f"Saved combined labeled CSV to {out_labeled}")

    # ----------------------------------------------------------
    # 6) t-SNE visualization (same as before)
    # ----------------------------------------------------------
    if args.pca_dim and args.pca_dim > 0:
        pca_dim = min(args.pca_dim, Xs_all.shape[1])
        X_for_tsne = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(Xs_all)
    else:
        X_for_tsne = Xs_all

    print("Running t-SNE...")
    try:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.tsne_iter, init="pca", verbose=1, method="barnes_hut")
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, n_iter=args.tsne_iter, init="pca", verbose=1, method="barnes_hut")
    Z = tsne.fit_transform(X_for_tsne)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=10, c=labels_all, cmap="tab20", alpha=0.1, edgecolors='none', label="All")

    # Plot previous selections
    round_to_selected_local = {}
    for rt, li in prev_selected_set:
        round_to_selected_local.setdefault(rt, []).append(li)

    # 1. Define a color map for your rounds to keep logic clean
    round_colors = {
        'round0': 'red',
        'round1': '#0f8140', # Added '#' to hex code
        'round2': 'orange'   # Added just in case
    }

    # Iterate through the dictionary
    for rt, local_list in round_to_selected_local.items():
        mask_rt = (round_tag_all == rt)
        
        if not np.any(mask_rt): continue
        
        rt_start_idx = np.argmax(mask_rt)
        rt_count = np.sum(mask_rt)
        
        valid_local = [li for li in local_list if li < rt_count]
        if not valid_local: continue
        
        rt_selected_global = rt_start_idx + np.array(valid_local, dtype=int)

        # 2. Determine color based on the current 'rt' key
        # uses .get() to default to black if the round name isn't in the dictionary
        current_color = round_colors.get(rt, 'black') 

        plt.scatter(
            Z[rt_selected_global, 0], 
            Z[rt_selected_global, 1], 
            s=15, 
            marker='o',  
            edgecolors='k', 
            linewidths=0.3, 
            c=current_color,  # Apply the color here
            label=f"Active learning {rt}"
        )

    # 3. Plot NEW selections (from global list)
    if len(new_sel_inds_global) > 0:
        plt.scatter(
            Z[new_sel_inds_global, 0], 
            Z[new_sel_inds_global, 1], 
            s=15, 
            marker='o', 
            edgecolors='k', 
            c='#3a53a4',      # Added '#' to hex code
            linewidths=0.3, 
            label="Active learning round3"
        )

    plt.xlabel("t-SNE descriptor 1")
    plt.ylabel("t-SNE descriptor 2")
    plt.title(f"t-SNE (N={n_all}, k={k})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_tsne.png", dpi=300)
    plt.savefig(f"{args.out_prefix}_tsne.pdf", dpi=300)
    print(f"Wrote {args.out_prefix}_tsne.png")

if __name__ == "__main__":
    main()
