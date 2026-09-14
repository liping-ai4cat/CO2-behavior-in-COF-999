#!/usr/bin/env python
# Barrier from a 2D FES grid by connectivity (minimax) on the discretized surface
# Input: FES.dat with columns: s1  s2  F
# You provide basin windows (ranges) for reactant A and product B in CV space.

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspe
from heapq import heappush, heappop
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
from scipy import stats
import os, sys, pickle
import math

plt.rcParams.update({'font.size': 12})
rcParams['ps.useafm'] = True
rcParams['pdf.fonttype'] = 42


import subprocess

try:
    # check_call runs the command and errors out if it fails
    subprocess.check_call('plumed sum_hills --hills HILLS --kt 0.025852 --outfile FES.dat', shell=True)
    print("Success: FES.dat created.")

except subprocess.CalledProcessError:
    print("Error: The PLUMED command failed. Check your HILLS file.")

FES_FILE = "FES.dat"

# --- load and grid ---
s1, s2, F = np.loadtxt(FES_FILE, comments="#", usecols=(0, 1, 2), unpack=True)
u1 = np.unique(s1); u2 = np.unique(s2)
i1 = {v: k for k, v in enumerate(u1)}; i2 = {v: k for k, v in enumerate(u2)}
G = np.full((len(u1), len(u2)), np.nan)
for x, y, z in zip(s1, s2, F):
    G[i1[x], i2[y]] = z
assert not np.isnan(G).any(), "Grid reconstruction failed; check FES file."

# indices in the basin windows
def sel(win):
    j1 = np.where((u1 >= win[0]) & (u1 <= win[1]))[0]
    j2 = np.where((u2 >= win[2]) & (u2 <= win[3]))[0]
    return np.array([(a, b) for a in j1 for b in j2], dtype=int)
s1min,s1max=np.min(s1),np.max(s1)
s2min,s2max=np.min(s2),np.max(s2)
# edit these windows for your two basins:

# edit these windows for your two basins:
if s2max >= 2:
    BASIN_A_s2min = 1.3
else:
    BASIN_A_s2min = 0.4

BASIN_A = (2.5, s1max, BASIN_A_s2min, s2max)     # (s1min, s1max, s2min, s2max)
BASIN_B = (1, 2, -4, s2max)

IA = sel(BASIN_A); IB = sel(BASIN_B)
assert IA.size and IB.size, "Empty basin window."

# locate minima inside each basin
a_idx = IA[np.argmin(G[IA[:, 0], IA[:, 1]])]
b_idx = IB[np.argmin(G[IB[:, 0], IB[:, 1]])]
FA = float(G[tuple(a_idx)]); FB = float(G[tuple(b_idx)])

# 8-neighbor stencil
nbr = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# connectivity at a threshold (for bisection on saddle energy)
def connected(thr):
    if FA > thr or FB > thr:
        return False
    seen = np.zeros_like(G, dtype=bool)
    qq = deque([tuple(a_idx)]); seen[tuple(a_idx)] = True
    while qq:
        i, j = qq.popleft()
        if (i, j) == tuple(b_idx):
            return True
        for di, dj in nbr:
            ii, jj = i + di, j + dj
            if 0 <= ii < G.shape[0] and 0 <= jj < G.shape[1]:
                if (not seen[ii, jj]) and (G[ii, jj] <= thr):
                    seen[ii, jj] = True
                    qq.append((ii, jj))
    return False

# minimal threshold that connects A and B => saddle energy
levels = np.unique(G.ravel())
lo, hi = 0, len(levels) - 1
ans = levels[-1]
while lo <= hi:
    mid = (lo + hi) // 2
    if connected(levels[mid]):
        ans = levels[mid]; hi = mid - 1
    else:
        lo = mid + 1

# minimax path to locate the TS cell explicitly
def minimax_path(G, start, goal):
    n, m = G.shape
    cost = np.full((n, m), np.inf)
    prev = np.full((n, m, 2), -1, dtype=int)
    si, sj = start; gi, gj = goal
    cost[si, sj] = G[si, sj]
    pq = []; heappush(pq, (cost[si, sj], si, sj))
    while pq:
        c, i, j = heappop(pq)
        if (i, j) == (gi, gj):
            break
        if c > cost[i, j]:
            continue
        for di, dj in nbr:
            ii, jj = i + di, j + dj
            if 0 <= ii < n and 0 <= jj < m:
                nc = max(c, G[ii, jj])
                if nc < cost[ii, jj]:
                    cost[ii, jj] = nc
                    prev[ii, jj] = (i, j)
                    heappush(pq, (nc, ii, jj))
    # reconstruct
    path = []
    if np.isfinite(cost[gi, gj]):
        i, j = gi, gj
        while (i, j) != (si, sj):
            path.append((i, j))
            pi, pj = prev[i, j]
            if pi < 0: break
            i, j = pi, pj
        path.append((si, sj))
        path = path[::-1]
    return path, float(cost[gi, gj])

path, saddle_energy = minimax_path(G, tuple(a_idx), tuple(b_idx))
ts_idx = max(path, key=lambda ij: G[ij]) if path else tuple(a_idx)
FTS = float(G[ts_idx])

DeltaG_fwd = float(FTS - FA)
DeltaG_rev = float(FTS - FB)

# report both eV and kJ/mol
to_kJmol = 96.485
sA = (float(u1[a_idx[0]]), float(u2[a_idx[1]]))
sB = (float(u1[b_idx[0]]), float(u2[b_idx[1]]))
sTS = (float(u1[ts_idx[0]]), float(u2[ts_idx[1]]))
out = {
    "sA": sA, "FA": FA, "sB": sB, "FB": FB, "sTS": sTS, "FTS": FTS,
    "DeltaG_fwd": float(FB-FA),
    "Ga_fwd": DeltaG_fwd, "Ga_rev": DeltaG_rev,
    "FA_kJmol": float(FA * to_kJmol), "FB_kJmol": float(FB * to_kJmol),
    "FTS_kJmol": float(FTS * to_kJmol),
    "DeltaG_kJmol": float((FB-FA) * to_kJmol),
    "Ga_fwd_kJmol": float(DeltaG_fwd * to_kJmol),
    "Ga_rev_kJmol": float(DeltaG_rev * to_kJmol),
}

def round3(x):
    if isinstance(x, (tuple, list)):
        return tuple(round(v, 3) for v in x)
    if isinstance(x, float):
        return round(x, 3)
    return x

out = {k: round3(v) for k, v in out.items()}
print(f'\n--- Running job 1937 ---\n')
print(out)

# --- Plot FES with annotations ---
U1, U2 = np.meshgrid(u1, u2, indexing="ij")
FkJ = G * to_kJmol
plt.figure(figsize=(6, 5))
cf = plt.contourf(U1, U2, FkJ, levels=30, cmap="viridis")
plt.colorbar(cf, label="Free energy (kJ/mol)")
# saddle contour
plt.contour(U1, U2, FkJ, levels=[FTS * to_kJmol], linewidths=1.2)

# markers
plt.plot(*sA, marker="o", ms=6)
plt.plot(*sB, marker="s", ms=6)
plt.plot(*sTS, marker="*", ms=10)

# annotations
plt.annotate(
    f"CO2(g)\nF={out['FA_kJmol']:.3f} kJ/mol",
    xy=sA, xytext=(10, -10), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", lw=1)
)
plt.annotate(
    f"*CO2\nF={out['FB_kJmol']:.3f} kJ/mol",
    xy=sB, xytext=(10, -10), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", lw=1)
)
plt.annotate(
    f"TS \nF={out['FTS_kJmol']:.3f}kJ/mol)",
    xy=sTS, xytext=(10, 10), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", lw=1)
)

plt.xlabel("d(N–C) (Å)", fontsize=12)
plt.ylabel("CN(N–H) – 2CN(O–H)", fontsize=12)
plt.title(f"Fa={out['Ga_fwd_kJmol']:.2f} kJ/mol, ΔF={out['DeltaG_kJmol']:.2f} kJ/mol")
plt.tight_layout()
plt.savefig("metadynamics.png", dpi=300, bbox_inches="tight")
#plt.savefig("metadynamics.pdf", dpi=300, bbox_inches="tight")
plt.close()

