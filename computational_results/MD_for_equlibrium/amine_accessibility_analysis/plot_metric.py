#!/usr/bin/env python3
"""
Plot a single amine-accessibility metric as its own bar chart, from
accessibility_summary.json. Works both as a CLI script and imported in Jupyter.

Jupyter:
    %matplotlib inline
    from plot_metric import load_summary, plot_metric
    S = load_summary("accessibility_summary.json")
    plot_metric("sasa", S)          # displays inline (don't assign to see it)
    plot_metric("facc", S, save=True)   # also write png/pdf
    fig, ax = plot_metric("phi", S) # capture to customize further

CLI (keyword flags; ignores unrelated args):
    python plot_metric.py                        # f_acc and <SASA>, saved to disk
    python plot_metric.py --metric facc sasa phi # any combination, saved
    python plot_metric.py -m sasa -j other.json  # choose metric and JSON
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt          # NOTE: no matplotlib.use("Agg") here

#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from ase.io import read, write
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 1. Update settings with a robust fallback list
plt.rcParams.update({
    'font.size': 14,
    'pdf.fonttype': 42,  # Ensures text remains editable/vector in PDF
    'ps.fonttype': 42,
    'figure.figsize': (5, 4.5),
    'savefig.dpi': 300,
})


# 1. Define the Global Family
plt.rcParams['font.family'] = 'sans-serif'

# 2. Define the Fallback Priority
# Matplotlib will try these in order. If 'Helvetica' fails, it hits 'TeX Gyre Heros'.
# If that fails, it hits 'Liberation Sans', and so on.
plt.rcParams['font.sans-serif'] = [
    'Helvetica',
    'TeX Gyre Heros',
    'Liberation Sans',
    'Arial',
    'DejaVu Sans'
]

# 3. Ensure the backend handles the fallback correctly for PDFs
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 2. FORCE a refresh of the font cache (do this once if the error persists)
# This clears out any "memory" of fonts not being found.
import shutil
import os
try:
    shutil.rmtree(matplotlib.get_cachedir())
except Exception:
    pass

DEFAULT_JSON = "/scratch/lipingliu/benckmarking_OMC_with_DFT/mutilayers_all/fine-tune-v10-round2-SOAP/COF-999/balance_simulation_100ps/114+CO2/amine_accessibility_analysis/accessibility_summary.json"
COLORS = {"COF-998": "#ff1e1e", "COF-999": "#398be5", "COF-1000": "#e59e00"}

# metric key -> (value field, SE field, y-axis label, filename stem)
METRICS = {
    "facc": ("f_acc_inst", "f_acc_SE", r"$f_{\mathrm{acc}}$  (SASA > 5 $\AA^2$)", "plot_f_acc"),
    "sasa": ("mean_group_SASA_per_amine_A2", "group_SASA_SE",
             r"$\langle\mathrm{SASA}\rangle$ ($\AA^2$)", "plot_SASA"),
    "phi":  ("void_fraction_phi", "phi_SE", r"void fraction $\phi$ (CO$_2$ probe)", "plot_phi"),
}


def load_summary(json_path=DEFAULT_JSON):
    """Load the summary JSON (list of per-material dicts)."""
    with open(json_path) as f:
        return json.load(f)


def plot_metric(key, summaries=None, json_path=DEFAULT_JSON,
                ax=None, save=False, outdir=None, bar_width=0.35):
    """Bar chart of one metric. Returns (fig, ax); displays inline in Jupyter.

    key      : 'facc' | 'sasa' | 'phi'
    summaries: parsed JSON list; if None, loaded from json_path
    ax       : plot into an existing Axes (for building subplots); else new fig
    save     : if True, write <stem>.png (300 dpi) and <stem>.pdf to outdir
    outdir   : where to save (default: dir of json_path, or cwd)
    bar_width: bar width (smaller = narrower bars)
    """
    if key not in METRICS:
        raise ValueError(f"unknown metric '{key}'; choose from {list(METRICS)}")
    if summaries is None:
        summaries = load_summary(json_path)

    vfield, sefield, ylabel, stem = METRICS[key]
    names = [s["material"] for s in summaries]
    vals = [s[vfield] for s in summaries]
    ses = [s.get(sefield, 0.0) for s in summaries]
    cols = [COLORS.get(n, "#888888") for n in names]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
    else:
        fig = ax.figure

    x = np.arange(len(names))
    ax.bar(x, vals, yerr=ses, capsize=5, color=cols,
           edgecolor="black", linewidth=0.8, width=bar_width)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)

    # (1) full frame: show all four spines (top + right included)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    # (2) no horizontal grid lines
    ax.grid(False)

    fig.tight_layout()

    if save:
        outdir = outdir or (os.path.dirname(os.path.abspath(json_path)) or ".")
        fig.savefig(os.path.join(outdir, stem + ".png"), dpi=300)
        fig.savefig(os.path.join(outdir, stem + ".pdf"))
        print(f"wrote {stem}.png (+ .pdf) to {outdir}")
    return fig, ax


def _in_notebook():
    """True when running inside a Jupyter/IPython kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def _cli():
    import argparse
    plt.switch_backend("Agg")   # headless: only when run as a real script
    p = argparse.ArgumentParser(description="Plot amine-accessibility metrics.")
    p.add_argument("-m", "--metric", nargs="+", choices=list(METRICS),
                   default=["facc", "sasa"], help="one or more metrics to plot")
    p.add_argument("-j", "--json", default=DEFAULT_JSON, help="summary JSON path")
    p.add_argument("-w", "--width", type=float, default=0.35, help="bar width")
    # parse_known_args ignores stray args (e.g. a Jupyter kernel's -f kernel.json)
    a, _ = p.parse_known_args()
    S = load_summary(a.json)
    for k in a.metric:
        plot_metric(k, S, json_path=a.json, save=True, bar_width=a.width)


# Only run the CLI when executed as a real script — NOT when pasted into a
# notebook cell (where __name__ is also "__main__").
if __name__ == "__main__" and not _in_notebook():
    _cli()
