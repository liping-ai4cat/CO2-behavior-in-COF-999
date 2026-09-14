#!/usr/bin/env python3
"""
Summarize amine-accessibility results from accessibility_summary.json.

Produces:
  - a formatted table printed to stdout
  - accessibility_results_table.csv   (tidy, one row per material)
  - accessibility_results_table.md    (Markdown table for a manuscript/notes)

Usage:  python summarize.py [path_to_summary_json]
"""
import sys, json, csv, os

def fmt(v, e=None, nd=3):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if e is None else f"{v:.{nd}f} ± {e:.{nd}f}"

def main():
    jpath = sys.argv[1] if len(sys.argv) > 1 else "accessibility_summary.json"
    with open(jpath) as f:
        S = json.load(f)
    outdir = os.path.dirname(os.path.abspath(jpath))

    # ---- ordered columns (material -> metrics) ----
    rows = []
    for s in S:
        sub = s["per_subtype"]
        rows.append({
            "material": s["material"],
            "n_amine": s["n_amine"],
            "f_acc": s["f_acc_inst"], "f_acc_SE": s["f_acc_SE"],
            "group_SASA_A2": s["mean_group_SASA_per_amine_A2"], "group_SASA_SE": s["group_SASA_SE"],
            "N_SASA_A2": s["mean_N_SASA_per_amine_A2"],
            "phi": s["void_fraction_phi"], "phi_SE": s["phi_SE"],
            "crowding_6A": s["mean_crowding_6A"],
            "primary_SASA": sub["primary"]["mean_group_SASA"],
            "secondary_SASA": sub["secondary"]["mean_group_SASA"],
            "tertiary_SASA": sub["tertiary"]["mean_group_SASA"],
            "n_frames": s["n_frames"],
            "drift_group_SASA": abs(s["stability_halfwindow"]["group_SASA_secondhalf"]
                                    - s["stability_halfwindow"]["group_SASA_firsthalf"]),
        })

    # ---- pretty print ----
    hdr = f"{'material':<10}{'n_am':>6}{'f_acc':>16}{'<grp-SASA>/A^2':>20}{'<N-SASA>':>10}{'phi':>16}{'crowd':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['material']:<10}{r['n_amine']:>6}"
              f"{fmt(r['f_acc'], r['f_acc_SE']):>16}"
              f"{fmt(r['group_SASA_A2'], r['group_SASA_SE'], 2):>20}"
              f"{fmt(r['N_SASA_A2'], nd=2):>10}"
              f"{fmt(r['phi'], r['phi_SE']):>16}"
              f"{fmt(r['crowding_6A'], nd=1):>8}")
    print("\nPer-subtype <group-SASA> (A^2):  primary / secondary / tertiary")
    for r in rows:
        print(f"  {r['material']:<10} {r['primary_SASA']:6.2f} / {r['secondary_SASA']:6.2f} / {r['tertiary_SASA']:6.2f}")

    # ordering check on the two headline metrics
    def ordered(key):
        vals = [r[key] for r in rows]
        return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    print("\nMonotonic increase (as listed)?  "
          f"<group-SASA>: {ordered('group_SASA_A2')}   phi: {ordered('phi')}   f_acc: {ordered('f_acc')}")

    # ---- CSV ----
    cols = ["material", "n_amine", "f_acc", "f_acc_SE", "group_SASA_A2", "group_SASA_SE",
            "N_SASA_A2", "phi", "phi_SE", "crowding_6A",
            "primary_SASA", "secondary_SASA", "tertiary_SASA", "drift_group_SASA", "n_frames"]
    with open(os.path.join(outdir, "accessibility_results_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    # ---- Markdown ----
    with open(os.path.join(outdir, "accessibility_results_table.md"), "w") as f:
        f.write("| Material | N amines | f_acc (SASA>5 Å²) | ⟨group-SASA⟩ (Å²) | ⟨N-SASA⟩ (Å²) | void fraction φ |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['material']} | {r['n_amine']} | {fmt(r['f_acc'], r['f_acc_SE'])} "
                    f"| {fmt(r['group_SASA_A2'], r['group_SASA_SE'], 2)} | {fmt(r['N_SASA_A2'], nd=2)} "
                    f"| {fmt(r['phi'], r['phi_SE'])} |\n")

    print(f"\nWrote accessibility_results_table.csv and .md to {outdir}")

if __name__ == "__main__":
    main()
