# CO2 behaviour in COF-999: a fine-tuned UMA-S potential, its DFT benchmark, and the simulations built on it

This repository holds the data and code supporting:

> **Molecular origins of CO2 capture behavior in Amine-appended nanoporous frameworks**
> Liping Liu, Zihui Zhou, Hilal Daglar, Ilja Siepmann*, Omar M. Yaghi*, Laura Gagliardi*
> ChemRxiv, https://doi.org/10.26434/chemrxiv.15003518/v3

Amine-appended covalent organic frameworks-999 (COF-999) capture CO2 directly from air, but the reactive
step — CO2 chemisorption into an amine to form a carbamate/carbamic acid/bicarbonate — is real events and out of thereaches
of ab initio molecular dynamics on a framework of this size (~2000 atoms). This work closes that gap
in four stages: create a diverse set of COF dataset with DFT, fine-tune FAIRChem's **UMA-S**
machine-learned interatomic potential (MLIP) on them, verify the fine-tuned model against
DFT and pretrained models, and then use it to run the nanosecond-scale simulations that DFT
cannot afford — simulated annealing to find the framework's stacking, 100 ps equilibrium MD
to characterise its hydrogen-bond network and amine accessibility, and 160 well-tempered
metadynamics windows to obtain per-amine CO2 chemisorption free energies in dry and humid
conditions.

Everything needed to reproduce, re-use, or audit those four stages is here.

---

## Where the files are

| | |
|---|---|
| **Code repository** | https://github.com/liping-ai4cat/CO2-behavior-in-COF-999 |
| **Data record** | Zenodo, [10.5281/zenodo.22741585](https://doi.org/10.5281/zenodo.22741585) |

The full bundle is **12.5 GB**, and 19 individual files exceed GitHub's 100 MB per-file hard
limit. It is therefore split. **The directory layout below is identical in both places**, so
a path in this README means the same thing whichever half you have:

| | size | contents |
|---|---|---|
| **Git repository** | 178 MB, 407 files | every script and config; all per-system metrics and energetics tables; the RDF, hydrogen-bond, accessibility and speciation tables; all figures; the validation and test DFT trajectories; `HILLS`/`COLVAR` for the two worked metadynamics examples |
| **Zenodo record** | 13 files, 11.0 GB (421 files, 12.3 GB extracted) | `Metadynamics/*/stationary_points_src/` (318 files, 6.98 GB); `all_force_component_errors.csv` (12 files, 1.51 GB); `*forces_components.pdf` (44 files, 1.35 GB); `inference_ckpt_18000.pt` (1.17 GB); `uma.traj` (32 files, 0.53 GB); `meta_40_stride.traj` (358 MB); `DFT_Dataset/VASP-input/example/` (12 files, 254 MB); `DFT_Dataset/train/train.traj` (159 MB) |

If you clone the repository alone you can redraw every reported figure and inspect every
protocol, but you will not have the model checkpoint or the training trajectory. Download
those from the Zenodo record. Zenodo stores files without directory structure, so the bulk
ships as tar archives whose members carry paths relative to this tree — extracting them at
the repository root restores the exact layout described below:

```bash
for a in /path/to/downloads/*.tar /path/to/downloads/*.tar.gz; do tar -xf "$a" -C .; done
cp /path/to/downloads/FT-uma-s-v3_inference_ckpt_18000.pt \
   Training/fine-tuned-UMA-s-models/FT-uma-s-v3/inference_ckpt_18000.pt
```

`ZENODO_CONTENTS.md` in the record maps every archive to what it restores, and `MD5SUMS` /
`SHA256SUMS` cover every file. The record holds 13 downloadable items totalling 11.0 GB,
which expand to the 421 files and 12.3 GB listed below (`vasprun.xml` and the per-component
error CSVs compress heavily; the trajectories and PDFs barely at all). Throughout this README, paths that are **linked** are in the
repository; paths shown in plain code font and marked *(data record)* are not.

A [.gitignore](.gitignore) excludes exactly this bulk, so `git add -A` in a fresh clone
reproduces the code half without manual pruning.

## Directory map

| directory | size | files | what it is |
|---|---|---|---|
| [Diversity_selection/](Diversity_selection/) | 28 KB | 2 | SOAP descriptors + k-means selection of which structures to label with DFT |
| [DFT_Dataset/](DFT_Dataset/) | 454 MB | 26 | the DFT reference data (train/val/test) and a complete worked VASP input/output example |
| [Training/](Training/) | 1.1 GB | 15 | the UMA-S fine-tuning configuration, its full training log, and the production checkpoint |
| [Benchmarking/](Benchmarking/) | 3.3 GB | 356 | MLIP-vs-DFT error analysis: five models on train/val, two models on 12 held-out systems |
| [computational_results/](computational_results/) | 7.0 GB | 426 | the production simulations — simulated annealing, equilibrium MD, and metadynamics |

Two helper scripts at the root reproduce the *release* rather than the science.
[github_sharing.sh](github_sharing.sh) runs a publication preflight — file sizes against
GitHub's limits, symlinks, licensed material, secrets, leftover placeholders — then commits
and pushes the code half. [zenodo_deposit.sh](zenodo_deposit.sh) packages the bulk into
archives that extract back to these same paths, uploads them, and mints the data DOI. Both
default to a sandbox/dry mode and require a typed confirmation before anything is published.

## The workflow, in order

```
1. Diversity_selection/     SOAP descriptors -> k-means -> pick structures to label
        |
2. DFT_Dataset/             VASP PBE-D3 single points -> train / val / test
        |
3. Training/                fine-tune UMA-S (uma-s-1p1) on the "omc" head
        |
4. Benchmarking/            verify against DFT on train/val and on held-out chemistry
        |
5. computational_results/   5a. Simulated annealing  -> all_relaxed_atoms.traj
                                    |
                            5b. Equilibrium MD (100 ps)    5c. Metadynamics (2 ns x 80 (40 Primary + 40 Secondary amines) x 2 (dry+wet))
```

Stages 5b and 5c both start from the annealed structures produced by 5a, so annealing must
run first. Stages 1–3 were iterated as three rounds of active learning, rounds 1–3
(v1 → v2 → v3); the
benchmark in stage 4 is what shows each round improving.

---

## 1. Diversity selection

Two scripts, run in sequence, decide which candidate structures are worth the cost of a DFT
calculation.

**[SOAP-generation.py](Diversity_selection/SOAP-generation.py)** builds one global descriptor
per structure using `dscribe`:

| parameter | value |
|---|---|
| `r_cut` | 6.0 Å |
| `n_max` / `l_max` | 9 / 3 |
| `periodic` | `True` |
| element averaging | per-atom SOAP averaged **by element in the fixed order C, H, N, O**, then concatenated |

The result is a `4 x n_feat` vector per structure; an element absent from a structure
contributes a zero block, which keeps the descriptor length constant. Structures are streamed
in batches through a multiprocessing pool and appended to CSV as `natoms, features`.

**[k-means-for-diversity.py](Diversity_selection/k-means-for-diversity.py)** then selects a
subset. Features are standardised (`StandardScaler`), clustered with `MiniBatchKMeans`
(`batch_size 2048`) above 5,000 structures or `KMeans(n_init="auto")` below, using
`k = ceil(sqrt(N))` clamped to [2, 200].

Selection is **cluster-coverage based rather than farthest-point**: for every cluster that
contains no structure already selected in an earlier round, the structure **closest to the
cluster centroid** is taken. Clusters already covered are skipped. This is what makes each
active-learning round add only genuinely new regions of descriptor space instead of
re-sampling what the model has already seen. An optional second strategy targets two picks
per cluster, adding the structure farthest from the centroid.

Outputs are `*_selected.csv` (the picks, as row indices into the feature matrix),
`*_labeled.csv` (cluster assignment for every structure), a cached `*_features.npy`, and a
PCA(200) → t-SNE map colouring clusters and overlaying each round's selections.

> Note: the script's docstring asks for `dscribe==2.1.1`; the environment actually used was
> **2.1.2**. See [Software environment](#software-environment).

## 2. DFT reference data

### Calculation settings

All reference calculations are **VASP** static single points with stress. A complete worked
example — inputs and outputs for one 822-atom structure — is in
`DFT_Dataset/VASP-input/example/` *(data record)*. It carries the inputs and the
converged outputs (`OUTCAR`, `OSZICAR`, `CONTCAR`, `vasprun.xml`, `EIGENVAL`, `IBZKPT`,
`XDATCAR`); the `PROCAR` and `DOSCAR` that `LORBIT = 11` produced incidentally are not
deposited, as no result in this work uses projected or density-of-states output. The `INCAR`
verbatim:

```
ADDGRID = True     ENAUG = 1360      IVDW = 11        LREAL = Auto
ALGO = Normal      ENCUT = 520.0     LAECHG = False   LVTOT = False
EDIFF = 1e-07      GGA = Pe          LASPH = True     NELM = 500
IBRION = -1        ISIF = 3          LCHARG = False   NELMDL = -10
ISMEAR = 0         ISPIN = 1         LELF = False     NPAR = 8
ISYM = 0           SIGMA = 0.1       LMIXTAU = True   NSW = 0
                                     LORBIT = 11      PREC = Normal
                                     LWAVE = False
```

In words:

| | |
|---|---|
| functional | PBE (`GGA = Pe`) |
| dispersion | **Grimme D3 with zero damping** (`IVDW = 11`) |
| plane-wave cutoff | 520 eV (1.3 x the 400 eV maximum `ENMAX` of the potentials), `ENAUG` 1360 eV |
| k-points | Γ-centred **1 x 1 x 3** (pymatgen, grid density 1060/atom) → 2 irreducible points |
| electronic convergence | `EDIFF = 1e-7` eV, up to 500 SCF steps |
| smearing | Gaussian, `SIGMA = 0.1` eV |
| spin | **non-spin-polarised** (`ISPIN = 1`) |
| symmetry | switched off (`ISYM = 0`) |
| ionic steps | none — static (`IBRION = -1`, `NSW = 0`), stress computed (`ISIF = 3`) |
| accuracy aids | `PREC Normal` with `ADDGRID`, `LASPH` (aspherical gradient corrections) |

Pseudopotentials are PAW PBE, in the order H, C, N, O:

```
PAW_PBE H 15Jun2001   ZVAL=1  ENMAX=250 eV
PAW_PBE C 08Apr2002   ZVAL=4  ENMAX=400 eV
PAW_PBE N 08Apr2002   ZVAL=5  ENMAX=400 eV
PAW_PBE O 08Apr2002   ZVAL=6  ENMAX=400 eV
```

**VASP `POTCAR` files are licence-restricted and are not redistributed here.** The `TITEL`
strings above identify them exactly; any VASP licensee can reproduce the input from their own
distribution.

### The datasets

All splits are **ASE binary trajectories** (`.traj`). Every frame is periodic and carries a
`SinglePointCalculator` exposing `energy`, `forces`, `free_energy`, and `stress` (6-component
Voigt, eV/Å³):

| split | file | frames |
|---|---|---|
| train | `DFT_Dataset/train/train.traj` *(data record)* | **4,469** |
| validation | [DFT_Dataset/val/val.traj](DFT_Dataset/val/val.traj) | **497** |
| test | [DFT_Dataset/test/](DFT_Dataset/test/), 11 trajectories | **1,466** |

**Train and validation are deliberately not COF-999 only.** Roughly **500 frames are drawn
from the OMC (organic molecular crystal) dataset** and mixed in, so that fine-tuning on a
narrow chemistry does not destroy the base model's general organic-crystal accuracy. The
COF-999 portion is 3,974 train and 442 validation frames; the remainder (495 and 55) is the
OMC admixture. Consistent with this, system sizes span 28–1,560 atoms (median 588), and 246
train / 30 validation frames contain elements beyond C/H/N/O — S, Cl, Br, F, P, Si, B, I. The
rest of the OMC admixture is CHNO-only organics. **This matters when reading the benchmark
tables** (see [Known caveats](#known-caveats)).

Upstream these existed as two sibling splits, and the scripts in this bundle reference both:

| upstream split | frames train / val | contents | used for |
|---|---|---|---|
| `round3_SOAP_with_omc_data` | **4,469 / 497** | COF-999 + the ~500-frame OMC admixture | **fine-tuning**; deposited here as `DFT_Dataset/{train,val}` |
| `round3_SOAP_only_COF-999` | **3,974 / 442** | COF-999 only — zero frames with elements beyond C/H/N/O | **the train/validation benchmark tables** below |

The training log settles which was used to fit the model: it records **5,418,358 training
atoms**, exactly twice the 2,709,179 atoms of `with_omc_data/train` under the 2x oversampling,
and **312,692 validation atoms**, exactly `with_omc_data/val`. The `only_COF-999` split would
give 5,292,328 and 307,139.

Rounds are numbered from 1, so the model version equals the round: `FT-uma-s-v1` is round 1
and `FT-uma-s-v3` is round 3. **On disk, and in every path hard-coded in the scripts and
logs, these two directories are named `round2_SOAP_with_omc_data` and
`round2_SOAP_only_COF-999`** — a legacy of the earlier 0-indexed numbering. The filesystem
was deliberately not renamed, so the paths quoted in [Reproducing](#reproducing) stay
literally correct.

The test split is COF-998/999/1000 chemistry only, in 11 independent families that were never
seen during training:

| label in the figures | directory / file stem | frames | atoms | formula (frame 0) |
|---|---|---|---|---|
| COF-999 (dry) | `the_lowest_energy_states_from_annealing_MD` | 93 | 876 | C348H456N60O12 |
| \*CO₂ᴾʰʸˢⁱ (dry) | `Physisorption_basin` | 16 | 879 | C349H456N60O14 |
| \*CO₂ᶜʰᵉᵐ (dry) | `chemisorption` | 1,000 | 372 | C157H185N21O9 |
| COF-999 (wet) | `the_lowest_energy_states_from_annealing_MD+H2O` | 61 | 969 | C348H518N60O43 |
| SA-MD (dry) | `Annealing_MD` | 50 | 876 | C348H456N60O12 |
| SA-MD (wet) | `mutiple_H2O_MD` | 50 | 894 | C348H468N60O18 |
| 1CO₂ MTD (dry) | `CO2_adsorption_dry_COF999` | 41 | 879 | C349H456N60O14 |
| 1CO₂ MTD (wet) | `CO2_adsorption_humid_COF999` | 39 | 972 | C349H518N60O45 |
| nCO₂ MTD (dry) | `mutiple_CO2_dry_META` | 50 | 912 | C360H456N60O36 |
| nCO₂ MTD (wet) | `mutiple_CO2_humid_META` | 50 | 1005 | C360H518N60O67 |
| 1×1×3 supercell | `113` | 16 | 1314 | C522H684N90O18 |

Rows are in the order the families appear in the manuscript figures. **The mapping is not
inferable from the directory names** — the wet counterpart of `Annealing_MD` is
`mutiple_H2O_MD`, and the two `CO2_adsorption_*_COF999` directories carry the single-CO₂
metadynamics labels. Go through this table rather than guessing from a directory name; the
directory name is what you need to locate the files on disk.

[gather_all_images.py](DFT_Dataset/test/gather_all_images.py) is the script that assembled
these from the raw VASP `OUTCAR`s. Rather than take every ionic step, it frame-samples each
trajectory by **farthest-point sampling along the energy axis**: seed with the
minimum- and maximum-energy frames, then repeatedly add the frame whose nearest-selected
energy gap is largest, stopping at `MIN_DELTA_E = 1e-4` eV, capped at
`MAX_SELECT_PER_FILE = 100` (scaled up for long trajectories). This gives energy-diverse
frames rather than 100 near-identical ones from a converged relaxation tail.

## 3. Fine-tuning

The production model is a full fine-tune of **UMA-S 1.1** (`uma-s-1p1.pt`) driven by the
`fairchem` v2 Hydra CLI:

```bash
fairchem -c Training/fine-tuning-UMA-s/configs_FT-uma-s-v3/uma_sm_conserve_finetune.yaml
```

Config tree: [configs_FT-uma-s-v3/](Training/fine-tuning-UMA-s/configs_FT-uma-s-v3/) —
the main YAML plus `cluster/`, `dataset/`, `element_refs/`, and `tasks/` overrides. Files
suffixed `_bk`, `_debug`, or `_finetune` are inactive alternatives kept for reference; the
ones the run actually composed are `cluster/h100.yaml`, `dataset/uma.yaml`,
`element_refs/uma_v1_hof_lin_refs.yaml`, and `tasks/uma_conserving_stress.yaml`.

The complete run log, including all 28 evaluation blocks, is
[training_log_FT-uma-s-v3.out](Training/fine-tuning-UMA-s/training_log_FT-uma-s-v3.out).

### The shipped checkpoint

`Training/fine-tuned-UMA-s-models/FT-uma-s-v3/inference_ckpt_18000.pt` (1.1 GB, data
record).

**This is step 18,000 of 23,415 — roughly epoch 3.8 of 5 — not the final-epoch checkpoint.**
Step 18,000 was the evaluation point with the lowest validation energy MAE in the whole run,
so it was exported as the production model. Validation accuracy there and at the end:

| quantity | at step 18,000 (shipped) | at step 23,415 (final) |
|---|---|---|
| energy MAE | **0.0805 eV** (0.0002 eV/atom) | 0.0874 eV (0.0002 eV/atom) |
| forces MAE | **0.0051 eV/Å** (cosine similarity 0.9850) | 0.0051 eV/Å (0.9851) |
| stress MAE | **0.0002 eV/Å³** | 0.0003 eV/Å³ |
| total loss | 0.0156 | 0.0156 |

`FT-uma-s-v3` is the only checkpoint distributed. The `v1` and `v2` models from the earlier
active-learning rounds appear in the benchmark tables below, and the exact checkpoint paths
they used are recorded in the corresponding `Script.py`, but the weights themselves are not
deposited — `v3` supersedes them and is the model used for every simulation in the paper.

Load it with:

```python
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
predictor = load_predict_unit("path/to/inference_ckpt_18000.pt", device="cuda")
atoms.calc = FAIRChemCalculator(predictor, task_name="omc")
```

The `task_name="omc"` argument is required — the fine-tuned head is registered under that
task, and requesting any other task will fail or silently mispredict.

## 4. Benchmarking against DFT

Two evaluation campaigns, both driven by the same ~583-line `Script.py` in which **only the
checkpoint path differs** between model directories.

For every frame of every trajectory the script extracts the reference DFT energy, forces and
stress, computes the MLIP prediction, and writes:

| output | contents |
|---|---|
| `<system>_energetics.csv` | per-frame DFT vs MLIP energy (eV/atom), `delta_e_eV`, force MAE, total and RMS force norms, stress Frobenius norm and trace, pressure in GPa |
| `per_system_metrics.csv` | 26 columns: MAE / RMSE / mean error / R² / N for energy, total force, RMS force, force components, and stress components |
| `all_force_component_errors.csv` | one row per force component — the source of the large file sizes |
| `uma.traj` | the input frames with MLIP predictions attached as `info`/`arrays` entries |
| `*_parity.pdf`, `*_forces_components.pdf`, `*_stress_components.pdf` | parity plots |

Energies are reported in **meV/atom**, forces in **eV/Å**, stress in **eV/Å³**, pressure in
**GPa**.

### Train / validation: five models

[Benchmarking/train_val_errors_copy/](Benchmarking/train_val_errors_copy/) — this is where the
active-learning progression is evidenced.

| model | energy MAE (meV/atom) train / val | force-component MAE (eV/Å) train / val | stress-component MAE (eV/Å³) val |
|---|---|---|---|
| pretrained `uma-s-1p1` | 1.089 / 0.822 | 0.01489 / 0.01522 | 0.000400 |
| pretrained `uma-m-1p1` | 0.887 / 0.710 † | 0.00911 / 0.00908 † | 0.000378 † |
| FT-uma-s-v1 (round 1) | 0.353 / 0.351 | 0.00620 / 0.00627 | 0.000372 |
| FT-uma-s-v2 (round 2) | 0.206 / 0.201 | 0.00557 / 0.00561 | 0.000358 |
| **FT-uma-s-v3 (round 3)** | **0.125 / 0.128** | **0.00499 / 0.00507** | **0.000340** |

Each round of active learning cuts the energy error by roughly 40%, ending ~6.4x better than
the pretrained UMA-S. Train and validation errors track each other closely at every round,
so the gain is generalisation rather than memorisation.

**† The `uma-m-1p1` row is not computed over the same frames as the others.** That model
returned no prediction for any structure larger than 981 atoms, so its numbers cover 3,457
of 3,974 train and 375 of 442 validation frames — see [Known caveats](#known-caveats). Its
energy MAE is absent from `per_system_metrics.csv` and is computed here from the per-frame
`delta_e_eV` column of `all_energetics.csv`.

Because of that, comparing uma-m with the other models requires restricting all of them to
the frames uma-m could evaluate. On that common subset (energy MAE in meV/atom; force MAE
here is the per-frame mean of `force_mae_eVA`, not the component-weighted figure used in the
table above):

| model | energy MAE train / val | force MAE train / val |
|---|---|---|
| pretrained `uma-s-1p1` | 1.145 / 0.845 | 0.01379 / 0.01398 |
| pretrained `uma-m-1p1` | 0.887 / 0.710 | 0.00901 / 0.00890 |
| FT-uma-s-v1 | 0.291 / 0.280 | 0.00554 / 0.00560 |
| FT-uma-s-v2 | 0.177 / 0.169 | 0.00509 / 0.00513 |
| **FT-uma-s-v3** | **0.122 / 0.125** | **0.00476 / 0.00486** |

The larger pretrained model is indeed better than the smaller one — 0.710 vs 0.845 meV/atom
on validation — but **every fine-tuned uma-s model beats pretrained uma-m**, by 2.5x at round
1 and by 5.7x at round 3, on structures both can handle. Fine-tuning the small model buys
more than scaling up the pretrained one.

### Held-out test: pretrained vs production model

[Benchmarking/true_test/](Benchmarking/true_test/) — 11 systems of independent chemistry.

Ordered by pretrained error, largest first.

| label in the figures | directory | frames | energy MAE (meV/atom) pretrained → FT-v3 | force-component MAE (eV/Å) pretrained → FT-v3 |
|---|---|---|---|---|
| nCO₂ MTD (wet) | `mutiple_CO2_humid_META` | 50 | 1.956 → **0.106** | 0.0186 → **0.0056** |
| \*CO₂ᶜʰᵉᵐ (dry) | `chemisorption` | 1,000 | 1.132 → **0.118** | 0.0083 → **0.0032** |
| nCO₂ MTD (dry) | `mutiple_CO2_dry_META` | 50 | 0.871 → **0.039** | 0.0184 → **0.0057** |
| 1CO₂ MTD (wet) | `CO2_adsorption_humid_COF999` | 39 | 0.584 → **0.151** | 0.0161 → **0.0049** |
| COF-999 (wet) | `the_lowest_energy_states_from_annealing_MD+H2O` | 61 | 0.575 → **0.113** | 0.0089 → **0.0040** |
| SA-MD (wet) | `mutiple_H2O_MD` | 50 | 0.453 → **0.131** | 0.0141 → **0.0048** |
| SA-MD (dry) | `Annealing_MD` | 50 | 0.308 → **0.102** | 0.0146 → **0.0046** |
| 1×1×3 supercell | `113` | 16 | 0.261 → **0.069** | 0.0071 → **0.0035** |
| 1CO₂ MTD (dry) | `CO2_adsorption_dry_COF999` | 41 | 0.206 → **0.085** | 0.0141 → **0.0049** |
| COF-999 (dry) | `the_lowest_energy_states_from_annealing_MD` | 93 | 0.198 → **0.123** | 0.0071 → **0.0038** |
| \*CO₂ᴾʰʸˢⁱ (dry) | `Physisorption_basin` | 16 | 0.116 → 0.130 | 0.0076 → **0.0039** |

The largest gains are exactly where they matter for this work: chemisorbed CO₂ (9.6x) and
the multi-CO₂ metadynamics configurations (22x dry, 18x wet) — the reactive and crowded
configurations the pretrained model never saw. Force accuracy improves on every system
without exception. The physisorption basin is the one system where the pretrained energy MAE
is already at the floor (0.116 meV/atom) and fine-tuning does not improve it.


## 5. Production simulations

All three campaigns drive the **same** `FT-uma-s-v3` checkpoint with `task_name="omc"`,
through `FAIRChemCalculator` with
`InferenceSettings(tf32=True, activation_checkpointing=True, merge_mole=True, compile=True)`.

The systems, all as ASE `.traj` (there are no CIF or POSCAR files in this bundle):

| system | atoms | formula |
|---|---|---|
| COF-999 monolayer | 438 | C174H228N30O6 (30 N, 24 amine) |
| dry 4-layer cell (`114`) | 1,752 | C696H912N120O24 — 96 amine N, c = 16.118 Å |
| humid 4-layer cell | 1,944 | C696H1040N120O88 — framework + **64 H2O** |
| metadynamics cells | 1,755 / 1,947 | the above **+ exactly one CO2** |

### 5a. Simulated annealing — framework structure search

[computational_results/Simulated-annealing_MD/](computational_results/Simulated-annealing_MD/)
(52 MB). **No CO2 is present** — this stage determines the framework's own interlayer
stacking, the number of layers needed for convergence, and, in the humid cells, how the water
organises.

Each annealing cycle is a heat–hold–cool–quench loop:

| phase | protocol |
|---|---|
| dynamics | Langevin NVT, `dt = 1 fs`, `friction = 0.02 fs⁻¹` |
| heat | 173 → 473 K in 3 K increments x 50 steps (101 stages ≈ 5.05 ps) |
| hold | 5,000 steps = **5 ps at 473 K** |
| cool | 473 → 173 K on the same ladder (≈ 5.05 ps) |
| quench | `LBFGS` on a `FrechetCellFilter` — **full cell + ionic relaxation** — to `fmax = 0.03 eV/Å`, ≤ 500 steps |

That is ~15.1 ps per cycle at a nominal ramp of 0.06 K/fs, repeated up to 40 times, with each
quenched minimum appended to `all_relaxed_atoms.traj`. Basin hopping
(`relax_with_basin`, ASE `BasinHopping`) is present in the scripts but **commented out and
not used** in the reported runs.

Ten searches are included. Energies are for frame 0 of each trajectory:

| search | atoms | c (Å) | E (eV) |
|---|---|---|---|
| `dry/111` (1 layer) | 438 | 4.001 | −2703.6414 |
| `dry/112/AA` (2 layers) | 876 | 8.059 | −5407.3634 |
| `dry/113/AA` (3 layers) | 1,314 | 12.003 | −8110.9056 |
| `dry/114/AA` (4 layers) | 1,752 | 16.118 | −10814.7358 |
| `dry/114/Serrated` | 1,752 | 17.240 | **−10815.5799** |
| `dry/114/Staggered` | 1,752 | 17.075 | −10814.1027 |
| `dry/116/AA` (6 layers) | 2,628 | 24.177 | −16222.0901 |
| `dry/118/AA` (8 layers) | 3,504 | 32.236 | −21629.4723 |
| `humid/64H2O_big_cluster` | 1,944 | 15.114 | **−11783.1360** |
| `humid/64H2O_subnanocluster` | 1,944 | 16.166 | −11763.9430 |

Two results follow directly: among the 4-layer stackings, **Serrated is lowest** (0.84 eV
below AA, 1.48 eV below Staggered); and in the humid cell the 64 water molecules strongly
prefer **one large cluster** over dispersed sub-nanometre clusters, by ~19.2 eV.

Only the relaxed-structure trajectories are included — the underlying `uma_run.traj` MD and
the per-round relaxation logs are not (see [Scope](#scope-what-is-not-included)).

### 5b. Equilibrium MD and structural analysis

[computational_results/MD_for_equlibrium/](computational_results/MD_for_equlibrium/) (6.5 MB).
Starting from the lowest-energy annealed frame, for both the dry `114` cell and the humid
64-H2O cell:

| | |
|---|---|
| ensemble | Langevin **NVT**, fixed cell |
| timestep / friction | 1 fs / 0.02 fs⁻¹ |
| initial velocities | `MaxwellBoltzmannDistribution(173 K, force_temp=True)` |
| warm-up | 173 → 300 K in 3 K x 60-step stages (≈ 2.58 ps) |
| production | **100,000 steps = 100 ps at 300 K** |
| sampling | frames every 10 steps (~10,300 frames), log every 20 |

**The trajectories themselves are not included** — only the derived analyses below, each
computed over the last 10,000 frames.

**Radial distribution functions** (`RDF_analysis_*/plot_rdf.py`): `r_max = 10 Å`, 200 bins,
minimum-image PBC. The dry cell is analysed twice — once from the equilibrated trajectory and
once from the as-built structure (`RDF_analysis_Ideal/`) — so that equilibration-induced
structural change is visible rather than assumed. Pairs are N–N, C–C, C–N (dry) and N–N, N–O,
O–O (humid). Outputs are `rdf_*_averaged.csv` / `rdf_*_last10000.csv` with columns
`Distance_r_A, g_r_XY`, plus PNGs.

**Hydrogen-bond maps** (`Hbonds_analysis_equilibrium/Hbonds_elliptical.py`): for each donor
(N in the dry cell; N and O in the humid cell) the bonded H is found within 1.25 Å, then
acceptors within 1.4–5.0 Å; pairs with r < 4.5 Å and θ > 100° are recorded as
(r_XY, ∠D–H···A). These are histogrammed over r ∈ [2.4, 4.0] Å x θ ∈ [130, 180]° on a 100x100
grid and smoothed with `uniform_filter(size=3)`. Each map is overlaid with the **Wernet
elliptical hydrogen-bond criterion** (a = 0.50, b = 45.0; r₀ = 2.90 Å for Ow–Ow, 3.05 Å for
N–O, 3.10 Å for N–N), which separates genuine hydrogen bonds from incidental close contacts.
Outputs are `HBond_Data_*.csv` (`Distance_Angstrom, Angle_Degree, Smoothed_Density`) and PNGs
— N–N for the dry cell, N–N / N–O / O–N / O–O for the humid one.

**Amine accessibility** (`amine_accessibility_analysis/accessibility.py`) asks how many amines
a CO2 molecule can physically reach, across **COF-998, COF-999 and COF-1000**:

| | |
|---|---|
| method | Shrake–Rupley solvent-accessible surface area |
| probe radius | **1.65 Å — a CO2-sized probe**, not the usual water probe |
| atomic radii | Bondi van der Waals |
| sphere sampling | 256-point Fibonacci |
| periodicity | full PBC via `ase.neighborlist` |
| amine identification | any N that is not a nitrile N (nitrile C detected as 2-coordinated); subtype from H count — 2 primary, 1 secondary, 0 tertiary |
| void fraction φ | 15,000-point Monte-Carlo insertion over 40 frames, 3x3x3 ghost images, `cKDTree` |
| statistics | last 10,000 frames, 10-block standard errors |

| material | amines | f_acc (group SASA > 5 Å²) | ⟨group-SASA⟩ (Å²) | ⟨N-SASA⟩ (Å²) | void fraction φ |
|---|---|---|---|---|---|
| COF-998 | 72 | 0.061 ± 0.002 | 1.06 ± 0.05 | 0.21 | 0.056 ± 0.001 |
| COF-999 | 96 | 0.147 ± 0.005 | 2.87 ± 0.11 | 0.47 | 0.199 ± 0.001 |
| COF-1000 | 112 | 0.155 ± 0.004 | 3.58 ± 0.09 | 0.73 | 0.275 ± 0.001 |

Accessibility rises monotonically from COF-998 to COF-1000, and breaking it down by subtype
shows the effect is carried almost entirely by **primary** amines (⟨group-SASA⟩ 1.87 / 5.54 /
9.17 Å² for COF-998/999/1000); tertiary amines are effectively inaccessible everywhere
(≤ 0.02 Å²). Drift between the first and second half of the 100 ps window is ≤ 0.49 Å², so
the metric is equilibrated. `summarize.py` renders the summary table and checks that
monotonicity; `plot_metric.py` draws the individual publication panels.

> Caution: `plot_metric.py` calls `shutil.rmtree(matplotlib.get_cachedir())` at import time.
> That deletes your matplotlib font cache. It is harmless but surprising — the cache is
> rebuilt on next use.

### 5c. Metadynamics — CO2 chemisorption free energies

[computational_results/Metadynamics/](computational_results/Metadynamics/) (6.9 GB). This is
the central result: the free-energy surface for a **single CO2 reacting at a single, specific
amine site**, computed independently for **80 dry sites** (N index 1638–1751) and **79 humid
sites** (1830–1943).

PLUMED is driven through `ase.calculators.plumed.Plumed`, wrapping the UMA calculator.

| | |
|---|---|
| ensemble | Langevin **NVT**, 300 K |
| timestep / friction | 1 fs / 0.01 fs⁻¹ |
| length | `N_STEPS = 2,000,000` = **2 ns per window** |
| sampling | frames every 25 steps |
| hydrogen mass | **set to 3.0 amu** — mass repartitioning, so a 1 fs step remains stable through reactive proton transfer |
| thermalisation | 295 → 300 K in 1 K x 10-step stages, then 1 ps at 300 K (skipped on restart) |
| restart | three-tier: resume `meta.traj` with PLUMED `RESTART` → else `MD_round_heating.traj` → else `plus_CO2.traj` |

**Collective variables.** Two, chosen to separate the approach of CO2 from the proton
transfer that accompanies carbamate formation:

- **`d1`** — `DISTANCE` between the target amine N and the CO2 carbon. The reaction
  coordinate for C–N bond formation.
- **`cv2`** — `COMBINE ARG=cn1,cn2 COEFFICIENTS=1.0,-2.0`, i.e. **CN(N–H) − 2·CN(O–H)**,
  where `cn1` is the coordination of the amine N with the hydrogens initially bonded to it
  and `cn2` is the coordination of the two CO2 oxygens with those same hydrogens (both
  `R_0 = 1.4 Å, NN = 6, MM = 12`). This distinguishes NH2/NH from carbamate
  (NHCOO⁻, proton retained on N) and carbamic acid (NHCOOH, proton moved to a CO2 oxygen).

The H and O atom lists are detected per job (H within 1.25 Å of the N, O within 1.50 Å of the
C), so each window's PLUMED input is specific to its site.

**There is no `plumed.dat` file in this bundle, and none was ever written.**
`build_plumed_input()` inside the driver assembles the input as a Python list of strings and
hands it to PLUMED at runtime, so **the driver script is the authoritative record** of the
metadynamics setup. Reconstructed for the dry worked example (1-based N index 1642, C index
1753):

```
UNITS LENGTH=A ENERGY=eV TIME=fs
d1:    DISTANCE ATOMS=1642,1753
cn1:   COORDINATION GROUPA=1642 GROUPB=<H within 1.25 A of N>  R_0=1.4 NN=6 MM=12
cn2:   COORDINATION GROUPA=<the 2 CO2 O>  GROUPB=<same H>      R_0=1.4 NN=6 MM=12
cv2:   COMBINE ARG=cn1,cn2 COEFFICIENTS=1.0,-2.0 PERIODIC=NO
lw1:   LOWER_WALLS ARG=d1  AT=0.800  KAPPA=1000 EXP=4
uw1:   UPPER_WALLS ARG=d1  AT=<GRID_MAX_D-0.3> KAPPA=1000 EXP=4
lw2:   LOWER_WALLS ARG=cv2 AT=-4.800 KAPPA=1000 EXP=4
uw2:   UPPER_WALLS ARG=cv2 AT=2.800  KAPPA=1000 EXP=4
metad: METAD ARG=d1,cv2 SIGMA=0.2,0.15 HEIGHT=0.10 PACE=200 BIASFACTOR=10 FILE=HILLS \
             GRID_MIN=0.500,-5.000 GRID_MAX=<...>,3.000 GRID_BIN=200,300
PRINT STRIDE=25 ARG=d1,cn1,cn2,cv2,metad.bias FILE=COLVAR
```

Well-tempered parameters: **bias factor γ = 10**, hill height **0.10 eV**, deposition every
**200 steps**, σ = **0.2 Å** on `d1` and **0.15** on `cv2`, on a 200 x 300 grid spanning
`d1 ∈ [0.5, max(d1_initial + 1.5, 5.5)]` Å and `cv2 ∈ [−5.0, 3.0]`. Four restraining walls
(`KAPPA = 1000, EXP = 4`) sit just inside the grid edges to keep the system on the grid. The
first hill height is written as `0.1111 = 0.10 x γ/(γ−1)`, PLUMED's standard well-tempered
convention; heights decay to ~1e-4 eV by the end of a converged window.

**Free-energy reconstruction** (`example_job/cal_plot.py`) shells out to

```bash
plumed sum_hills --hills HILLS --kt 0.025852 --outfile FES.dat    # kT at 300 K, in eV
```

then defines the CO2(g) basin (large `d1`, `cv2` ≈ +1.7) and the bound `*CO2` basin
(`d1` ≈ 1.4 Å, `cv2` ≈ 0.7) and locates the **minimax (bottleneck) barrier** between them two
independent ways — bisection on an 8-neighbour flood-fill connectivity threshold, and an
explicit Dijkstra-style minimax path that also identifies the transition-state cell. It
reports F_A, F_B, F_TS, ΔG and forward/reverse barriers in both eV and kJ/mol, and renders the
annotated FES contour.

Two complete worked examples are included, each with its `HILLS`, `COLVAR`, driver,
analysis script and rendered surface:

| example | Fa (kJ/mol) | ΔF (kJ/mol) | CO2(g) | TS | *CO2 |
|---|---|---|---|---|---|
| dry, job 1641 | 35.93 | −136.48 | −55.63 | −19.70 | −192.11 |
| humid, job 1937 | 37.88 | −88.65 | −112.96 | −75.08 | −201.61 |

**Stationary points.** `stationary_points_src/` holds the MD frames whose collective-variable
values fall in the bound-state (`sB`) and transition-state (`sTS`) cells identified above:
**160 trajectories for the dry campaign** (80 sites x `sB`/`sTS`) and **158 for the humid one**
(79 sites x `sB`/`sTS`). Frame counts vary per file — `sB` is typically 200 frames, `sTS`
between 3 and 200. These are the structural evidence behind the free energies, and they feed
the two speciation analyses below. Initial-state (`sA`) frames are not deposited: the CO2(g)
basin is characterised by the free-energy surface itself and by `plus_CO2.traj`.

**What the bound state actually is** (`sB_where_is_CO2/where_is_CO2.py`): every frame of every
`*_sB.traj` is classified by covalent-radius connectivity (`R_SCALE = 1.20`), counting
hydrogens on the carbamate N and on the two CO2 oxygens, into `NHCOOH` (carbamic acid),
`NHCOO⁻` (carbamate), `NCOOH`, `NCOO`, `NH2COO` (zwitterion), or `HOCOH+NH`; in humid mode it
also counts H3O⁺ among the 64 waters. Jobs are partitioned by their metadynamics energetics
into `part_1` (ΔG ≤ 0 **and** Ga ≤ 70 kJ/mol) and `part_2` (ΔG ≤ 0, Ga > 70). The dry campaign
yields 23 `part_1` and 25 `part_2` sites, with dominant species NHCOO⁻ 17, NHCOOH 11, NCOO 10,
NCOOH 9 — i.e. **both carbamate and carbamic acid form, in comparable proportions**, rather
than a single product. Humid: 38 `part_1`, 19 `part_2`.

**Where the proton goes** (`sB_where_is_H/where_is_H_v5.py`): a reference H count per N and O
is built from `plus_CO2.traj`, each N is classified primary/secondary/tertiary and each O as
CO2- or water-derived, and every `sB` frame is then scored for sites that gained or lost a
proton relative to reference. Whenever a protonated and a deprotonated N coexist, all
N⁺···N⁻ distances are recorded (5,431 dry / 7,363 humid records) — quantifying the
intra-framework proton shuttling and ion-pair separation that accompanies carbamate
formation.

**Starting structures.** `plus_CO2.traj` holds the 80 candidate CO2 placements, one per amine
site: 1,755 atoms (C697H912N120O26) dry, 1,947 atoms (C697H1040N120O90) humid — the
equilibrated framework plus exactly one CO2. Each job reads one frame by index; the template
carries the literal placeholder `index=MMM`, and the worked examples use `index=3` (dry) and
`index=77` (humid). Job directory numbers encode the 0-based N index.

---

## Software environment

**Two different software stacks were used, and the distinction matters.** All analysis and all
production simulations ran in one environment; the fine-tuning run that produced the
checkpoint predates it and used an older `fairchem`/`torch` pair. Both are exported here as
complete `conda` specifications.

| | [environment.yml](environment.yml) | [environment-training.yml](environment-training.yml) |
|---|---|---|
| used for | **all analysis, benchmarking, and production simulations** | the fine-tuning run only |
| Python | 3.12.12 | 3.12 |
| `fairchem-core` | **2.13.0** | **2.4.0** |
| `torch` | **2.8.0** | **2.6.0** |
| `ase` | 3.27.0 | 3.26.0 |
| `e3nn` | 0.5.9 | 0.5.6 |
| `plumed` | **2.9.2** | — |
| `dscribe` | — | 2.1.2 |
| also | numpy 2.3.5, scipy 1.17.0, pandas 3.0.0, matplotlib 3.10.8, seaborn 0.13.2, scikit-learn 1.8.0, pymatgen 2025.10.7, numba 0.63.1, hydra-core 1.3.2, omegaconf 2.3.0, lmdb 1.7.3 | torchtnt 0.2.4, ase-db-backends 0.10.0, wandb 0.21.1 |

```bash
conda env create -f environment.yml            # analysis + simulations
conda env create -f environment-training.yml   # only if re-running the fine-tune
```

Which environment each stage needs:

| stage | environment |
|---|---|
| Diversity selection | `environment-training.yml` — it is the one with `dscribe` |
| Fine-tuning | `environment-training.yml` |
| Benchmarking, annealing, equilibrium MD, analysis | `environment.yml` |
| Metadynamics | `environment.yml` — it is the one with `plumed` |

Both exports pin CUDA 12 wheels (`nvidia-*-cu12`) and assume Linux x86-64 with an NVIDIA GPU;
drop those lines for a CPU-only install. The base `uma-s-1p1.pt` checkpoint is **not**
redistributed here — obtain it from
[FAIR Chemistry](https://github.com/facebookresearch/fairchem), which requires accepting
Meta's model terms.

## Reproducing

Scripts retain the absolute paths they ran with (`/scratch/...`, `/projects/...`,
`/home/lipingliu/.cache/fairchem/...`). **These are left unedited on purpose** — they are the
historical record of how the calculations were actually performed. To re-run anything, two
kinds of path need repointing:

1. **The training data path.** `configs_FT-uma-s-v3/dataset/uma.yaml` sets
   `snapshot_dir: /scratch/lipingliu/COF_999_CO2_capture/fine-tuning` with splits under
   `fine-tune-v10-selection_SOAP_latent/round2_SOAP/COF-999/{train,val}`. That directory no
   longer exists; the data was reorganised to
   `fine-tune-v10-selection_SOAP_latent/round2_SOAP_with_omc_data/{train,val}`, and those are
   the files deposited here as `DFT_Dataset/train/` and `DFT_Dataset/val/`. Point
   `snapshot_dir` there. Note the config expects `format: ase_db` whereas the deposited data
   is `.traj` — convert, or set `format` accordingly.
2. **The checkpoint and input paths in every driver and benchmark script** — `MODEL_ID_path`
   and `TRAJ_DIR` near the top of each `Script.py`. Point `MODEL_ID_path` at
   `Training/fine-tuned-UMA-s-models/FT-uma-s-v3/inference_ckpt_18000.pt`.

Also note `gather_all_images.py` refers to
`/scratch/lipingliu/double_Z/new_benchmarking_true_test`, the VASP output tree it originally
harvested; that raw tree is not deposited (see [Scope](#scope-what-is-not-included)).

The SLURM submission scripts used on the authors' clusters are **not** deposited: they encode
local partitions, accounts and walltimes and carry no scientific content. Every script here is
run directly (`python -u <script>.py`, or `fairchem -c <config>.yaml` for the fine-tune). One
consequence: `Benchmarking/train_val_errors_copy/*/Submit_new.py` is a job-array helper whose
final action is `sbatch nequip_falcon.qsub`, so it cannot run as-is — the evaluation it fans
out is the `Script.py` beside it, which does run directly.

## Known caveats

Four things a careful reader will notice. None affects the reported conclusions, but all
would be confusing if left unexplained.

1. **The train/validation benchmark tables count fewer frames than the deposited splits.**
   `per_system_metrics.csv` in `train_val_errors_copy/` reports `n_energy = 3974` (train) and
   `442` (validation), whereas the deposited `train.traj` and `val.traj` hold 4,469 and 497
   frames. This is not an inconsistency. Those benchmark runs pointed at the sibling
   `round3_SOAP_only_COF-999` split — the pure-COF-999 set, verified to contain **zero** frames
   with elements beyond C/H/N/O — while the deposited splits are `round3_SOAP_with_omc_data`,
   which add the ~500-frame OMC admixture (495 train, 55 validation). The quoted benchmark
   errors are therefore **COF-999 errors**, undiluted by general organic-crystal frames,
   whereas fine-tuning used the full combined splits. See
   [The datasets](#the-datasets).

2. **`pretrained-uma-m` silently produced no prediction for large structures.** Its energy
   fields in `per_system_metrics.csv` are blank, but the underlying data is not missing:
   `all_energetics.csv` holds all 3,974 train rows, of which **517 (13%) have an empty
   `uma_energy_eV`**, and a single non-finite value propagated through the aggregate, leaving
   it blank. The failures are not random — they are a clean size cutoff:

   | structure size | train frames | failed | val frames | failed |
   |---|---|---|---|---|
   | ≤ 968 atoms | 3,386 | 0 (0%) | 362 | 0 (0%) |
   | 969–981 atoms | 389 | 318 (82%) | 49 | 36 (74%) |
   | > 981 atoms | 199 | 199 (**100%**) | 31 | 31 (**100%**) |
   | total | 3,974 | 517 (13%) | 442 | 67 (15%) |

   The largest structure uma-m evaluated successfully has 981 atoms; the smallest it failed
   on has 969. Below 969 atoms nothing fails, above 981 everything does, and 969–981 is a
   partial band — the signature of a memory limit that depends on neighbour-list size rather
   than on atom count alone.

   Forces are missing on exactly the same frames, which is what the reduced component count
   (6,345,150 vs 7,938,492) reflects. The log records no error, so the most likely cause is
   GPU memory exhaustion on the larger model, failing per structure without raising.

   Consequences: uma-m's row covers only the smaller 87% of the set, and **any comparison
   against it must be restricted to those frames** — the matched-subset table in
   [Benchmarking](#train--validation-five-models) does this. Recompute with
   `pd.read_csv(...).dropna(subset=["uma_energy_eV"])`. Nothing else in the paper depends on
   uma-m; it is a reference point, not a production model.

3. **The `true_test` aggregate files contain duplicated rows.** The evaluation script
   discovers inputs with `rglob("*.traj")`, which matched both a system's trajectory and a
   nested copy of it. As a result `per_system_metrics.csv` has **18 rows for 11 unique
   systems** and `all_energetics.csv` has **2,766 rows for 1,466 unique `(system, frame)`
   pairs**. Deduplicate before use:

   ```python
   df = pd.read_csv("all_energetics.csv").drop_duplicates(["system", "frame"])
   metrics = pd.read_csv("per_system_metrics.csv").groupby("system").last()
   ```

   The duplicated rows carry near-identical values, so per-system metrics are unaffected;
   only naive row counts and unweighted aggregates would be.

4. **The `HILLS`/`COLVAR` time axis is not in femtoseconds.** PLUMED was given
   `timestep = 1.0` while `UNITS TIME=fs`, so one ASE time unit (10.1805 fs) was recorded as
   one step and the `time` column advances ~10.18 fs per step — inflated by 10.18x. The
   physical duration of a window is `n_hills x 200 x 1 fs`: the dry worked example reached
   ≈ 1.70 ns (8,506 hills, restarted) and the humid one 2.00 ns (10,003 hills, complete).
   **The well-tempered bias itself is unaffected**, since the metadynamics update depends on
   deposition count rather than wall-clock time, and `sum_hills` integrates over hills. Only
   the plotted time axis must be rescaled.

## Scope: what is not included

The release covers what is needed to reproduce the reported results and re-use the model. The
raw trajectories behind it total roughly a terabyte and are impractical to deposit. Not
included:

- **Raw MD and metadynamics trajectories** — `meta.traj` from the 159 metadynamics windows
  (~690 GB), `MD_UMA_final.traj` from the 100 ps equilibrium runs, and `uma_run.traj` from the
  annealing cycles. The derived tables, stationary-point structures, and two complete worked
  metadynamics examples are deposited in their place.
- **`COLVAR`/`HILLS` for the 157 windows other than the two worked examples.** The derived
  per-site free energies are what the figures and conclusions use.
- **Per-round annealing and MD logs** (`MD_round*.log`, `relax_round*.log`).
- **Per-job driver scripts.** Each of the 159 metadynamics jobs had its own copy, differing
  only in the frame index it pinned (`read(STRUCTURE, index=N)`) and the GPU partition in its
  filename. The campaign templates plus one substituted worked example per campaign are
  included instead.
- **The base `uma-s-1p1.pt` checkpoint** — distributed by FAIR Chemistry under its own terms.
- **FT-uma-s-v1 and v2 weights** — superseded by v3; their benchmark results are included.
- **VASP `POTCAR` files** — licence-restricted. Identified by `TITEL` string above.
- **Initial-state (`sA`) stationary-point trajectories** for either metadynamics campaign.
- **`PROCAR` and `DOSCAR` from the worked VASP example** — incidental `LORBIT = 11` output,
  unused in this work.
- **A superseded 8-system benchmark run**, fully replaced by the 12-system evaluation above.

Raw trajectories and the withheld bias data are available from the corresponding authors
on reasonable request.

## Licence

| | |
|---|---|
| Code (scripts, configs) | MIT — [LICENSE](LICENSE) |
| Data (trajectories, tables, figures) | CC-BY-4.0 |
| `inference_ckpt_18000.pt` | **[FAIR Chemistry License](https://huggingface.co/facebook/UMA)** |

The fine-tuned checkpoint is a derivative work of Meta's UMA-S and is released under the FAIR
Chemistry License, subject to the
[FAIR Chemistry Acceptable Use Policy](https://huggingface.co/facebook/UMA). Using it
requires having accepted Meta's terms for the base UMA model. See
[Training/fine-tuned-UMA-s-models/README.md](Training/fine-tuned-UMA-s-models/README.md).

## Citing this work

Please cite the paper:

```bibtex
@ARTICLE{Liu2026-MLIP,
  title        = {Molecular origins of {CO2} capture behavior in Amine-appended
                  nanoporous frameworks},
  author       = {Liu, Liping and Zhou, Zihui and Daglar, Hilal and Siepmann,
                  Ilja and Yaghi, Omar M and Gagliardi, Laura},
  journaltitle = {ChemRxiv},
  date         = {2026-08-10},
  urldate      = {2026-08-13},
  language     = {en}
}
```

and, if you use the deposited data or checkpoint, the data record:

```bibtex
@dataset{Liu2026-COF999-data,
  title     = {Molecular origins of {CO2} capture behavior in amine-appended
               nanoporous frameworks: DFT reference data, a fine-tuned UMA-S
               potential, and simulation results},
  author    = {Liu, Liping and Zhou, Zihui and Daglar, Hilal and Siepmann,
               Ilja and Yaghi, Omar M and Gagliardi, Laura},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.22741585},
  url       = {https://doi.org/10.5281/zenodo.22741585}
}
```

Methods this work depends on and which deserve their own citation: **UMA / fairchem** for the
base model and training framework, **PLUMED** for metadynamics, **DScribe** for the SOAP
descriptors, **ASE** for structure handling and dynamics, and **VASP** for the reference
calculations.
