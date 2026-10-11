"""
Showcase the data produced by parallel.py (selection_sim*.npy, output_sim*,
failed_blocks_sim*.npy) for a whole folder of simulations, POOLED together.

Earlier version of this script (simulations/014a_new_dopamine_cluster/1_srcSim/
show_results.py) produced one row of plots PER simulation -- fine for a
handful of sims, but unusable for 30+. This version concatenates every
requested simulation's trials into a single pooled sequence and shows ONE
summary table + ONE row of plots for the whole folder instead.

Usage
-----
    python show_results.py                  # uses all sims found in --dir (see DEFAULT_DIR below)
    python show_results.py --sims 1 2 3      # only sims 1, 2 and 3
    python show_results.py --dir /some/other/folder
    python show_results.py --save results.png  # override the default output path

What it reads (per simulation id N, same format as before)
------------------------------------------------------------
- output_simN            : tab-separated trial log written by np.savetxt in
                            parallel.py. One row per *real* trial.
- selection_simN.npy      : shape (num_blocks*100, 2), columns [decision,
                            correct]. Pre-allocated larger than needed; only
                            used here to sanity-check alignment with output_simN.
- failed_blocks_simN.npy  : shape (num_blocks,), 1 if a block did not finish
                            successfully, 0 otherwise.

How pooling works
------------------
Every simulation's trials are concatenated end-to-end into one combined
DataFrame, with a new sequential 'pooled_trial' index and a globally unique
'pooled_block' id (so block boundaries across different simulations never
collide). The x-axis for the trial-level plots is therefore "pooled trial
(all sims concatenated)", not any single simulation's own real timeline --
this is an explicit, deliberate simplification to get ONE readable view of
the whole dataset instead of 30 separate ones.

What it shows
-------------
1. A single printed summary (total trials/blocks pooled across every
   simulation, overall accuracy, failed blocks, mean block length, mean
   response time).
2. One row of 3 plots, pooled across all simulations:
   - selected vs. rewarded position over the pooled trial sequence, with
     block boundaries and failed blocks marked
   - rolling accuracy (rewarded-selection rate) over the pooled sequence
   - block-length histogram (every block from every simulation)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reutiliza el estilo/paleta ya validados en analyses/thesis_figures/plot_results.py
# (viven en otra carpeta de este repo) en vez de duplicarlos. No se reutiliza
# su _save_and_maybe_show porque esa funcion guarda siempre bajo el OUTPUT_DIR
# *de ese modulo*; este script define el suyo propio mas abajo.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analyses" / "thesis_figures"))

from plot_results import _style_axes, LINE_COLOR, PRIMARY_INK  # noqa: E402

# Colores de estado (ver skill de dataviz, references/palette.md): reservados
# para "bueno/advertencia/critico", nunca reusados como color de serie.
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"
VIOLET = "#4a3aa7"  # slot categorico 7, solo para el histograma

OUTPUT_COLUMNS = [
    "trial",
    "start",
    "dop_input",
    "end",
    "correct",
    "decision",
    "dopamine",
    "time",
]

# ---------------------------------------------------------------------------
# CONFIGURACION: de donde salen los datos por defecto.
# ---------------------------------------------------------------------------
DEFAULT_DIR = (
    REPO_ROOT / "stashed_runs" / "new_task"
    / "first full run new task and block def - 30 sims 60 blocks - 30-09-2026"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def _save_and_maybe_show(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    try:
        with warnings.catch_warnings():
            # Sin pantalla disponible (p.ej. corriendo en el cluster), matplotlib
            # solo emite un UserWarning, no una excepcion -- se silencia porque
            # la figura ya quedo guardada.
            warnings.simplefilter("ignore", UserWarning)
            plt.show(block=False)
    except Exception:
        pass


def discover_sim_ids(folder: Path) -> list[int]:
    """Find every sim id N for which output_simN exists in `folder`."""
    ids = []
    for f in folder.glob("output_sim*"):
        suffix = f.name[len("output_sim"):]
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def load_simulation(folder: Path, sim_id: int) -> dict:
    """Load and align the three files produced for one simulation id."""
    output_path = folder / f"output_sim{sim_id}"
    selection_path = folder / f"selection_sim{sim_id}.npy"
    failed_path = folder / f"failed_blocks_sim{sim_id}.npy"

    for p in (output_path, selection_path, failed_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected file: {p}")

    trials = pd.read_csv(
        output_path, sep="\t", comment="#", header=None, names=OUTPUT_COLUMNS,
        usecols=range(len(OUTPUT_COLUMNS)),
    )

    selection = np.load(selection_path)[: len(trials)]
    if not np.array_equal(selection[:, 1], trials["correct"].to_numpy()):
        print(
            f"warning: sim {sim_id}: selection_sim{sim_id}.npy does not line up "
            f"with output_sim{sim_id} as expected; using output_sim only",
            file=sys.stderr,
        )

    failed_blocks = np.load(failed_path)

    # Blocks change whenever the rewarded ("correct") position changes.
    block_id = (trials["correct"] != trials["correct"].shift()).cumsum() - 1
    trials = trials.assign(block=block_id)

    return {
        "sim_id": sim_id,
        "trials": trials,
        "failed_blocks": failed_blocks,
    }


def load_all_simulations(folder: Path, sim_ids: list[int]) -> pd.DataFrame:
    """Loads every requested simulation and concatenates their trials into
    one pooled DataFrame, adding:
      - 'pooled_trial': sequential 0-indexed trial number across ALL sims.
      - 'pooled_block': globally unique block id (never collides across sims).
      - 'is_failed_block': per-trial flag, carried over from that sim's own
        failed_blocks_simN.npy for the block that trial belongs to.
      - 'sim_id': which simulation this trial came from (kept for reference).
    """
    pooled_frames = []
    trial_offset = 0
    block_offset = 0

    for sim_id in sim_ids:
        sim = load_simulation(folder, sim_id)
        trials = sim["trials"].copy()
        failed_blocks = sim["failed_blocks"]

        trials["is_failed_block"] = trials["block"].apply(
            lambda b: bool(failed_blocks[b]) if b < len(failed_blocks) else False
        )
        trials["pooled_trial"] = np.arange(len(trials)) + trial_offset
        trials["pooled_block"] = trials["block"] + block_offset
        trials["sim_id"] = sim_id

        pooled_frames.append(trials)
        trial_offset += len(trials)
        block_offset += (int(trials["block"].max()) + 1) if len(trials) else 0

    if not pooled_frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + [
            "block", "is_failed_block", "pooled_trial", "pooled_block", "sim_id",
        ])
    return pd.concat(pooled_frames, ignore_index=True)


def summarize_pooled(pooled: pd.DataFrame, n_sims: int) -> dict:
    rewarded = pooled["decision"] == pooled["correct"]
    no_response = pooled["decision"] <= 0
    block_lengths = pooled.groupby("pooled_block").size()
    n_failed_blocks = pooled.groupby("pooled_block")["is_failed_block"].first().sum()

    return {
        "n_sims": n_sims,
        "trials": len(pooled),
        "blocks": pooled["pooled_block"].nunique(),
        "accuracy_%": round(float(rewarded.mean()) * 100, 1),
        "no_response_%": round(float(no_response.mean()) * 100, 2),
        "failed_blocks": int(n_failed_blocks),
        "mean_block_len": round(float(block_lengths.mean()), 2),
        "mean_response_time_ms": round(float(pooled["time"].mean()), 1),
    }


def print_summary(summary: dict) -> None:
    print("\nPooled summary across all simulations")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def plot_pooled(pooled: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1. Selected vs. rewarded position, pooled across all sims ---
    ax = axes[0]
    rewarded_mask = pooled["decision"] == pooled["correct"]
    ax.plot(pooled["pooled_trial"], pooled["correct"], color=PRIMARY_INK, lw=1, label="rewarded position")
    ax.scatter(
        pooled["pooled_trial"][rewarded_mask], pooled["decision"][rewarded_mask],
        color=STATUS_GOOD, s=6, label="rewarded selection",
    )
    ax.scatter(
        pooled["pooled_trial"][~rewarded_mask], pooled["decision"][~rewarded_mask],
        color=STATUS_CRITICAL, s=6, label="unrewarded selection",
    )
    for _, grp in pooled[pooled["is_failed_block"]].groupby("pooled_block"):
        ax.axvspan(grp["pooled_trial"].min(), grp["pooled_trial"].max(), color=STATUS_WARNING, alpha=0.25)
    ax.set_ylim(0.5, float(pooled["correct"].max()) + 0.5)
    _style_axes(ax, "pooled trial (all sims concatenated)", "position",
                "Selections vs. rewarded position\n(shaded = failed block)")
    ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=PRIMARY_INK)

    # --- 2. Rolling accuracy, pooled ---
    ax = axes[1]
    window = 20
    rolling_acc = rewarded_mask.rolling(window, min_periods=1).mean()
    ax.plot(pooled["pooled_trial"], rolling_acc, color=LINE_COLOR, linewidth=1.5)
    ax.set_ylim(0, 1.05)
    _style_axes(ax, "pooled trial (all sims concatenated)", "rewarded rate",
                f"Rolling accuracy (window={window} trials)")

    # --- 3. Block-length histogram, pooled ---
    ax = axes[2]
    block_lengths = pooled.groupby("pooled_block").size()
    max_len = int(block_lengths.max()) if len(block_lengths) else 1
    ax.hist(block_lengths, bins=range(1, max_len + 2), color=VIOLET, alpha=0.85)
    _style_axes(ax, "trials per block", "# blocks", "Block-length distribution")

    fig.tight_layout()
    _save_and_maybe_show(fig, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dir", type=Path, default=DEFAULT_DIR,
        help=f"folder containing output_sim*/selection_sim*.npy/failed_blocks_sim*.npy (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--sims", type=int, nargs="+", default=None,
        help="specific simulation ids to pool (default: every sim found in --dir)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help=f"override the output PNG path (default: {OUTPUT_DIR / 'pooled_summary.png'})",
    )
    args = parser.parse_args()

    sim_ids = args.sims if args.sims is not None else discover_sim_ids(args.dir)
    if not sim_ids:
        print(f"No output_sim* files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Pooling {len(sim_ids)} simulation(s) from {args.dir}...")
    pooled = load_all_simulations(args.dir, sim_ids)

    summary = summarize_pooled(pooled, n_sims=len(sim_ids))
    print_summary(summary)

    save_path = Path(args.save) if args.save else OUTPUT_DIR / "pooled_summary.png"
    plot_pooled(pooled, save_path)


if __name__ == "__main__":
    main()
