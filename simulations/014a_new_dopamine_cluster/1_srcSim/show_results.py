"""
Showcase the data produced by parallel.py (selection_sim*.npy, output_sim*,
failed_blocks_sim*.npy) for one or several simulations of the never-rewarded
experiment (014a_new_dopamine_cluster).

Usage
-----
    python show_results.py                  # uses all sims found in this folder
    python show_results.py --sims 1 2 3      # only sims 1, 2 and 3
    python show_results.py --dir ../4_dataEv/stn_gpe_factor_idx_3
    python show_results.py --save results.png  # save instead of only showing

What it reads (per simulation id N)
------------------------------------
- output_simN         : tab-separated trial log written by np.savetxt in
                         parallel.py. Columns: Trial, Start, Dop_Input, End,
                         correct, decision, dopamine, time. This file has
                         exactly one row per *real* trial, so it is used to
                         know how many rows of the larger, pre-allocated
                         selection_simN.npy array actually contain data.
- selection_simN.npy   : shape (num_blocks*100, 2), columns [decision, correct].
                         Pre-allocated larger than needed; only the first
                         len(output_simN) rows are real.
- failed_blocks_simN.npy : shape (num_blocks,), 1 if a block did not finish
                         successfully (hit the 30-trial cap or 3 consecutive
                         non-responses), 0 otherwise.

What it shows
-------------
1. A printed summary table (trials, blocks, accuracy, failed blocks, mean
   block length, mean response time) for every simulation, plus totals.
2. A figure with one row of plots per simulation:
   - selected vs. rewarded position over trials, with block boundaries and
     failed blocks marked
   - rolling accuracy (rewarded-selection rate) over trials
   - block-length histogram
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    # selection_simN.npy is pre-allocated to num_blocks*100 rows; only the
    # first len(trials) rows were actually written during the run.
    selection = np.load(selection_path)[: len(trials)]
    # Columns [decision, correct] should match output_simN's own columns;
    # keep output_simN as the source of truth since its header/dtypes are
    # explicit, and only use `selection` to sanity-check the alignment.
    if not np.array_equal(selection[:, 1], trials["correct"].to_numpy()):
        print(
            f"warning: sim {sim_id}: selection_sim{sim_id}.npy does not line up "
            "with output_sim{sim_id} as expected; using output_sim only",
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


def summarize(sim: dict) -> dict:
    trials = sim["trials"]
    n_trials = len(trials)
    n_blocks = trials["block"].nunique()
    rewarded = trials["decision"] == trials["correct"]
    no_response = trials["decision"] <= 0

    block_lengths = trials.groupby("block").size()

    return {
        "sim_id": sim["sim_id"],
        "trials": n_trials,
        "blocks": n_blocks,
        "accuracy": rewarded.mean(),
        "no_response_rate": no_response.mean(),
        "failed_blocks": int(sim["failed_blocks"].sum()),
        "mean_block_len": block_lengths.mean(),
        "mean_response_time_ms": trials["time"].mean(),
    }


def print_summary_table(summaries: list[dict]) -> None:
    df = pd.DataFrame(summaries).set_index("sim_id")
    df["accuracy"] = (df["accuracy"] * 100).round(1)
    df["no_response_rate"] = (df["no_response_rate"] * 100).round(2)
    df["mean_block_len"] = df["mean_block_len"].round(2)
    df["mean_response_time_ms"] = df["mean_response_time_ms"].round(1)
    df = df.rename(
        columns={
            "accuracy": "accuracy_%",
            "no_response_rate": "no_response_%",
        }
    )

    print("\nPer-simulation summary")
    print(df.to_string())

    totals = {
        "trials": df["trials"].sum(),
        "blocks": df["blocks"].sum(),
        "accuracy_%": df["accuracy_%"].mean().round(1),
        "no_response_%": df["no_response_%"].mean().round(2),
        "failed_blocks": df["failed_blocks"].sum(),
        "mean_block_len": df["mean_block_len"].mean().round(2),
        "mean_response_time_ms": df["mean_response_time_ms"].mean().round(1),
    }
    print("\nAcross all simulations (accuracy/response-time are averaged, not summed):")
    for k, v in totals.items():
        print(f"  {k}: {v}")


def plot_simulations(sims: list[dict], save_path: str | None) -> None:
    n = len(sims)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.2 * n), squeeze=False)

    for row, sim in enumerate(sims):
        trials = sim["trials"]
        sim_id = sim["sim_id"]

        # --- 1. Selected vs. rewarded position over trials ---
        ax = axes[row][0]
        rewarded_mask = trials["decision"] == trials["correct"]
        ax.plot(trials.index, trials["correct"], color="black", lw=1, label="rewarded position")
        ax.scatter(
            trials.index[rewarded_mask], trials["decision"][rewarded_mask],
            color="tab:green", s=8, label="rewarded selection",
        )
        ax.scatter(
            trials.index[~rewarded_mask], trials["decision"][~rewarded_mask],
            color="tab:red", s=8, label="unrewarded selection",
        )
        # mark block boundaries (start of each new block)
        block_starts = trials.index[trials["block"] != trials["block"].shift()]
        for b in block_starts:
            ax.axvline(b, color="grey", lw=0.4, alpha=0.5)
        # mark failed blocks (shade the corresponding trial range)
        failed = np.where(sim["failed_blocks"] == 1)[0]
        for b in failed:
            block_trials = trials.index[trials["block"] == b]
            if len(block_trials):
                ax.axvspan(block_trials.min(), block_trials.max(), color="orange", alpha=0.2)
        ax.set_ylim(0.5, 5.5)
        ax.set_ylabel(f"sim {sim_id}\nposition")
        if row == 0:
            ax.set_title("selections vs. rewarded position\n(orange = failed block)")
            ax.legend(loc="upper right", fontsize=7, markerscale=1.5)
        if row == n - 1:
            ax.set_xlabel("trial")

        # --- 2. Rolling accuracy ---
        ax = axes[row][1]
        window = 20
        rolling_acc = rewarded_mask.rolling(window, min_periods=1).mean()
        ax.plot(trials.index, rolling_acc, color="tab:blue")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("rewarded rate")
        if row == 0:
            ax.set_title(f"rolling accuracy\n(window={window} trials)")
        if row == n - 1:
            ax.set_xlabel("trial")

        # --- 3. Block-length histogram ---
        ax = axes[row][2]
        block_lengths = trials.groupby("block").size()
        ax.hist(block_lengths, bins=range(1, 32), color="tab:purple", alpha=0.8)
        ax.set_ylabel("# blocks")
        if row == 0:
            ax.set_title("block-length distribution")
        if row == n - 1:
            ax.set_xlabel("trials per block")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dir", type=Path, default=Path(__file__).parent,
        help="folder containing output_sim*/selection_sim*.npy/failed_blocks_sim*.npy (default: this script's folder)",
    )
    parser.add_argument(
        "--sims", type=int, nargs="+", default=None,
        help="specific simulation ids to show (default: every sim found in --dir)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="save the figure to this path instead of opening an interactive window",
    )
    args = parser.parse_args()

    sim_ids = args.sims if args.sims is not None else discover_sim_ids(args.dir)
    if not sim_ids:
        print(f"No output_sim* files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    sims = [load_simulation(args.dir, sim_id) for sim_id in sim_ids]
    summaries = [summarize(sim) for sim in sims]

    print_summary_table(summaries)
    plot_simulations(sims, args.save)


if __name__ == "__main__":
    main()
