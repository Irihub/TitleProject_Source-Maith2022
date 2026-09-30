"""
Optuna hyperparameter search for the new_task model, as described in the
title project proposal ("Metodologia a usar", section 5.1):

Tunable parameters (search range: +-50% around the value used in this
codebase / reported in Maith et al. 2023):
    - GPe Poisson noise rate [Hz]                    (baseline: 100.0)
    - GPeE -> GPe Poisson noise weight [nS]          (baseline: 0.015)
    - STN -> GPe connection probability, same action (baseline: 0.05)
    - STN -> GPe connection probability, diff action (baseline: 0.3)
    - STN -> GPe initial synaptic weight [nS]        (baseline: 0.00063)

Search: 30 Optuna trials (TPE sampler), each evaluating 5 simulations of
30 blocks (4500 blocks total), matching the proposal's stated budget.

Objective (minimized): the mean of three standardized squared errors, one
each for "success rate" (bootstrap_power.per_block_success_rate),
"learning rate" (per-block success-probability slope,
bootstrap_power.per_block_learning_rate), and "switch probability"
(bootstrap_power.per_block_switch_probability), between the model's
pooled blocks for this trial's parameters and the NHP CALIBRATION subset
(70% of the naive-phase blocks, split off once and cached; the remaining
30% is set aside untouched for the final TOST validation, see
load_naive_calibration_stats() below). Success rate and learning rate are
both derived from the same underlying success matrix, so they share the
same calibration/validation row split; switch uses its own independent
split of the switch matrix (see load_naive_calibration_stats()).

This script only ORCHESTRATES simulations: for each trial it launches
parallel.py as a subprocess per simulation (as everywhere else in this
repo, since ANNarchy compiles a fresh C++ network per process), waits for
them to finish, reads their saved output, then deletes the raw per-sim
files it no longer needs. It requires the CLI parameters added to
parallel.py (see the diff shown separately) and reuses
extract_model_pilot.blocks_from_simulation() plus bootstrap_power's
metric functions, so those two files must be importable (see sys.path
setup below) and must not have diverged from what this script expects.

Usage
-----
    cd simulations/new_task/1_srcSim
    python hpo_search.py

Re-running this script continues the same persistent Optuna study
(stored in hpo_study.db next to this file) rather than starting over.

NOT YET RUN END-TO-END: this was written and syntax-checked, but not
executed against a real ANNarchy environment (none is available where
this was authored). Recommended before committing to the full run: do a
smoke test with OPTUNA_TRIALS=1, SIMS_PER_TRIAL=1, BLOCKS_PER_SIM=2 (edit the
constants below) to confirm the subprocess/import/cleanup plumbing works
on your machine before launching the real 30x5x30-block search.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Make extract_model_pilot.py / bootstrap_power.py importable (they live in
# analyses/thesis_power_analysis/, a different folder in this repo, and this
# script deliberately reuses their block-extraction and metric functions
# rather than re-implementing them a second time).
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[2]
ANALYSES_DIR = REPO_ROOT / "analyses" / "thesis_power_analysis"
sys.path.insert(0, str(ANALYSES_DIR))

from extract_model_pilot import blocks_from_simulation  # noqa: E402
from bootstrap_power import (  # noqa: E402
    load_trial_matrix,
    per_block_success_rate,
    per_block_learning_rate,
    per_block_switch_probability,
)

# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------
BASELINE = {
    "gpe_poisson_rate": 100.0,
    "gpe_poisson_weight": 0.015,
    "stn_gpe_prob_same": 0.05,
    "stn_gpe_prob_diff": 0.3,
    "stn_gpe_init_weight": 0.00063,
}
# +-50% around each baseline value, as specified in the proposal.
RANGES = {name: (0.5 * value, 1.5 * value) for name, value in BASELINE.items()}

OPTUNA_TRIALS = 1
SIMS_PER_TRIAL = 5
BLOCKS_PER_SIM = 30
N_TRIALS_PER_BLOCK = 15  # matches extract_model_pilot.py's default; keep in sync

CALIBRATION_FRACTION = 0.70
CALIBRATION_SEED = 20240101


def _safe_remove(path: Path) -> None:
    """Path.unlink(missing_ok=True) needs Python >=3.8; this repo targets 3.7.4."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _split_indices(n: int, calib_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_calib = int(round(n * calib_frac))
    return idx[:n_calib], idx[n_calib:]


def load_naive_calibration_stats(force_resplit: bool = False) -> dict:
    """
    Loads the naive-phase NHP data (the same behaviorSuccessNaive.xlsx /
    behaviorSwitchNaive.xlsx files run_power_analysis.py's FILES["naive"]
    points at), splits its blocks into a calibration subset (used here)
    and a validation subset (reserved for the final TOST), and returns the
    calibration subset's success-rate/learning-rate/switch mean and SD.

    The split is cached to naive_calibration_split.npz on first run and
    reused afterwards, so re-running/resuming this search never silently
    recomputes a different split -- which would break the
    calibration/validation independence the title project relies on.

    Success rate and learning rate are both derived from the same success
    matrix, so they share one split (lr_calib_idx/lr_valid_idx). Switch
    uses its own independent split of the switch matrix (different seed),
    matching how run_power_analysis.py already treats success/switch: two
    separate per-block arrays, never required to be row-aligned with each
    other (a block "incomplete" in one sheet isn't assumed to line up with
    the same row in the other).
    """
    cache_path = ANALYSES_DIR / "naive_calibration_split.npz"
    validation_path = ANALYSES_DIR / "naive_validation_blocks.npz"

    if cache_path.exists() and not force_resplit:
        cached = np.load(cache_path)
        return {k: cached[k].item() for k in cached.files}

    success = load_trial_matrix(ANALYSES_DIR / "behaviorSuccessNaive.xlsx", sheet_name="Sheet1")
    switch = load_trial_matrix(ANALYSES_DIR / "behaviorSwitchNaive.xlsx", sheet_name="Sheet1")
    success_all = per_block_success_rate(success)
    learning_rate_all = per_block_learning_rate(success)
    switch_all = per_block_switch_probability(switch)

    lr_calib_idx, lr_valid_idx = _split_indices(len(learning_rate_all), CALIBRATION_FRACTION, CALIBRATION_SEED)
    sw_calib_idx, sw_valid_idx = _split_indices(len(switch_all), CALIBRATION_FRACTION, CALIBRATION_SEED + 1)

    stats = {
        "success_mean": float(success_all[lr_calib_idx].mean()),
        "success_sd": float(success_all[lr_calib_idx].std(ddof=1)),
        "learning_rate_mean": float(learning_rate_all[lr_calib_idx].mean()),
        "learning_rate_sd": float(learning_rate_all[lr_calib_idx].std(ddof=1)),
        "switch_mean": float(switch_all[sw_calib_idx].mean()),
        "switch_sd": float(switch_all[sw_calib_idx].std(ddof=1)),
    }
    np.savez(cache_path, **stats)

    # Reserve the validation subset untouched, for the final TOST later.
    np.savez(
        validation_path,
        success_validation=success[lr_valid_idx],
        switch_validation=switch[sw_valid_idx],
    )
    print(f"[calibration] naive success: mean={stats['success_mean']:.4f} "
          f"sd={stats['success_sd']:.4f} (n_calib={len(lr_calib_idx)})")
    print(f"[calibration] naive learning_rate: mean={stats['learning_rate_mean']:.4f} "
          f"sd={stats['learning_rate_sd']:.4f} (n_calib={len(lr_calib_idx)})")
    print(f"[calibration] naive switch: mean={stats['switch_mean']:.4f} "
          f"sd={stats['switch_sd']:.4f} (n_calib={len(sw_calib_idx)})")
    print(f"[calibration] validation subsets saved to {validation_path} "
          f"(n_valid success={len(lr_valid_idx)}, switch={len(sw_valid_idx)}) "
          f"-- reserved for the final TOST, do not use before then.")
    return stats


def run_simulations(sim_ids: list[int], params: dict, num_blocks: int) -> None:
    """Launches len(sim_ids) parallel.py subprocesses concurrently and waits
    for all of them to finish. Raises RuntimeError if any of them fail."""
    procs = []
    for sim_id in sim_ids:
        cmd = [
            sys.executable, "parallel.py", str(sim_id),
            "--num-blocks", str(num_blocks),
            "--gpe-poisson-rate", str(params["gpe_poisson_rate"]),
            "--gpe-poisson-weight", str(params["gpe_poisson_weight"]),
            "--stn-gpe-prob-same", str(params["stn_gpe_prob_same"]),
            "--stn-gpe-prob-diff", str(params["stn_gpe_prob_diff"]),
            "--stn-gpe-init-weight", str(params["stn_gpe_init_weight"]),
        ]
        log_path = SRC_DIR / f"hpo_log_sim{sim_id}.txt"
        log_file = open(log_path, "w")
        proc = subprocess.Popen(cmd, cwd=SRC_DIR, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((sim_id, proc, log_file, log_path))

    failures = []
    for sim_id, proc, log_file, log_path in procs:
        proc.wait()
        log_file.close()
        if proc.returncode != 0:
            failures.append((sim_id, log_path))
        else:
            _safe_remove(log_path)  # only keep logs for failed runs, for debugging

    if failures:
        details = ", ".join(f"sim {sid} (see {path})" for sid, path in failures)
        raise RuntimeError(f"parallel.py failed for: {details}")


def cleanup_simulation_files(sim_id: int) -> None:
    """Removes one simulation's raw output after its metrics have been
    extracted, to keep disk usage bounded across up to 150 simulations."""
    for name in (
        f"selection_sim{sim_id}.npy", f"failed_blocks_sim{sim_id}.npy", f"output_sim{sim_id}",
        f"mw_c_sd1_sim{sim_id}.npy", f"mw_c_sd2_sim{sim_id}.npy", f"mw_c_stn_sim{sim_id}.npy",
        f"mw_c_thal_sim{sim_id}.npy", f"mw_stn_gpe_sim{sim_id}.npy",
        f"gpe_rate_values_sim{sim_id}.npy", f"gpe_rate_times_sim{sim_id}.npy",
        f"choice_times_sim{sim_id}.npy",
    ):
        _safe_remove(SRC_DIR / name)
    shutil.rmtree(SRC_DIR / f"annarchy_sim{sim_id}", ignore_errors=True)


def objective(trial: optuna.Trial) -> float:
    params = {name: trial.suggest_float(name, *RANGES[name]) for name in BASELINE}
    sim_ids = [trial.number * SIMS_PER_TRIAL + i + 1 for i in range(SIMS_PER_TRIAL)]

    try:
        run_simulations(sim_ids, params, BLOCKS_PER_SIM)
    except RuntimeError as e:
        print(f"[trial {trial.number}] simulation failure, penalizing: {e}")
        return 1e6

    all_success, all_switch = [], []
    for sim_id in sim_ids:
        success_rows, switch_rows = blocks_from_simulation(SRC_DIR, sim_id, n_trials=N_TRIALS_PER_BLOCK)
        # Checked independently: success_rows/switch_rows can have different
        # lengths (a block can be NaN-padded/dropped in one metric and not
        # the other -- see blocks_from_simulation's docstring).
        if len(success_rows):
            all_success.append(success_rows)
        if len(switch_rows):
            all_switch.append(switch_rows)
        cleanup_simulation_files(sim_id)

    if not all_success or not all_switch:
        print(f"[trial {trial.number}] no usable blocks for these parameters "
              f"(success blocks={len(all_success)}, switch blocks={len(all_switch)}); penalizing.")
        return 1e6

    success_matrix = np.vstack(all_success)
    switch_matrix = np.vstack(all_switch)

    success_mean = float(per_block_success_rate(success_matrix).mean())
    learning_rate_mean = float(per_block_learning_rate(success_matrix).mean())
    switch_mean = float(per_block_switch_probability(switch_matrix).mean())

    calib = load_naive_calibration_stats()
    loss_success = ((success_mean - calib["success_mean"]) / calib["success_sd"]) ** 2
    loss_learning_rate = ((learning_rate_mean - calib["learning_rate_mean"]) / calib["learning_rate_sd"]) ** 2
    loss_switch = ((switch_mean - calib["switch_mean"]) / calib["switch_sd"]) ** 2
    loss = (loss_success + loss_learning_rate + loss_switch) / 3

    trial.set_user_attr("success_mean", success_mean)
    trial.set_user_attr("learning_rate_mean", learning_rate_mean)
    trial.set_user_attr("switch_mean", switch_mean)
    trial.set_user_attr("n_blocks", int(len(success_matrix)))
    return loss


def main() -> None:
    load_naive_calibration_stats()  # compute/cache the split once, up front

    study = optuna.create_study(
        study_name="new_task_hpo",
        storage=f"sqlite:///{SRC_DIR / 'hpo_study.db'}",
        sampler=TPESampler(seed=CALIBRATION_SEED, multivariate=True),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    print("\nBest parameters found:")
    for name, value in study.best_params.items():
        print(f"  {name}: {value:.6g}  (baseline: {BASELINE[name]:.6g})")
    print(f"Best objective (standardized squared error): {study.best_value:.4f}")

    df = study.trials_dataframe()
    df.to_csv(SRC_DIR / "hpo_trials.csv", index=False)
    print(f"\nFull trial history saved to {SRC_DIR / 'hpo_trials.csv'}")
    print(f"Optuna study saved to {SRC_DIR / 'hpo_study.db'} "
          f"(resumable: re-running this script continues the same study).")


if __name__ == "__main__":
    main()
