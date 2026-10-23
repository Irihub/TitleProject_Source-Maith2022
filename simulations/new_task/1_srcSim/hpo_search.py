"""
Optuna hyperparameter search for the new_task model, as described in the
title project proposal ("Metodologia a usar", section 5.1).

This script now tunes WHATEVER parameters are listed in BASELINE below --
which parameters that is has changed over time as the search moved on
(see the comments around BASELINE/STUDY_NAME). To switch back to a
previous search: uncomment its BASELINE/STUDY_NAME block, comment out the
current one, and re-run -- nothing about a previous search is deleted,
and each search lives under its own `study_name` inside the SAME
hpo_study.db (Optuna supports multiple independent studies per storage
file), so resuming an old one never overwrites another.

CURRENT search -- Cortex->SD2 (indirect pathway) plasticity parameters
(search range: +-50% around the value hardcoded in parallel.py):
    - CSD2.factor [dimensionless] (dopamine-modulated learning-rate
      multiplier for the Cortex->SD2 synapse; negative by design -- see
      the discussion of IzhikevichSynapse_D2 in parallel.py) (baseline: -0.9)
    - CSD2.delta [nS] (per-presynaptic-spike weight decay) (baseline: 0.000005)

PREVIOUS search -- GPe/STN-GPe parameters (kept commented below):
    - GPe Poisson noise rate [Hz]                    (baseline: 100.0)
    - GPeE -> GPe Poisson noise weight [nS]          (baseline: 0.015)
    - STN -> GPe connection probability, same action (baseline: 0.05)
    - STN -> GPe connection probability, diff action (baseline: 0.3)
    - STN -> GPe initial synaptic weight [nS]        (baseline: 0.00063)

Search: see OPTUNA_TRIALS/N_PARALLEL_OPTUNA_TRIALS/SIMS_PER_TRIAL/
BLOCKS_PER_SIM below for the current budget.

Objective (minimized): the mean of two per-trial curve errors, see
curve_loss() below. For each of "success" and "switch", the model's pooled
blocks for this trial's parameters give one mean value per trial position
within the block (trial 1..N_TRIALS_PER_BLOCK, averaged over blocks), which
is compared with the same curve of the NHP CALIBRATION subset (70% of the
naive-phase blocks, split off once and cached; the remaining 30% is set
aside untouched for the final TOST validation, see
load_naive_calibration_stats() below). The error of each curve is the mean
over trial positions of the plain squared difference, in probability units
(both metrics live in [0, 1], so no standardization is needed). Learning
rate is not in the objective: it is derived from the success matrix
(bootstrap_power.per_block_learning_rate), so matching the success curve
covers it. Success and switch use independent row splits of their
matrices (see load_naive_calibration_stats()).

The previous objective (the mean of three standardized squared errors on
the block-level MEANS of success rate, learning rate and switch
probability) is gone; studies run with it are not comparable to this one
and live under other STUDY_NAMEs.

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
import warnings
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
#
# BASELINE is the single source of truth for which parameters this search
# tunes: each key here must exactly match a parallel.py CLI flag name once
# "_" is replaced by "-" and prefixed with "--" (e.g. "csd2_factor" ->
# "--csd2-factor"), since run_simulations() below builds parallel.py's
# command line generically from this dict (see _flag_name()). To tune a
# different/additional set of parameters: add/remove entries here (and
# make sure parallel.py has a matching --flag with a sensible default),
# change STUDY_NAME so you don't resume the wrong search by accident, and
# re-run -- nothing else in this file needs to change.
# ---------------------------------------------------------------------------

# --- Previous search (GPe/STN-GPe parameters) -- kept for reference, not
# deleted. To resume it: uncomment this BASELINE/STUDY_NAME, comment out
# the ones below, and re-run (it continues under its own study_name,
# independent of whatever search is active below).
# BASELINE = {
#     "gpe_poisson_rate": 100.0,
#     "gpe_poisson_weight": 0.015,
#     "stn_gpe_prob_same": 0.05,
#     "stn_gpe_prob_diff": 0.3,
#     "stn_gpe_init_weight": 0.00063,
# }
# STUDY_NAME = "new_task_hpo"

# --- Current search (Cortex->SD2 plasticity parameters) --------------------
BASELINE = {
    "csd2_factor": -0.9,
    "csd2_delta": 0.000005,
}
# Was "new_task_hpo_csd2" under the previous (block-mean, 3-metric) objective.
# The objective changed to curve_loss(), so a new study name is required:
# with load_if_exists=True, reusing the old name would mix trials scored by
# two different losses in one study.
STUDY_NAME = "new_task_hpo_csd2_curveloss"

# +-50% around each baseline value's MAGNITUDE, as specified in the
# proposal -- i.e. (value - 0.5*|value|, value + 0.5*|value|), not
# (0.5*value, 1.5*value). The two formulas agree for positive baselines
# (e.g. 100.0 -> (50, 150) either way) but the naive 0.5x/1.5x version
# silently produces a BACKWARDS (low > high) range for a negative baseline
# like csd2_factor=-0.9 (0.5*-0.9=-0.45 > 1.5*-0.9=-1.35), which Optuna's
# suggest_float would reject. This version stays correctly ordered and
# keeps both bounds on the same side of zero as the baseline (a 50%
# offset is smaller in magnitude than the baseline itself) -- important
# for csd2_factor, which must stay negative to preserve the D1/D2 sign
# distinction.
RANGES = {name: (value - 0.5 * abs(value), value + 0.5 * abs(value)) for name, value in BASELINE.items()}
RANGES["csd2_factor"] = (-6.0, -0.7)
RANGES["csd2_delta"] = (0.000002, 0.000008)

OPTUNA_TRIALS = 20
N_PARALLEL_OPTUNA_TRIALS = 5
SIMS_PER_TRIAL = 20
BLOCKS_PER_SIM = 45

N_TRIALS_PER_BLOCK = 15  # matches extract_model_pilot.py's default; keep in sync
CALIBRATION_FRACTION = 0.70
CALIBRATION_SEED = 20240101

# Version of what naive_calibration_split.npz stores. Bump it whenever the
# cached stats change meaning, so a stale cache is refreshed instead of being
# silently reused (a stale learning-rate entry went unnoticed for a while).
#   1 (no version key): block-level success/learning-rate/switch mean+SD, with
#      learning rate still the OLS slope of the first version of the analysis.
#   2: learning rate = two-point difference (bootstrap_power.per_block_learning_rate),
#      plus the per-trial mean success/switch curves that curve_loss() uses.
CALIBRATION_STATS_VERSION = 2


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


def _calibration_stats(success: np.ndarray, switch: np.ndarray,
                       lr_calib_idx: np.ndarray, sw_calib_idx: np.ndarray) -> dict:
    """Everything cached for the calibration subset: block-level mean/SD of
    the three metrics (the TOST metrics; no longer in the objective, but used
    by the thesis_sim_overview scripts) and the per-trial mean success/switch
    curves (what curve_loss() compares against)."""
    if success.shape[1] != N_TRIALS_PER_BLOCK or switch.shape[1] != N_TRIALS_PER_BLOCK:
        raise ValueError(
            f"NHP matrices have {success.shape[1]} (success) / {switch.shape[1]} (switch) trial columns, "
            f"but N_TRIALS_PER_BLOCK={N_TRIALS_PER_BLOCK}; the model curves would not line up with them."
        )
    success_calib = success[lr_calib_idx]
    switch_calib = switch[sw_calib_idx]
    success_rate = per_block_success_rate(success_calib)
    learning_rate = per_block_learning_rate(success_calib)
    switch_prob = per_block_switch_probability(switch_calib)
    return {
        "success_mean": float(success_rate.mean()),
        "success_sd": float(success_rate.std(ddof=1)),
        "learning_rate_mean": float(learning_rate.mean()),
        "learning_rate_sd": float(learning_rate.std(ddof=1)),
        "switch_mean": float(switch_prob.mean()),
        "switch_sd": float(switch_prob.std(ddof=1)),
        "success_curve": np.nanmean(success_calib, axis=0),
        "switch_curve": np.nanmean(switch_calib, axis=0),
        "stats_version": CALIBRATION_STATS_VERSION,
    }


def _refresh_stale_calibration_cache(cache_path: Path, validation_path: Path, old_stats: dict) -> dict:
    """Recomputes the cached calibration stats for the EXISTING split.

    Used when naive_calibration_split.npz was written by an older version of
    this file (see CALIBRATION_STATS_VERSION). The split itself is never
    redrawn: the indices are re-derived from the fixed seed and checked
    against both the reserved validation blocks and the old cached
    success/switch stats, and the function refuses to continue if any of
    them disagree. Only the cache file is rewritten; the validation file is
    not touched."""
    success = load_trial_matrix(ANALYSES_DIR / "behaviorSuccessNaive.xlsx", sheet_name="Sheet1")
    switch = load_trial_matrix(ANALYSES_DIR / "behaviorSwitchNaive.xlsx", sheet_name="Sheet1")
    lr_calib_idx, lr_valid_idx = _split_indices(len(success), CALIBRATION_FRACTION, CALIBRATION_SEED)
    sw_calib_idx, sw_valid_idx = _split_indices(len(switch), CALIBRATION_FRACTION, CALIBRATION_SEED + 1)

    if validation_path.exists():
        saved = np.load(validation_path)
        if not (np.array_equal(saved["success_validation"], success[lr_valid_idx])
                and np.array_equal(saved["switch_validation"], switch[sw_valid_idx])):
            raise RuntimeError(
                f"The split re-derived from CALIBRATION_SEED/CALIBRATION_FRACTION does not match the reserved "
                f"validation blocks in {validation_path}; refusing to refresh {cache_path}, because that would "
                f"change which blocks count as calibration."
            )

    stats = _calibration_stats(success, switch, lr_calib_idx, sw_calib_idx)
    for key in ("success_mean", "success_sd", "switch_mean", "switch_sd"):
        if key in old_stats and not np.isclose(old_stats[key], stats[key], rtol=1e-9, atol=0.0):
            raise RuntimeError(
                f"Recomputed {key}={stats[key]!r} differs from the cached {old_stats[key]!r}; the stale cache "
                f"does not come from the current NHP files/split, refusing to overwrite it."
            )

    print(f"[calibration] refreshing stale cache {cache_path.name} (version "
          f"{old_stats.get('stats_version', 1)} -> {CALIBRATION_STATS_VERSION}); split unchanged.")
    for key in ("learning_rate_mean", "learning_rate_sd"):
        if key in old_stats:
            print(f"[calibration]   {key}: {old_stats[key]:.4f} -> {stats[key]:.4f}")
    np.savez(cache_path, **stats)
    return stats


def load_naive_calibration_stats(force_resplit: bool = False) -> dict:
    """
    Loads the naive-phase NHP data (the same behaviorSuccessNaive.xlsx /
    behaviorSwitchNaive.xlsx files run_power_analysis.py's FILES["naive"]
    points at), splits its blocks into a calibration subset (used here)
    and a validation subset (reserved for the final TOST), and returns the
    calibration subset's statistics: the block-level mean/SD of success
    rate, learning rate and switch probability, plus the per-trial mean
    success and switch curves ("success_curve"/"switch_curve", arrays of
    N_TRIALS_PER_BLOCK values) that the objective compares against.

    The split is cached to naive_calibration_split.npz on first run and
    reused afterwards, so re-running/resuming this search never silently
    recomputes a different split -- which would break the
    calibration/validation independence the title project relies on. A
    cache written by an older version of this file (see
    CALIBRATION_STATS_VERSION) is refreshed in place on the SAME split.

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
        stats = {k: (cached[k].item() if cached[k].ndim == 0 else cached[k]) for k in cached.files}
        if stats.get("stats_version") == CALIBRATION_STATS_VERSION:
            return stats
        return _refresh_stale_calibration_cache(cache_path, validation_path, stats)

    success = load_trial_matrix(ANALYSES_DIR / "behaviorSuccessNaive.xlsx", sheet_name="Sheet1")
    switch = load_trial_matrix(ANALYSES_DIR / "behaviorSwitchNaive.xlsx", sheet_name="Sheet1")

    lr_calib_idx, lr_valid_idx = _split_indices(len(success), CALIBRATION_FRACTION, CALIBRATION_SEED)
    sw_calib_idx, sw_valid_idx = _split_indices(len(switch), CALIBRATION_FRACTION, CALIBRATION_SEED + 1)

    stats = _calibration_stats(success, switch, lr_calib_idx, sw_calib_idx)
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


def _flag_name(param_name: str) -> str:
    """Maps a BASELINE key to its parallel.py CLI flag, e.g.
    "csd2_factor" -> "--csd2-factor". This is the naming convention every
    BASELINE entry (and its matching parallel.py argparse flag) must
    follow for run_simulations() to stay fully generic."""
    return "--" + param_name.replace("_", "-")


def run_simulations(sim_ids: list[int], params: dict, num_blocks: int) -> None:
    """Launches len(sim_ids) parallel.py subprocesses concurrently and waits
    for all of them to finish. Raises RuntimeError if any of them fail."""
    procs = []
    for sim_id in sim_ids:
        cmd = [sys.executable, "parallel.py", str(sim_id), "--num-blocks", str(num_blocks)]
        for name, value in params.items():
            cmd.extend([_flag_name(name), str(value)])
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


def curve_loss(success_matrix: np.ndarray, switch_matrix: np.ndarray, calib: dict) -> dict:
    """The objective: the mean of two per-trial curve errors.

    success_matrix/switch_matrix are blocks x N_TRIALS_PER_BLOCK 0/1 matrices
    (NaN-padded for short blocks) in the format blocks_from_simulation()
    returns. For each one, the model's mean over blocks at every trial
    position is compared with the NHP calibration curve (calib["success_curve"]
    / calib["switch_curve"]); the error is the mean over trial positions of
    the plain squared difference, in probability units. Every trial position
    weighs the same, and trial positions with no model data at all are
    skipped. The loss is the mean of the success and the switch error.

    Returns {"loss", "loss_success", "loss_switch", "success_curve",
    "switch_curve"}; the curves are the model's per-trial means.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column gives NaN, skipped below
        success_curve = np.nanmean(success_matrix, axis=0)
        switch_curve = np.nanmean(switch_matrix, axis=0)
        loss_success = float(np.nanmean((success_curve - calib["success_curve"]) ** 2))
        loss_switch = float(np.nanmean((switch_curve - calib["switch_curve"]) ** 2))
    return {
        "loss": (loss_success + loss_switch) / 2,
        "loss_success": loss_success,
        "loss_switch": loss_switch,
        "success_curve": success_curve,
        "switch_curve": switch_curve,
    }


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

    calib = load_naive_calibration_stats()
    result = curve_loss(success_matrix, switch_matrix, calib)

    # Not part of the loss, but the quantities the final TOST compares, so
    # a trial's attributes show how it stands on them.
    trial.set_user_attr("loss_success_curve", result["loss_success"])
    trial.set_user_attr("loss_switch_curve", result["loss_switch"])
    trial.set_user_attr("success_mean", float(per_block_success_rate(success_matrix).mean()))
    trial.set_user_attr("learning_rate_mean", float(per_block_learning_rate(success_matrix).mean()))
    trial.set_user_attr("switch_mean", float(per_block_switch_probability(switch_matrix).mean()))
    trial.set_user_attr("n_blocks", int(len(success_matrix)))
    return result["loss"]


def main() -> None:
    load_naive_calibration_stats()  # compute/cache the split once, up front

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{SRC_DIR / 'hpo_study.db'}",
        sampler=TPESampler(seed=CALIBRATION_SEED, multivariate=True),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=N_PARALLEL_OPTUNA_TRIALS)

    print("\nBest parameters found:")
    for name, value in study.best_params.items():
        print(f"  {name}: {value:.6g}  (baseline: {BASELINE[name]:.6g})")
    print(f"Best objective (mean per-trial squared difference, probability units): {study.best_value:.5f}")

    # Named per STUDY_NAME so switching searches never overwrites a
    # previous search's exported trial history (the study_name itself is
    # always safe either way, since hpo_study.db keeps every study it has
    # ever seen under its own name -- this is just about the derived CSV).
    trials_csv_path = SRC_DIR / f"hpo_trials_{STUDY_NAME}.csv"
    df = study.trials_dataframe()
    df.to_csv(trials_csv_path, index=False)
    print(f"\nFull trial history saved to {trials_csv_path}")
    print(f"Optuna study '{STUDY_NAME}' saved to {SRC_DIR / 'hpo_study.db'} "
          f"(resumable: re-running this script continues the same study; "
          f"other studies in the same file, e.g. a previous search, are untouched).")


if __name__ == "__main__":
    main()
