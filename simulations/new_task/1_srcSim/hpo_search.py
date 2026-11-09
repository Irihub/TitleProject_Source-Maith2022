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

CURRENT search -- Cortex->SD2 (indirect pathway) plasticity parameters plus
the SC (Thalamus) Poisson noise, rate and weight tuned separately (ranges and
discretization steps: see RANGES/STEPS below):
    - CSD2.factor [dimensionless] (dopamine-modulated learning-rate
      multiplier for the Cortex->SD2 synapse; negative by design -- see
      the discussion of IzhikevichSynapse_D2 in parallel.py) (baseline: -0.9)
    - CSD2.delta [nS] (per-presynaptic-spike weight decay) (baseline: 0.000005)
    - CSD2.max_weight [nS] (hard ceiling of the Cortex->SD2 weights; with a
      large csd2_factor all channels saturate against it, which freezes
      this pathway) (baseline: 0.0011)
    - SC noise rate [Hz]: Poisson firing rate of ThalN, the noise input of the
      SC (parallel.py --sc-noise-rate)                         (baseline: 100.0)
    - SC noise weight [nS]: synaptic weight of that noise onto the SC
      (parallel.py --sc-noise-weight)                          (baseline: 0.005)
      Rate and weight change both the average drive (rate * weight) and the
      noise variance (rate * weight^2) of the SC; the single-scale version that
      kept the average drive fixed lives in hpo_search_sc_noise.py.

PREVIOUS search -- GPe/STN-GPe parameters (kept commented below):
    - GPe Poisson noise rate [Hz]                    (baseline: 100.0)
    - GPeE -> GPe Poisson noise weight [nS]          (baseline: 0.015)
    - STN -> GPe connection probability, same action (baseline: 0.05)
    - STN -> GPe connection probability, diff action (baseline: 0.3)
    - STN -> GPe initial synaptic weight [nS]        (baseline: 0.00063)

Search: see OPTUNA_TRIALS/N_PARALLEL_OPTUNA_TRIALS/SIMS_PER_TRIAL/
BLOCKS_PER_SIM below for the current budget. OPTUNA_TRIALS is the TOTAL number
of completed trials the study should reach: re-running the script continues the
same study (hpo_study.db) and only runs the trials still missing.

Stopping it, and what happens to the database
---------------------------------------------
Graceful (no new trial starts; the running ones finish and are saved; nothing
needs fixing afterwards; the same command later continues the search):
    touch STOP_HPO_SEARCH                 (a file next to this script; removed
                                           again at the next start), or
    pkill -TERM -f hpo_search.py          (or Ctrl-C if it runs in the foreground)
A second SIGTERM/SIGINT exits immediately. NOTE: Ctrl-C in a terminal, and the
SIGTERM that SLURM sends at the time limit or on scancel, also reach the
simulations, which then die; their trials are recorded as FAILED (see below).
With an HPO_DEADLINE (Unix timestamp, set by the SLURM job script) the search
stops by itself starting trials that would not finish before it: after each trial
it compares the time left with 1.2 x the longest trial so far.

Hard stop (kill -9, node failure, time limit): the trials that were running stay
marked RUNNING in the database. Nothing has to be repaired by hand: at the next
start the script marks them FAILED (the sampler ignores failed trials) and runs
as many new trials as are missing to reach OPTUNA_TRIALS. A simulation that fails
while the script keeps running is handled the same way (the trial is recorded as
FAILED, not as a fake bad loss). The temporary files of killed simulations
(annarchy_sim*, output_sim*, ...) stay in this folder; they are NOT deleted
automatically, because full runs may share this folder, so the script only warns
and tells you how many there are.

A lock file (hpo_search.lock, containing the PID) prevents two instances from
running at once, which would make the recovery above fail the other instance's
running trials. A lock left by a dead process is detected and replaced (on
non-POSIX systems the liveness check is skipped).

Objective (minimized): the per-point difference between the model and the NHP,
averaged over points and over the three metrics, see pointwise_loss() below.
Each metric is a curve over the trial position within a block, averaged over
the trial's pooled blocks and compared with the same curve of the NHP
CALIBRATION subset (70% of the naive-phase blocks, split off once and cached;
the remaining 30% is set aside untouched for the final TOST validation, see
load_naive_calibration_stats() below):
    - success rate:  mean success at each of the 15 trial positions
    - learning rate: the learning slope, the mean change of success from each
                     trial to the next (14 points; the curve of the plots)
    - switch chance: mean switch probability at each of the 15 positions
For each curve the error is the mean over its points of the squared difference
(POINT_ERROR = "squared"; "absolute" gives the mean absolute difference), in
probability units, so no standardization is needed; the loss is the mean of the
three curve errors. Success and learning rate share one row split of the NHP
data and switch uses its own (see load_naive_calibration_stats()).

This objective rewards matching the SHAPE of the three curves; it does not
measure how far each block-mean metric is from the NHP in the units of the
final TOST. So each trial also records the three TOST offsets, the largest of
them (tost_loss(), the previous objective) and whether all are inside the
margin, to compare candidates on the TOST afterwards.

Previous objectives are gone and studies run with them are not comparable
with this one (they live under other STUDY_NAMEs): the largest absolute offset
of the three TOST metrics, tost_loss() below (kept, and recorded per trial);
the mean of two per-trial curves, curve_loss() below (kept, the analysis scripts
still use it); and before them the mean of three standardized SQUARED errors on
the block means with 3 simulations per trial.

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

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
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
    "csd2_max_weight": 0.0011,
    "sc_noise_rate": 100.0,
    "sc_noise_weight": 0.005,
}
# History of this search's name:
#   "new_task_hpo_csd2"                    -- previous (block-mean, 3-metric) objective
#   "new_task_hpo_csd2_curveloss"          -- curve_loss(), tuning csd2_factor + csd2_delta
#   "new_task_hpo_csd2_maxw_curveloss"     -- curve_loss(), now also tuning csd2_max_weight
#   "new_task_hpo_csd2_maxw_tostloss"      -- tost_loss(): largest |offset| of the 3 TOST metrics
#   "new_task_hpo_csd2_scnoise_pointwise"  -- pointwise_loss(); csd2 x3 + SC noise rate and weight
# Each change of objective or of the tuned parameter set needs a new name:
# with load_if_exists=True, reusing a name would mix trials scored by a
# different loss / over a different search space in one study.
STUDY_NAME = "new_task_hpo_csd2_scnoise_pointwise"

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
RANGES["csd2_max_weight"] = (0.0003, 0.002)
# SC noise: wide ranges on purpose. In the noise-scale search (rate/k at weight*k,
# same average drive) nothing happened up to scale ~3 and the behaviour broke down
# beyond ~5, i.e. weights of 0.015-0.025 at low rates; the +-50% default around the
# baseline would not reach that region, and here rate and weight also move the
# average drive (rate * weight), which that search kept fixed.
RANGES["sc_noise_rate"] = (25.0, 200.0)       # Hz
RANGES["sc_noise_weight"] = (0.0025, 0.025)   # nS

# Discretization of the search space: a parameter listed here is sampled on the
# grid low, low+step, low+2*step, ... up to high (Optuna's suggest_float(step=)),
# which makes the space finite and lets TPE learn from repeated/nearby values.
# A parameter with no entry is sampled continuously. Each range must be a whole
# number of steps (checked by _check_steps() at start-up): otherwise Optuna
# silently moves `high` down to the last grid point.
STEPS = {
    "csd2_factor": 0.1,
    "csd2_delta": 0.0000002,
    "csd2_max_weight": 0.00005,
    "sc_noise_rate": 5.0,
    "sc_noise_weight": 0.00025,
}

# How the difference at each point of a curve is turned into an error in
# pointwise_loss(): "squared" (mean squared difference) or "absolute" (mean
# absolute difference).
POINT_ERROR = "squared"

# OPTUNA_TRIALS is the TOTAL number of completed trials the study should reach,
# not the number per run: starting the script again after a stop, a crash or a
# SLURM time limit runs only the trials still missing (see main()).
OPTUNA_TRIALS = 20
N_PARALLEL_OPTUNA_TRIALS = 5
SIMS_PER_TRIAL = 20
BLOCKS_PER_SIM = 45

# Stopping and restarting (see the module docstring): a file with this name next
# to the script asks for a graceful stop; the lock file holds the PID of the
# running instance; HPO_DEADLINE is an optional environment variable with a Unix
# timestamp after which no trial that could not finish in time is started (the
# SLURM job script sets it from the job's end time).
STOP_FILE = SRC_DIR / "STOP_HPO_SEARCH"
LOCK_FILE = SRC_DIR / "hpo_search.lock"
DEADLINE_ENV_VAR = "HPO_DEADLINE"
DEADLINE_SAFETY_FACTOR = 1.2  # a new trial needs 1.2 x the longest trial so far to fit before the deadline

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
#   3: plus the learning-slope curve (mean change of success from each trial to
#      the next) that pointwise_loss() uses.
CALIBRATION_STATS_VERSION = 3


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
        "slope_curve": np.nanmean(np.diff(success_calib, axis=1), axis=0),
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


TOST_METRICS = ("success", "learning_rate", "switch")  # the three metrics of the final TOST
EQUIVALENCE_MARGIN_SD = 0.2                            # the TOST margin, in NHP block-SD units (proposal: d = 0.2)


def tost_offsets(success_matrix: np.ndarray, switch_matrix: np.ndarray, calib: dict) -> dict:
    """Offset of each TOST metric: (model mean - NHP calibration mean) / NHP
    calibration SD, with the "mean" taken over blocks of the per-block metric
    (bootstrap_power.per_block_*), exactly what the TOST compares. Positive:
    the model is above the NHP. The unit is NHP block-SDs, the unit of the
    equivalence margin."""
    means = {
        "success": per_block_success_rate(success_matrix).mean(),
        "learning_rate": per_block_learning_rate(success_matrix).mean(),
        "switch": per_block_switch_probability(switch_matrix).mean(),
    }
    return {m: float((means[m] - calib[f"{m}_mean"]) / calib[f"{m}_sd"]) for m in TOST_METRICS}


def tost_loss(success_matrix: np.ndarray, switch_matrix: np.ndarray, calib: dict) -> dict:
    """The objective: the largest absolute offset among the three TOST metrics
    (see the module docstring). Returns {"loss", "offsets", "worst_metric",
    "within_margin"}, where within_margin says whether all three offsets are
    inside +-EQUIVALENCE_MARGIN_SD (necessary, not sufficient, for the TOST:
    the final test also needs the margin to hold with the sampling
    uncertainty, which in practice asks for offsets well below the margin)."""
    offsets = tost_offsets(success_matrix, switch_matrix, calib)
    worst_metric = max(offsets, key=lambda m: abs(offsets[m]))
    loss = abs(offsets[worst_metric])
    return {
        "loss": loss,
        "offsets": offsets,
        "worst_metric": worst_metric,
        "within_margin": loss <= EQUIVALENCE_MARGIN_SD,
    }


def learning_slope_curve(success_matrix: np.ndarray) -> np.ndarray:
    """The learning slope per trial: for each trial position t, the mean over
    blocks of (success at t+1) - (success at t), N_TRIALS_PER_BLOCK - 1 points.
    This is the curve of the "learning slope vs trial" plot (per-block
    differences, then averaged, so the NaN padding of short blocks only affects
    its own block). Its mean over the points is the block-mean learning rate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column gives NaN, skipped by the losses
        return np.nanmean(np.diff(success_matrix, axis=1), axis=0)


def pointwise_loss(success_matrix: np.ndarray, switch_matrix: np.ndarray, calib: dict) -> dict:
    """The objective: the per-point difference between model and NHP, averaged
    over the points of each curve and over the three curves (see the module
    docstring). The curves are the model's pooled-block means per trial position
    for success and switch, and learning_slope_curve() for the learning rate;
    they are compared with calib["success_curve"], calib["switch_curve"] and
    calib["slope_curve"]. Points with no model data are skipped. POINT_ERROR
    chooses squared or absolute differences.

    Returns {"loss", "terms" (one error per curve), "curves" (the model's)}."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        curves = {
            "success": np.nanmean(success_matrix, axis=0),
            "learning_rate": learning_slope_curve(success_matrix),
            "switch": np.nanmean(switch_matrix, axis=0),
        }
        references = {"success": calib["success_curve"], "learning_rate": calib["slope_curve"],
                      "switch": calib["switch_curve"]}
        terms = {}
        for name, curve in curves.items():
            difference = curve - references[name]
            terms[name] = float(np.nanmean(difference ** 2 if POINT_ERROR == "squared" else np.abs(difference)))
    return {"loss": float(np.mean(list(terms.values()))), "terms": terms, "curves": curves}


def curve_loss(success_matrix: np.ndarray, switch_matrix: np.ndarray, calib: dict) -> dict:
    """A previous objective (still used by the analysis scripts): the mean of
    two per-trial curve errors.

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


def _check_steps() -> None:
    """Fails early (instead of Optuna silently shrinking `high`) if STEPS names
    an unknown parameter or a range is not a whole number of steps."""
    for name, step in STEPS.items():
        if name not in BASELINE:
            raise ValueError(f"STEPS has an entry for '{name}', which is not in BASELINE.")
        if step <= 0:
            raise ValueError(f"STEPS['{name}'] must be positive (got {step}).")
        low, high = RANGES[name]
        n_steps = (high - low) / step
        if abs(n_steps - round(n_steps)) > 1e-6:
            raise ValueError(
                f"RANGES['{name}']=({low}, {high}) is {n_steps:.4f} steps of {step}, not a whole number; "
                f"adjust the range or STEPS['{name}'] so that (high - low) / step is an integer."
            )


# ---------------------------------------------------------------------------
# Stopping, restarting and locking (see the module docstring)
# ---------------------------------------------------------------------------

TrialState = optuna.trial.TrialState
_stop_requested = threading.Event()
_longest_trial_seconds = 0.0  # longest finished trial of this run, for the deadline check
_state_lock = threading.Lock()


def stop_requested() -> bool:
    return _stop_requested.is_set() or STOP_FILE.exists()


def _on_signal(signum, frame) -> None:
    """First SIGINT/SIGTERM: stop gracefully. The default behaviour is restored
    so that a second signal exits immediately."""
    if not _stop_requested.is_set():
        _stop_requested.set()
        print("[stop] signal received: running trials will finish and be saved, no new trial will start. "
              "Send the signal again to exit immediately.", flush=True)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)


def seconds_until_deadline() -> float | None:
    """Seconds left until HPO_DEADLINE, or None when it is not set/valid."""
    value = os.environ.get(DEADLINE_ENV_VAR, "").strip()
    if not value:
        return None
    try:
        return float(value) - time.time()
    except ValueError:
        return None


def stop_callback(study: optuna.Study, trial) -> None:
    """Runs after every finished trial, before the next one of that worker
    starts: once a stop was requested -- or the time left before HPO_DEADLINE is
    not enough for another trial of the length seen so far -- no new trial
    starts (the ones already running finish normally)."""
    global _longest_trial_seconds
    if trial.datetime_start is not None and trial.datetime_complete is not None:
        with _state_lock:
            _longest_trial_seconds = max(_longest_trial_seconds,
                                         (trial.datetime_complete - trial.datetime_start).total_seconds())
    if stop_requested():
        study.stop()
        return
    left = seconds_until_deadline()
    if left is not None and left < DEADLINE_SAFETY_FACTOR * _longest_trial_seconds:
        print(f"[deadline] {left / 3600:.1f} h left and the longest trial took "
              f"{_longest_trial_seconds / 3600:.1f} h: not starting more trials.", flush=True)
        study.stop()


def recover_stale_trials(study: optuna.Study) -> list:
    """Marks trials still RUNNING as FAILED. With the lock file held, no other
    instance of this script is running, so any RUNNING trial is left over from a
    hard stop."""
    stale = [t.number for t in study.trials if t.state == TrialState.RUNNING]
    for number in stale:
        study.tell(number, state=TrialState.FAIL)
    if stale:
        print(f"[recovery] trials {stale} were still marked RUNNING (a previous run was stopped hard); "
              f"marked as FAILED, the search will make up for them.")
    return stale


def warn_about_leftover_files() -> None:
    """Counts temporary simulation files left by killed runs. Not deleted: full
    runs may use the same folder and the same file names."""
    leftovers = list(SRC_DIR.glob("annarchy_sim*")) + list(SRC_DIR.glob("output_sim*")) \
        + list(SRC_DIR.glob("selection_sim*.npy")) + list(SRC_DIR.glob("failed_blocks_sim*.npy"))
    if leftovers:
        print(f"[warning] {len(leftovers)} simulation file(s)/folder(s) from earlier runs are in {SRC_DIR} "
              f"(annarchy_sim*, output_sim*, ...). They are not removed automatically; if no other run is using "
              f"this folder you can delete them (they count against your file quota).")


def _pid_alive(pid: int) -> bool:
    if os.name != "posix":
        return False  # no portable check; assume the lock is stale
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock() -> None:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().split()[0])
        except (ValueError, IndexError, OSError):
            pid = None
        if pid is not None and pid != os.getpid() and _pid_alive(pid):
            raise SystemExit(
                f"Another instance of this search seems to be running (PID {pid}, lock file {LOCK_FILE}). "
                f"Stop it first (touch {STOP_FILE.name}), or delete the lock file if you are sure it is not running."
            )
        print(f"[lock] replacing a stale lock file ({LOCK_FILE.name}).")
    LOCK_FILE.write_text(f"{os.getpid()}\n")


def release_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except OSError:
        pass


def objective(trial: optuna.Trial) -> float:
    params = {name: trial.suggest_float(name, *RANGES[name], step=STEPS.get(name)) for name in BASELINE}
    sim_ids = [trial.number * SIMS_PER_TRIAL + i + 1 for i in range(SIMS_PER_TRIAL)]

    all_success, all_switch = [], []
    try:
        # A RuntimeError here (a simulation process failed) is caught by
        # study.optimize(catch=...): the trial is recorded as FAILED, not as a
        # fake bad loss, and the search makes up for it.
        run_simulations(sim_ids, params, BLOCKS_PER_SIM)
        for sim_id in sim_ids:
            success_rows, switch_rows = blocks_from_simulation(SRC_DIR, sim_id, n_trials=N_TRIALS_PER_BLOCK)
            # Checked independently: success_rows/switch_rows can have different
            # lengths (a block can be NaN-padded/dropped in one metric and not
            # the other -- see blocks_from_simulation's docstring).
            if len(success_rows):
                all_success.append(success_rows)
            if len(switch_rows):
                all_switch.append(switch_rows)
    finally:
        for sim_id in sim_ids:
            cleanup_simulation_files(sim_id)  # also on failure; missing files are fine

    if not all_success or not all_switch:
        print(f"[trial {trial.number}] no usable blocks for these parameters "
              f"(success blocks={len(all_success)}, switch blocks={len(all_switch)}); penalizing.")
        return 1e6

    success_matrix = np.vstack(all_success)
    switch_matrix = np.vstack(all_switch)

    calib = load_naive_calibration_stats()
    result = pointwise_loss(success_matrix, switch_matrix, calib)
    tost = tost_loss(success_matrix, switch_matrix, calib)

    # What the objective is made of (one error per curve), the TOST view of the
    # same trial (offsets, their maximum, inside the margin?) and the raw means,
    # so a trial's attributes show where it stands on everything.
    for metric, term in result["terms"].items():
        trial.set_user_attr(f"loss_{metric}", term)
    for metric, offset in tost["offsets"].items():
        trial.set_user_attr(f"offset_{metric}", offset)
    trial.set_user_attr("loss_tost", tost["loss"])
    trial.set_user_attr("worst_metric", tost["worst_metric"])
    trial.set_user_attr("within_margin", bool(tost["within_margin"]))
    trial.set_user_attr("success_mean", float(per_block_success_rate(success_matrix).mean()))
    trial.set_user_attr("learning_rate_mean", float(per_block_learning_rate(success_matrix).mean()))
    trial.set_user_attr("switch_mean", float(per_block_switch_probability(switch_matrix).mean()))
    trial.set_user_attr("success_trial1", float(result["curves"]["success"][0]))
    trial.set_user_attr("success_trial15", float(result["curves"]["success"][-1]))
    trial.set_user_attr("n_blocks", int(len(success_matrix)))
    return result["loss"]


def run_search() -> None:
    """Creates/loads the study, recovers trials left running by a hard stop and
    runs the trials still missing to reach OPTUNA_TRIALS completed ones."""
    storage = f"sqlite:///{SRC_DIR / 'hpo_study.db'}"
    # Open the study once with the default sampler just to look at its history.
    study = optuna.create_study(study_name=STUDY_NAME, storage=storage, direction="minimize", load_if_exists=True)
    recover_stale_trials(study)
    n_complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
    remaining = max(0, OPTUNA_TRIALS - n_complete)

    if remaining == 0:
        print(f"The study already has {n_complete} completed trial(s) (OPTUNA_TRIALS = {OPTUNA_TRIALS}); "
              f"nothing to run. Raise OPTUNA_TRIALS to search further.")
    else:
        # The sampler is seeded with the number of trials already in the study: a
        # restart with the same seed would replay the same random initial
        # parameters and repeat trials that are already done.
        study = optuna.load_study(
            study_name=STUDY_NAME, storage=storage,
            sampler=TPESampler(seed=CALIBRATION_SEED + len(study.trials), multivariate=True),
        )
        print(f"{n_complete} of {OPTUNA_TRIALS} trials completed; running the {remaining} missing one(s), "
              f"{N_PARALLEL_OPTUNA_TRIALS} at a time, {SIMS_PER_TRIAL} simulations x {BLOCKS_PER_SIM} blocks each.")
        study.optimize(objective, n_trials=remaining, n_jobs=N_PARALLEL_OPTUNA_TRIALS,
                       catch=(RuntimeError,), callbacks=[stop_callback])
        n_complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
        if stop_requested() or n_complete < OPTUNA_TRIALS:
            print(f"[stop] {n_complete} of {OPTUNA_TRIALS} trials completed; run the script again to continue.")

    if n_complete == 0:
        print("No completed trial yet; nothing to report.")
        return

    print("\nBest parameters found:")
    for name, value in study.best_params.items():
        print(f"  {name}: {value:.6g}  (baseline: {BASELINE[name]:.6g})")
    best_attrs = study.best_trial.user_attrs
    print(f"Best objective (mean {POINT_ERROR} per-point difference of the 3 curves, probability units): "
          f"{study.best_value:.5f}")
    print("  TOST offsets of the best trial (NHP-SD, margin 0.2): " + ", ".join(
        f"{m} {best_attrs[f'offset_{m}']:+.2f}" for m in TOST_METRICS if f"offset_{m}" in best_attrs))

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


def main() -> None:
    global _longest_trial_seconds
    if POINT_ERROR not in ("squared", "absolute"):
        raise ValueError(f"POINT_ERROR must be 'squared' or 'absolute' (is {POINT_ERROR!r}).")
    _check_steps()
    acquire_lock()
    try:
        _stop_requested.clear()  # a fresh start never inherits an earlier stop request
        _longest_trial_seconds = 0.0
        if STOP_FILE.exists():
            STOP_FILE.unlink()
            print(f"[stop] removed a leftover {STOP_FILE.name} from an earlier run.")
        install_signal_handlers()
        warn_about_leftover_files()
        load_naive_calibration_stats()  # compute/cache the split once, up front
        run_search()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
