"""
Optuna search over a single parameter: the strength of the SC (Thalamus) noise,
parallel.py's --sc-noise-scale (1 = the baseline noise; a scale k keeps the
average noise drive and makes it k times more variable, see parallel.py).

Everything else is fixed at the best Cortex->SD2 parameters of the previous
search (study "new_task_hpo_csd2_maxw_curveloss", best trial 5, objective
0.0200): see FIXED_PARAMS. The point is not to re-tune the model but to find
how much decision noise brings the plateau (last-trial success) and the first
trial of a block to the NHP values, which the previous search could not do:
the model made no mistakes after learning (last-trial success 1.00 vs 0.915 for
the NHP), which is what the learning-rate TOST reads.

This file has its own budget settings (OPTUNA_TRIALS, N_PARALLEL_OPTUNA_TRIALS,
SIMS_PER_TRIAL, BLOCKS_PER_SIM below), independent of hpo_search.py's; it only
reuses that file's simulation plumbing and the calibration data.

Search space and trial scheduling
---------------------------------
sc_noise_scale takes the 16 integer values 1..16 (SC_NOISE_VALUES) and every
value is evaluated once: the scan is complete after 16 trials, so a larger
OPTUNA_TRIALS just means "run every value that is still missing". Optuna's
GridSampler is deliberately NOT used: its handling of failed/interrupted trials
differs between Optuna versions (the cluster pins 2.10.1, see requirements.txt)
and it does not re-run the points of failed trials. Instead QueueSampler hands
out the values that have no completed trial yet, in a fixed pseudo-random order
(so an early stop leaves a spread of values, not just the smallest ones), and
Optuna only does the bookkeeping (storage, trial history, CSV export).

Objective (minimized)
---------------------
A weighted mean of three terms, all in probability-squared units so that they
are comparable:

    loss = (W_SUCCESS * L_success + W_SWITCH * L_switch + W_LR * L_lr)
           / (W_SUCCESS + W_SWITCH + W_LR)

    L_success, L_switch : hpo_search.curve_loss(): mean over trial positions of
        the squared difference between the model's and the NHP calibration's
        per-trial mean success / switch probability.
    L_lr : ((learning_rate_mean_model - learning_rate_mean_NHP) * (N - 1))^2,
        N = trials per block. The block-mean learning rate equals
        (p_last - p_first) / (N - 1), so multiplying by N - 1 turns its offset
        into a difference of (p_last - p_first) in probability units. This is
        the quantity the learning-rate TOST compares, and it depends only on
        the first and last trial, where the per-trial curve losses barely look.

W_LR is larger than the other two weights (2 vs 1): the curve terms average
over all 15 trials, dominated by the learning transient, while L_lr is one
number that pins the plateau, the thing this search is about.

Each trial's user attributes record the three terms, the block means of the
three TOST metrics, success at trial 1 and trial 15, and the mean switch
probability over the last 5 trials (the plateau), so the noise scan can be read
directly from the exported CSV.

Stopping it, and what happens to the database
---------------------------------------------
Preferred (graceful): no new trial starts, the ones already running finish and
are saved, nothing needs fixing afterwards. Either
    touch STOP_SC_NOISE_SEARCH            (a file next to this script; deleted
                                           again at the next start), or
    pkill -TERM -f hpo_search_sc_noise.py (or Ctrl-C, if it runs in the
                                           foreground)
A second SIGTERM/SIGINT exits immediately. The orchestrator is matched by name,
so `pkill -f hpo_search_sc_noise.py` does not touch the parallel.py simulations.
NOTE: Ctrl-C in a terminal is also delivered to the simulations, which then die
and their trials are recorded as failed (see below); use the stop file or pkill
-TERM when you want the running trials to finish.

Hard stop (kill -9, node reboot, simulations killed with pkill -f parallel.py):
trials that were running stay marked RUNNING in the database and the simulations'
temporary files stay in this folder. Nothing has to be repaired by hand: at the
next start the script marks those trials as FAILED and, since only values with a
COMPLETED trial count as done, evaluates their noise scales again. A simulation
that fails while the script keeps running is handled the same way (the trial is
recorded as FAILED, not as a fake bad loss, and is retried at the next start).
Leftover annarchy_sim*/output_sim* files in this folder are harmless (a retry
gets new simulation ids) and can be deleted.

A lock file (hpo_sc_noise.lock, containing the PID) prevents two instances from
running at once, which would make the recovery above fail running trials of the
other instance. A lock left by a dead process is detected and replaced (on
non-POSIX systems the liveness check is skipped).

Usage
-----
    cd simulations/new_task/1_srcSim
    nohup python hpo_search_sc_noise.py > hpo_sc_noise_log.txt 2>&1 &
"""

from __future__ import annotations

import os
import random
import signal
import threading

import numpy as np
import optuna

import hpo_search as base

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Fixed at the best parameters of the previous search; passed to every
# simulation as parallel.py flags (flag name = "--" + key with "_" -> "-").
FIXED_PARAMS = {
    "csd2_factor": -2.8,
    "csd2_delta": 2.2e-06,
    "csd2_max_weight": 0.00155,
}
SC_NOISE_VALUES = list(range(1, 17))  # sc_noise_scale values: 1, 2, ..., 16
STUDY_NAME = "new_task_hpo_sc_noise_scale"

# Weights of the three loss terms (see the module docstring).
W_SUCCESS = 1.0
W_SWITCH = 1.0
W_LR = 2.0

# Budget -- independent of hpo_search.py's own settings.
OPTUNA_TRIALS = 100            # max trials started per invocation; the scan has len(SC_NOISE_VALUES) in total,
                               # so anything >= 16 means "all that are still missing"
N_PARALLEL_OPTUNA_TRIALS = 5   # trials running at the same time (each launches SIMS_PER_TRIAL simulations)
SIMS_PER_TRIAL = 20            # simulations per trial (per noise scale)
BLOCKS_PER_SIM = 45            # blocks per simulation

ORDER_SEED = 20241028          # fixed pseudo-random order in which the missing values are evaluated
PLATEAU_TRIALS = 5             # last trial positions averaged for the switch-plateau attribute
STOP_FILE = base.SRC_DIR / "STOP_SC_NOISE_SEARCH"
LOCK_FILE = base.SRC_DIR / "hpo_sc_noise.lock"

TrialState = optuna.trial.TrialState


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def learning_rate_term(learning_rate_mean: float, calib: dict) -> float:
    """Squared offset of the block-mean learning rate, rescaled by N - 1 into
    probability units (a difference of p_last - p_first)."""
    return float(((learning_rate_mean - calib["learning_rate_mean"]) * (base.N_TRIALS_PER_BLOCK - 1)) ** 2)


def combined_loss(success_matrix: np.ndarray, switch_matrix: np.ndarray, calib: dict) -> dict:
    """The objective of this search plus its parts: the curve losses of
    hpo_search.curve_loss() and the learning-rate term, weighted by W_*.
    Returns {"loss", "loss_success", "loss_switch", "loss_lr",
    "learning_rate_mean", "success_curve", "switch_curve"}."""
    curves = base.curve_loss(success_matrix, switch_matrix, calib)
    learning_rate_mean = float(base.per_block_learning_rate(success_matrix).mean())
    loss_lr = learning_rate_term(learning_rate_mean, calib)
    total_weight = W_SUCCESS + W_SWITCH + W_LR
    loss = (W_SUCCESS * curves["loss_success"] + W_SWITCH * curves["loss_switch"] + W_LR * loss_lr) / total_weight
    return {
        "loss": loss,
        "loss_success": curves["loss_success"],
        "loss_switch": curves["loss_switch"],
        "loss_lr": loss_lr,
        "learning_rate_mean": learning_rate_mean,
        "success_curve": curves["success_curve"],
        "switch_curve": curves["switch_curve"],
    }


# ---------------------------------------------------------------------------
# Scheduling: hand out the missing noise scales
# ---------------------------------------------------------------------------

class QueueSampler(optuna.samplers.BaseSampler):
    """Returns, for each sampled trial, the next value of an in-process list
    (thread-safe, since trials run in parallel threads). It ignores the
    distribution it is asked to sample from: the list holds the noise scales
    to evaluate."""

    def __init__(self, values=()):
        self._values = list(values)
        self._lock = threading.Lock()

    def set_values(self, values) -> None:
        with self._lock:
            self._values = list(values)

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        with self._lock:
            if not self._values:
                raise RuntimeError("QueueSampler has no noise scale left to hand out.")
            return self._values.pop(0)


def missing_values(study: optuna.Study) -> list:
    """Noise scales without a completed trial (failed/running ones do not count)."""
    done = {t.params["sc_noise_scale"] for t in study.trials
            if t.state == TrialState.COMPLETE and "sc_noise_scale" in t.params}
    return [v for v in SC_NOISE_VALUES if v not in done]


# ---------------------------------------------------------------------------
# Stopping, recovery and locking
# ---------------------------------------------------------------------------

_stop_requested = threading.Event()


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


def stop_callback(study: optuna.Study, trial) -> None:
    """Runs after every finished trial: once a stop was requested, no new
    trial starts (the ones already running finish normally)."""
    if stop_requested():
        study.stop()


def recover_stale_trials(study: optuna.Study) -> list:
    """Marks trials still RUNNING as FAILED. With the lock file held, no other
    instance of this script is running, so any RUNNING trial is left over from a
    hard stop. Their noise scales have no completed trial, so missing_values()
    schedules them again."""
    stale = [t.number for t in study.trials if t.state == TrialState.RUNNING]
    for number in stale:
        study.tell(number, state=TrialState.FAIL)
    if stale:
        print(f"[recovery] trials {stale} were still marked RUNNING (a previous run was stopped hard); "
              f"marked as FAILED, their noise scales will be evaluated again.")
    return stale


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


# ---------------------------------------------------------------------------
# Objective and search
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    sc_noise_scale = trial.suggest_int("sc_noise_scale", SC_NOISE_VALUES[0], SC_NOISE_VALUES[-1])
    params = dict(FIXED_PARAMS)
    params["sc_noise_scale"] = sc_noise_scale
    sim_ids = [trial.number * SIMS_PER_TRIAL + i + 1 for i in range(SIMS_PER_TRIAL)]

    all_success, all_switch = [], []
    try:
        # A RuntimeError here (a simulation failed) is caught by study.optimize(catch=...):
        # the trial is recorded as FAILED and its noise scale is retried at the next start.
        base.run_simulations(sim_ids, params, BLOCKS_PER_SIM)
        for sim_id in sim_ids:
            success_rows, switch_rows = base.blocks_from_simulation(base.SRC_DIR, sim_id, n_trials=base.N_TRIALS_PER_BLOCK)
            # Checked independently: success_rows/switch_rows can have different
            # lengths (see blocks_from_simulation's docstring).
            if len(success_rows):
                all_success.append(success_rows)
            if len(switch_rows):
                all_switch.append(switch_rows)
    finally:
        for sim_id in sim_ids:
            base.cleanup_simulation_files(sim_id)  # also on failure; missing files are fine

    if not all_success or not all_switch:
        raise RuntimeError(f"no usable blocks for sc_noise_scale={sc_noise_scale}")

    success_matrix = np.vstack(all_success)
    switch_matrix = np.vstack(all_switch)
    result = combined_loss(success_matrix, switch_matrix, base.load_naive_calibration_stats())

    trial.set_user_attr("loss_success_curve", result["loss_success"])
    trial.set_user_attr("loss_switch_curve", result["loss_switch"])
    trial.set_user_attr("loss_learning_rate", result["loss_lr"])
    trial.set_user_attr("success_mean", float(base.per_block_success_rate(success_matrix).mean()))
    trial.set_user_attr("learning_rate_mean", result["learning_rate_mean"])
    trial.set_user_attr("switch_mean", float(base.per_block_switch_probability(switch_matrix).mean()))
    trial.set_user_attr("success_trial1", float(result["success_curve"][0]))
    trial.set_user_attr("success_trial15", float(result["success_curve"][-1]))
    trial.set_user_attr("switch_plateau", float(np.nanmean(result["switch_curve"][-PLATEAU_TRIALS:])))
    trial.set_user_attr("n_blocks", int(len(success_matrix)))
    return result["loss"]


def run_search(study: optuna.Study, sampler: QueueSampler) -> list:
    """Evaluates the noise scales that have no completed trial yet (at most
    OPTUNA_TRIALS of them) and returns the list that was scheduled."""
    missing = missing_values(study)
    if not missing:
        print(f"All {len(SC_NOISE_VALUES)} noise scales already have a completed trial; nothing to run.")
        return []

    order = list(missing)
    random.Random(ORDER_SEED).shuffle(order)
    scheduled = order[:max(0, OPTUNA_TRIALS)]
    sampler.set_values(scheduled)
    print(f"Evaluating {len(scheduled)} of {len(missing)} missing noise scale(s), {N_PARALLEL_OPTUNA_TRIALS} at a time, "
          f"{SIMS_PER_TRIAL} simulations x {BLOCKS_PER_SIM} blocks each: {scheduled}")
    study.optimize(objective, n_trials=len(scheduled), n_jobs=N_PARALLEL_OPTUNA_TRIALS,
                   catch=(RuntimeError,), callbacks=[stop_callback])
    if stop_requested():
        print("[stop] stopped as requested; the saved trials are in the study, re-run the script to continue.")
    return scheduled


def print_scan(study: optuna.Study) -> None:
    """One row per noise scale, in scale order, next to the NHP reference."""
    calib = base.load_naive_calibration_stats()
    print("\nLoss and plateau diagnostics per noise scale "
          f"(NHP reference: success trial 1 = {calib['success_curve'][0]:.3f}, "
          f"trial 15 = {calib['success_curve'][-1]:.3f}, "
          f"switch plateau = {np.mean(calib['switch_curve'][-PLATEAU_TRIALS:]):.3f}, "
          f"learning rate = {calib['learning_rate_mean']:.4f}):")
    print(f"  {'scale':>5} {'loss':>9} {'L_succ':>9} {'L_switch':>9} {'L_lr':>9} "
          f"{'p(trial1)':>10} {'p(trial15)':>11} {'sw.plateau':>11} {'lr mean':>8} {'blocks':>7}")
    done = [t for t in study.trials if t.state == TrialState.COMPLETE and "success_trial15" in t.user_attrs]
    for t in sorted(done, key=lambda t: t.params["sc_noise_scale"]):
        a = t.user_attrs
        print(f"  {t.params['sc_noise_scale']:>5} {t.value:>9.5f} {a['loss_success_curve']:>9.5f} "
              f"{a['loss_switch_curve']:>9.5f} {a['loss_learning_rate']:>9.5f} {a['success_trial1']:>10.3f} "
              f"{a['success_trial15']:>11.3f} {a['switch_plateau']:>11.3f} {a['learning_rate_mean']:>8.4f} {a['n_blocks']:>7}")
    n_failed = sum(t.state == TrialState.FAIL for t in study.trials)
    still_missing = missing_values(study)
    if n_failed:
        print(f"  ({n_failed} failed/interrupted trial(s) in the study history; they are not in the table)")
    if still_missing:
        print(f"  Noise scales still without a completed trial: {still_missing}")


def main() -> None:
    acquire_lock()
    try:
        _stop_requested.clear()  # a fresh start never inherits an earlier stop request
        if STOP_FILE.exists():
            STOP_FILE.unlink()
            print(f"[stop] removed a leftover {STOP_FILE.name} from an earlier run.")
        install_signal_handlers()
        base.load_naive_calibration_stats()  # compute/refresh the cached calibration stats once, up front

        sampler = QueueSampler()
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=f"sqlite:///{base.SRC_DIR / 'hpo_study.db'}",
            sampler=sampler,
            direction="minimize",
            load_if_exists=True,
        )
        recover_stale_trials(study)
        run_search(study, sampler)

        print(f"\nFixed parameters: {FIXED_PARAMS}")
        if any(t.state == TrialState.COMPLETE for t in study.trials):
            print(f"Best sc_noise_scale: {study.best_params['sc_noise_scale']}  (objective {study.best_value:.5f})")
        print_scan(study)

        trials_csv_path = base.SRC_DIR / f"hpo_trials_{STUDY_NAME}.csv"
        study.trials_dataframe().to_csv(trials_csv_path, index=False)
        print(f"\nFull trial history saved to {trials_csv_path}")
        print(f"Optuna study '{STUDY_NAME}' saved to {base.SRC_DIR / 'hpo_study.db'} "
              f"(re-running continues it: only the noise scales without a completed trial are evaluated).")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
