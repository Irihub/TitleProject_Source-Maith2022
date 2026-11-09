#!/bin/bash
# =============================================================================
# Optuna search (hpo_search.py) as ONE SLURM job.   UNTESTED TEMPLATE: fill in the
# lines marked ADJUST for your cluster before submitting.
#
# How it works: hpo_search.py is the ORCHESTRATOR. For every trial it starts
# SIMS_PER_TRIAL copies of parallel.py (one single-threaded process per
# simulation, via subprocess), N_PARALLEL_OPTUNA_TRIALS trials at a time, all on
# the machine where it runs. So the job needs ONE CPU PER SIMULATION running at
# the same time, on ONE node (subprocess cannot reach other nodes):
#
#     CPUs = N_PARALLEL_OPTUNA_TRIALS * SIMS_PER_TRIAL + 1 (the orchestrator)
#
# With the constants now in hpo_search.py (5 trials in parallel, 20 simulations
# each) that is 5 * 20 + 1 = 101. KEEP --cpus-per-task IN SYNC with those
# constants. If no node has that many cores, lower N_PARALLEL_OPTUNA_TRIALS and/or
# SIMS_PER_TRIAL in hpo_search.py (fewer simulations per trial makes the
# objective noisier) and lower --cpus-per-task to match.
#
# No SLURM-specific code is needed in hpo_search.py or parallel.py: this file only
# reserves the resources and starts the same command you ran on the old server.
#
# Submit from the folder that contains hpo_search.py:
#     cd simulations/new_task/1_srcSim
#     sbatch slurm/hpo_search.sbatch
# Follow it with:  squeue -u $USER   |   tail -f hpo_search_<jobid>.log
# Cancel with:     scancel <jobid>
#
# Time limits and restarting: one trial took roughly 5-14 hours on the old server (20
# trials ~ 26 h with 5 in parallel), so the whole search can exceed the partition's
# time limit. That is fine:
#   - OPTUNA_TRIALS in hpo_search.py is the TOTAL number of completed trials wanted.
#     Submitting this script again continues the same study (hpo_study.db) and runs
#     only the trials still missing.
#   - This script gives hpo_search.py the job's end time (HPO_DEADLINE, below), and
#     it stops STARTING trials that would not finish before then (it compares the
#     time left with 1.2 x the longest trial so far), so the job ends cleanly with
#     every finished trial saved instead of being killed in the middle of one.
#   - To stop earlier by hand without losing running trials:
#         touch STOP_HPO_SEARCH        (in the folder with hpo_search.py)
#     Running trials finish and are saved, no new one starts. Do not use scancel
#     for this: it kills the simulations and those trials are lost (recorded as
#     failed and made up for at the next start, but their compute time is gone).
# If a job is killed anyway (time limit, node failure), the next submission marks the
# interrupted trials as failed and makes up for them; nothing needs fixing by hand.
# =============================================================================
#SBATCH --job-name=hpo_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=101          # ADJUST: N_PARALLEL_OPTUNA_TRIALS * SIMS_PER_TRIAL + 1
#SBATCH --mem=64G                    # ADJUST: measure with a small test, see below
#SBATCH --time=2-00:00:00            # ADJUST: at most the partition's limit (days-hh:mm:ss)
#SBATCH --output=hpo_search_%j.log
##SBATCH --partition=PARTITION       # ADJUST if the cluster needs it (remove one leading '#')
##SBATCH --account=ACCOUNT           # ADJUST if the cluster needs it (remove one leading '#')

set -eo pipefail

# Run from the folder the job was submitted from (must be .../new_task/1_srcSim).
cd "${SLURM_SUBMIT_DIR}"

# ADJUST: where conda lives and the environment's name.
CONDA_BASE="${HOME}/miniconda3"
CONDA_ENV="thesisEnv"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ADJUST if the cluster provides the compiler as a module (ANNarchy compiles C++
# with OpenMP at the start of every simulation):
# module load gcc

export OMP_NUM_THREADS=1             # parallel.py already uses 1 thread per simulation

# Tell hpo_search.py when the job ends (Unix timestamp), so it stops starting trials
# that could not finish in time. If SLURM does not report an end time this is
# skipped, and the search simply runs until the limit (or until it is done).
END_TIME="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | sed -n 's/.*EndTime=\([^ ]*\).*/\1/p' | head -1 || true)"
if [ -n "${END_TIME}" ] && [ "${END_TIME}" != "Unknown" ]; then
    HPO_DEADLINE="$(date -d "${END_TIME}" +%s 2>/dev/null || true)"
    if [ -n "${HPO_DEADLINE}" ]; then
        export HPO_DEADLINE
        echo "job ends at ${END_TIME} (HPO_DEADLINE=${HPO_DEADLINE})"
    fi
fi

echo "job ${SLURM_JOB_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} CPUs, started $(date)"
g++ --version | head -1
python --version
python hpo_search.py
echo "finished $(date)"

# Memory and time: after a SMALL test (temporarily set OPTUNA_TRIALS = 1,
# SIMS_PER_TRIAL = 4, BLOCKS_PER_SIM = 3 in hpo_search.py) run
#     sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State
# and scale --mem and --time from it. Many simulations compile at the same time at
# the start of a trial, which is the memory peak.
