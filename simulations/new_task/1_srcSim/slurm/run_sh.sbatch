#!/bin/bash
# =============================================================================
# A full run through run.sh as ONE SLURM job on one node (the alternative to
# full_run.sbatch, which uses a job array with one task per simulation).
#
# run.sh runs the simulations in batches ("Durchgaenge"): PARALLEL simulations
# at the same time, then the next batch, DURCHGAENGE batches in total, so
#     total simulations = PARALLEL * DURCHGAENGE   (ids 1 .. PARALLEL*DURCHGAENGE)
#
# PARALLEL is taken from --cpus-per-task below (one CPU per simulation, since
# parallel.py is single-threaded), so to change how many simulations run at once
# you only change --cpus-per-task. DURCHGAENGE is set below.
# Examples:
#     30 simulations at once, 1 batch:   --cpus-per-task=30, DURCHGAENGE=1  -> 30 sims
#     10 simulations at once, 3 batches: --cpus-per-task=10, DURCHGAENGE=3  -> 30 sims
# Fewer CPUs per batch start sooner on a busy cluster, but the run takes about
# DURCHGAENGE times longer. A batch node has 128 CPUs.
#
# Both can also be given at submission without editing this file:
#     sbatch --cpus-per-task=10 --export=ALL,DURCHGAENGE=3 slurm/run_sh.sbatch
#
# The MODEL parameters (num_blocks, csd2_*, sc_noise_*) are set in run.sh, as before.
#
# Submit from the folder that contains run.sh and parallel.py:
#     cd simulations/new_task/1_srcSim
#     sbatch slurm/run_sh.sbatch
# Follow it with:  squeue -u $USER   |   tail -f run_sh_<jobid>.log
#
# CAUTION: run.sh deletes every *annarchy* folder in this directory after each
# batch, and writes output_sim1.. like hpo_search.py's trials do. Do not run this
# while an Optuna search (hpo_search.sbatch) is running in the same folder, and
# move the results of a previous run into stashed_runs/ first.
# =============================================================================
#SBATCH --job-name=run_sh
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30           # = PARALLEL, simulations per batch
#SBATCH --mem=64G                    # ADJUST after a first run (sacct ... MaxRSS); ~2 GB per simulation as a start
#SBATCH --time=2-00:00:00            # ADJUST: must cover ALL batches (DURCHGAENGE x one simulation's duration)
#SBATCH --output=run_sh_%j.log

set -eo pipefail

DURCHGAENGE="${DURCHGAENGE:-1}"                # number of batches, run one after the other
PARALLEL="${PARALLEL:-${SLURM_CPUS_PER_TASK}}" # simulations per batch (default: the CPUs requested)

cd "${SLURM_SUBMIT_DIR}"

if [ "${PARALLEL}" -gt "${SLURM_CPUS_PER_TASK}" ]; then
    echo "WARNING: PARALLEL=${PARALLEL} > --cpus-per-task=${SLURM_CPUS_PER_TASK}: simulations will share CPUs and run slower."
fi

# ADJUST if needed: where conda lives and the environment's name (conda info --base).
CONDA_BASE="${HOME}/miniconda3"
CONDA_ENV="thesisEnv"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ADJUST if the cluster provides the compiler as a module:
# module load gcc

export OMP_NUM_THREADS=1
export PARALLEL DURCHGAENGE

echo "job ${SLURM_JOB_ID} on $(hostname): ${PARALLEL} simulations per batch x ${DURCHGAENGE} batch(es)" \
     "= $((PARALLEL * DURCHGAENGE)) simulations, started $(date)"
grep -E '^(num_blocks|csd2_|sc_noise_)' run.sh   # the model parameters used, for the log

bash run.sh

echo "finished $(date); batch durations in fertig_zeit.txt"
