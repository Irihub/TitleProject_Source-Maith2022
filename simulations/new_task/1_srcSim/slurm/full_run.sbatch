#!/bin/bash
# =============================================================================
# A full run (the SLURM version of run.sh): one simulation of parallel.py per
# array task, tasks 1..30 = simulation ids 1..30.   UNTESTED TEMPLATE: fill in the
# lines marked ADJUST for your cluster before submitting.
#
# A job ARRAY is the natural way to run independent simulations: SLURM schedules
# each task on whatever node has a free CPU, so no single node has to be large.
# No SLURM-specific code is needed in parallel.py: each task just runs
#     python parallel.py <sim_id> <flags>
# with sim_id = $SLURM_ARRAY_TASK_ID, exactly what run.sh does in its loop.
#
# Submit from the folder that contains parallel.py:
#     cd simulations/new_task/1_srcSim
#     sbatch slurm/full_run.sbatch
# Follow it with:  squeue -u $USER   |   cancel with:  scancel <jobid>
# Results appear next to parallel.py as output_sim<N>, selection_sim<N>.npy, ... like
# with run.sh; move them into a new folder under stashed_runs/ when all tasks end.
# =============================================================================
#SBATCH --job-name=full_run
#SBATCH --array=1-30                 # one task per simulation (run.sh: parallel=30)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # parallel.py is single-threaded (setup(num_threads=1))
#SBATCH --mem=64G                    # ADJUST: measure with sacct, see below
#SBATCH --time=1-00:00:00            # ADJUST: how long ONE simulation of NUM_BLOCKS blocks takes
#SBATCH --output=full_run_%A_%a.log  # %A = array job id, %a = task id
##SBATCH --partition=PARTITION       # ADJUST if the cluster needs it (remove one leading '#')
##SBATCH --account=ACCOUNT           # ADJUST if the cluster needs it (remove one leading '#')

set -eo pipefail

# ---- model parameters: same meaning as in run.sh --------------------------------
NUM_BLOCKS=60
CSD2_FACTOR=-0.9
CSD2_DELTA=0.000005
CSD2_MAX_WEIGHT=0.0011
SC_NOISE_RATE=100        # Hz
SC_NOISE_WEIGHT=0.005    # nS
SC_NOISE_SCALE=1.0       # optional multiplier on top of the two above (1.0 = none)
# ---------------------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR}"

# ADJUST: where conda lives and the environment's name.
CONDA_BASE="${HOME}/miniconda3"
CONDA_ENV="thesisEnv"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ADJUST if the cluster provides the compiler as a module:
# module load gcc

export OMP_NUM_THREADS=1

SIM_ID="${SLURM_ARRAY_TASK_ID}"
echo "simulation ${SIM_ID} on $(hostname), started $(date)"

python parallel.py "${SIM_ID}" --num-blocks "${NUM_BLOCKS}" \
    --csd2-factor "${CSD2_FACTOR}" --csd2-delta "${CSD2_DELTA}" --csd2-max-weight "${CSD2_MAX_WEIGHT}" \
    --sc-noise-rate "${SC_NOISE_RATE}" --sc-noise-weight "${SC_NOISE_WEIGHT}" --sc-noise-scale "${SC_NOISE_SCALE}"

# run.sh deletes the compiled network folders (*annarchy*) after the run; here each
# task removes its own.
rm -rf "annarchy_sim${SIM_ID}"
echo "simulation ${SIM_ID} finished $(date)"

# After a first run, check what a task needed and tune --mem and --time:
#     sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State
