startTime=$(date +%s)

# parallel    = simulations run at the same time in one batch ("Durchgang").
# durchgaenge = number of batches, run one after the other.
# Total simulations = parallel * durchgaenge, with ids 1 .. parallel*durchgaenge.
# Both can be set from outside (slurm/run_sh.sbatch does this), e.g.
#     PARALLEL=10 DURCHGAENGE=3 bash run.sh
# Without them the defaults below are used, as before.
parallel=${PARALLEL:-30}
durchgaenge=${DURCHGAENGE:-1}

# Block count per simulation, passed through to parallel.py's --num-blocks.
num_blocks=60

# Cortex->SD2 (indirect pathway) plasticity parameters, passed through to
# parallel.py's --csd2-factor/--csd2-delta/--csd2-max-weight. Defaults match
# parallel.py's own hardcoded baseline (-0.9 / 0.000005 / 0.0011); override
# here for a manual batch.
csd2_factor=-0.9
csd2_delta=0.000005
csd2_max_weight=0.0011

# SC (Thalamus) Poisson noise, passed through to parallel.py's --sc-noise-rate
# [Hz] and --sc-noise-weight [nS]. Defaults match parallel.py's baseline
# (100.0 / 0.005); these are the two parameters hpo_search.py tunes.
sc_noise_rate=100.0
sc_noise_weight=0.005

# Optional extra multiplier on top of rate and weight (parallel.py's
# --sc-noise-scale): rate / k, weight * k. Leave at 1.0 when setting rate and
# weight directly (e.g. from an Optuna result).
sc_noise_scale=1.0

for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python parallel.py $y --num-blocks $num_blocks \
                        --csd2-factor $csd2_factor --csd2-delta $csd2_delta \
                        --csd2-max-weight $csd2_max_weight \
                        --sc-noise-rate $sc_noise_rate --sc-noise-weight $sc_noise_weight \
                        --sc-noise-scale $sc_noise_scale &
        done
        wait
        sleep 5
	rm -r *annarchy*
	endTime=$(date +%s)
	dif=$((endTime - startdurchgangTime))
    echo "${durchgang} ${dif}" >> fertig_zeit.txt
	sleep 5
done

endTime=$(date +%s)
dif=$((endTime - startTime))
echo $dif >> fertig_zeit.txt

