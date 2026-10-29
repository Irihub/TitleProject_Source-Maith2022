startTime=$(date +%s)

let parallel=30
let durchgaenge=1

# Block count per simulation, passed through to parallel.py's --num-blocks.
num_blocks=60

# Cortex->SD2 (indirect pathway) plasticity parameters, passed through to
# parallel.py's --csd2-factor/--csd2-delta/--csd2-max-weight. Defaults match
# parallel.py's own hardcoded baseline (-0.9 / 0.000005 / 0.0011); override
# here for a manual batch.
csd2_factor=-0.9
csd2_delta=0.000005
csd2_max_weight=0.0011

# SC (Thalamus) noise strength, passed through to parallel.py's --sc-noise-scale.
# 1.0 = baseline (unchanged); k > 1 keeps the average noise drive but makes it k
# times more variable (rate / k, weight * k).
sc_noise_scale=1.0

for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python parallel.py $y --num-blocks $num_blocks \
                        --csd2-factor $csd2_factor --csd2-delta $csd2_delta \
                        --csd2-max-weight $csd2_max_weight \
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

