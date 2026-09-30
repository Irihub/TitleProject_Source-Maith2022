startTime=$(date +%s)

let parallel=30
let durchgaenge=1

# Block count per simulation, passed through to parallel.py's --num-blocks.
num_blocks=60

for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python parallel.py $y --num-blocks $num_blocks &
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

