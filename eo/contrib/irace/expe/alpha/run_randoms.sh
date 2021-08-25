
#!/bin/bash

# Number of runs (=seeds).
runs=50

# You most probably want to run on release builds.
exe="./release/fastga"

outdir="$(date --iso-8601=minutes)_results_randoms"
mkdir -p ${outdir}
mkdir -p ${outdir}/raw
mkdir -p ${outdir}/raw/data
mkdir -p ${outdir}/raw/logs

n=1
algoid=0
for algoid in $(seq 0 100); do
    echo ""
    date

    a=( $((RANDOM%5)) $((RANDOM%7)) $((RANDOM%11)) $((RANDOM%5)) $((RANDOM%7)) $((RANDOM%11)) $((RANDOM%11)) )
    algo="--crossover-rate=${a[0]} --cross-selector=${a[1]} --crossover=${a[2]}  --mutation-rate=${a[3]} --mut-selector=${a[4]} --mutation=${a[5]}  --replacement=${a[6]}"

    for pb in $(seq 0 18) ; do
        perc=$(echo "scale=3;${n}/(100*18)*100.0" | bc)
        echo "${perc}% : algo ${algoid}/100, problem ${pb}/18"
        # echo -n "Runs: "

        for seed in $(seq ${runs}) ; do # Iterates over runs/seeds.
            # This is the command to be ran.
            cmd="${exe} --full-log=1 --problem=${pb} --seed=${seed} ${algo}"
            # echo ${cmd} # Print the command.

            # Forge a directory/log file name
            # (remove double dashs and replace spaces with underscore).
            name_run="pb=${pb}_seed=${seed}_$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
            # echo $name_run

            # Progress print.
            # echo -n "${seed} "

            # Actually start the command.
            ${cmd} > "${outdir}/raw/data/${name_run}.dat" 2> "${outdir}/raw/logs/${name_run}.log"

            # Check for the most common problem in the log file.
            cat "${outdir}/raw/logs/${name_run}.log" | grep "illogical performance"
        done # seed
        # echo ""

        n=$(($n+1))
    done # pb

    algoid=$(($algoid+1))
done

# Move IOH logs in the results directory.
mv ./FastGA_* ${outdir}

echo "Done"
date
