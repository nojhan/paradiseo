#!/bin/bash

# Number of runs (=seeds).
runs=50

# You most probably want to run on release builds.
exe="./release/fastga"

outdir="$(date --iso-8601=minutes)_results_elites"
mkdir ${outdir}

# FIXME
algos=(
"--crossover-rate=1 --cross-selector=2 --crossover=1  --mutation-rate=2 --mut-selector=2 --mutation=8  --replacement=8"
"--crossover-rate=4 --cross-selector=5 --crossover=2  --mutation-rate=3 --mut-selector=4 --mutation=9  --replacement=9"
"--crossover-rate=1 --cross-selector=3 --crossover=8  --mutation-rate=2 --mut-selector=6 --mutation=3  --replacement=2"
"--crossover-rate=2 --cross-selector=1 --crossover=1  --mutation-rate=2 --mut-selector=6 --mutation=9  --replacement=0"
"--crossover-rate=4 --cross-selector=2 --crossover=2  --mutation-rate=4 --mut-selector=3 --mutation=7  --replacement=0"
"--crossover-rate=0 --cross-selector=3 --crossover=2  --mutation-rate=4 --mut-selector=2 --mutation=6  --replacement=0"
"--crossover-rate=3 --cross-selector=0 --crossover=3  --mutation-rate=4 --mut-selector=3 --mutation=10 --replacement=0"
"--crossover-rate=0 --cross-selector=1 --crossover=0  --mutation-rate=3 --mut-selector=2 --mutation=10 --replacement=0"
"--crossover-rate=2 --cross-selector=2 --crossover=2  --mutation-rate=4 --mut-selector=5 --mutation=10 --replacement=0"
"--crossover-rate=4 --cross-selector=2 --crossover=2  --mutation-rate=4 --mut-selector=5 --mutation=9  --replacement=0"
"--crossover-rate=3 --cross-selector=2 --crossover=10 --mutation-rate=4 --mut-selector=2 --mutation=10 --replacement=0"
"--crossover-rate=2 --cross-selector=2 --crossover=5  --mutation-rate=4 --mut-selector=3 --mutation=9  --replacement=0"
"--crossover-rate=3 --cross-selector=6 --crossover=2  --mutation-rate=4 --mut-selector=1 --mutation=10 --replacement=0"
"--crossover-rate=1 --cross-selector=5 --crossover=9  --mutation-rate=4 --mut-selector=2 --mutation=8  --replacement=0"
"--crossover-rate=2 --cross-selector=5 --crossover=2  --mutation-rate=4 --mut-selector=6 --mutation=8  --replacement=0"
"--crossover-rate=2 --cross-selector=2 --crossover=10 --mutation-rate=4 --mut-selector=6 --mutation=10 --replacement=0"
"--crossover-rate=3 --cross-selector=2 --crossover=2  --mutation-rate=4 --mut-selector=5 --mutation=10 --replacement=0"
"--crossover-rate=4 --cross-selector=2 --crossover=2  --mutation-rate=4 --mut-selector=1 --mutation=8  --replacement=0"
"--crossover-rate=4 --cross-selector=2 --crossover=2  --mutation-rate=4 --mut-selector=6 --mutation=9  --replacement=0"
)

pb=0 # Loop counter.
for algo in "${algos[@]}" ; do
    echo "Problem ${pb}"
    echo -n "Runs: "

    for seed in $(seq ${runs}) ; do # Iterates over runs/seeds.
        # This is the command to be ran.
        cmd="${exe} --full-log=1 --problem=${pb} --seed=${seed} ${algo}"
        # echo ${cmd} # Print the command.

        # Forge a directory/log file name
        # (remove double dashs and replace spaces with underscore).
        name_run="pb=${pb}_seed=${seed}_$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
        # echo $name_run

        # Progress print.
        echo -n "${seed} "

        # Actually start the command.
        ${cmd} > "${outdir}/${name_run}.dat" 2> "${outdir}/${name_run}.log"

        # Check for the most common problem in the log file.
        cat "${outdir}/${name_run}.log" | grep "illogical performance"
    done

    echo ""
    perc=$(echo "scale=2;${pb}/${#algos[@]}*100" | bc)
    echo -e "${perc}%\n"
    pb=$((pb+1))

done

echo "Done"
