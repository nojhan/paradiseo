#!/bin/bash

algos=(
"--crossover-rate=0 --cross-selector=0 --crossover=0 --mutation-rate=0 --mut-selector=0 --mutation=0 --replacement=0"
"--crossover-rate=1 --cross-selector=0 --crossover=0 --mutation-rate=0 --mut-selector=0 --mutation=0 --replacement=0"
"--crossover-rate=2 --cross-selector=0 --crossover=0 --mutation-rate=0 --mut-selector=0 --mutation=0 --replacement=0"
)


i=1 # Loop counter.
for algo in "${algos[@]}" ; do
    echo "${algo}"

    name="$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
    ./run_algo.sh ${algo} &> "expe_${name}.log"

    perc=$(echo "scale=2;${i}/${#algos[@]}*100" | bc)
    echo -e "${perc}%\n"
    i=$((i+1))
done

echo "Done"
