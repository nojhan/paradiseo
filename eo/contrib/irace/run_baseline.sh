#!/bin/bash

outdir="$(date --iso-8601=minutes)_results_baselines"
mkdir ${outdir}

algos=(
# (λ+λ)EA
"--full-log=1 --crossover-rate=0 --cross-selector=0 --crossover=0 --mutation-rate=0 --mut-selector=0 --mutation=1 --replacement=0"
# (λ+λ)fEA
"--full-log=1 --crossover-rate=0 --cross-selector=0 --crossover=0 --mutation-rate=0 --mut-selector=0 --mutation=5 --replacement=0"
# (λ+λ)xGA
"--full-log=1 --crossover-rate=2 --cross-selector=2 --crossover=2 --mutation-rate=2 --mut-selector=2 --mutation=1 --replacement=0"
# (λ+λ)1ptGA
"--full-log=1 --crossover-rate=2 --cross-selector=2 --crossover=5 --mutation-rate=2 --mut-selector=2 --mutation=1 --replacement=0"
)


i=1 # Loop counter.
for algo in "${algos[@]}" ; do
    echo "${algo}"

    name="$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
    ./run_algo.sh ${outdir} ${algo} &> "expe_${name}.log"

    perc=$(echo "scale=2;${i}/${#algos[@]}*100" | bc)
    echo -e "${perc}%\n"
    i=$((i+1))
done

echo "Done"
