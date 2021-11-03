#!/bin/bash
# Number of runs (=seeds).
runs=5
basename=$1
mevals=$2
nbAlgo=2
echo "Start JOB maxEv= $mevals $(date -Iseconds) ----------------------"
. /etc/profile.d/modules.sh
export MODULEPATH=${MODULEPATH}${MODULEPATH:+:}/opt/dev/Modules/Anaconda:/opt/dev/Modules/Compilers:/opt/dev/Modules/Frameworks:/opt/dev/Modules/Libraries:/opt/dev/Modules/Tools:/opt/dev/Modules/IDEs:/opt/dev/Modules/MPI
module load LLVM/clang-llvm-10.0
cp ${HOME}/code/paradiseo/eo/contrib/irace/release/fastga .
# You most probably want to run on release builds.
exe="./fastga"

#outdir="/scratchbeta/$USER/$(date --iso-8601=minutes)_results_randoms"
outdir="${basename}/maxEv=${mevals}_nbAlgo=${nbAlgo}_$(date --iso-8601=minutes)_results_randoms"
mkdir -p ${outdir}
n=1
algoid=0
for algoid in $(seq ${nbAlgo}); do
    #date
    r1=$(echo "scale=2 ; ${RANDOM}/32767" | bc)
    r2=$(echo "scale=2 ; ${RANDOM}/32767" | bc)
    a=(${r1} $((RANDOM%7)) $((RANDOM%10)) ${r2} $((RANDOM%7)) $((RANDOM%11)) $((RANDOM%11)) $((RANDOM%50 +1)) $((RANDOM%50 +1)) )
    #condition for value of replacement, pop-size and offspringsize
    while [[ (1 -lt ${a[6]} && ${a[7]} -lt ${a[8]}) || ( ${a[6]} -eq 1  && ${a[7]} -ne ${a[8]}) ]]
    do
        #echo "get in ------------------replacement ${a[6]} popsize  ${a[7]} offspringsize ${a[8]}"
        r1=$(echo "scale=2 ; ${RANDOM}/32767" | bc)
        r2=$(echo "scale=2 ; ${RANDOM}/32767" | bc)
        a=(${r1} $((RANDOM%7)) $((RANDOM%10)) ${r2} $((RANDOM%7)) $((RANDOM%11)) $((RANDOM%11)) $((RANDOM%50 +1)) $((RANDOM%50 +1)))
    done 
    algo="--crossover-rate=${a[0]} --cross-selector=${a[1]} --crossover=${a[2]} --mutation-rate=${a[3]} --mut-selector=${a[4]} --mutation=${a[5]} --replacement=${a[6]} --pop-size=${a[7]} --offspring-size=${a[8]}"
    echo " start algo ${a}------ $(date --iso-8601=minutes)"
    algodir="$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
    for pb in $(seq 0 18) ; do
        perc=$(echo "scale=3;${n}/(10*18)*10.0" | bc)
        #echo "${perc}% : algo ${algoid}/100, problem ${pb}/18 $(date --iso-8601=minutes)"
        # echo -n "Runs: "
        name_dir="pb=${pb}_$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
        
        mkdir -p ${outdir}/${algodir}/data/${name_dir}
        mkdir -p ${outdir}/${algodir}/logs/${name_dir}
        
        for seed in $(seq ${runs}) ; do # Iterates over runs/seeds.
            # This is the command to be ran.
            cmd="${exe} --problem=${pb} --seed=${seed} --instance=${seed} ${algo} --max-evals=${mevals}"
            name_run="pb=${pb}_seed=${seed}_$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
            # echo $name_run
            #echo $algo
            
            ${cmd} > ${outdir}/${algodir}/data/${name_dir}/${name_run}.dat 2> ${outdir}/${algodir}/logs/${name_dir}/${name_run}.log
            # Check for the most common problem in the log file.
            #cat "${outdir}/raw/logs/${name_run}.log" | grep "illogical performance"
        done # seed
        # echo ""

        n=$(($n+1))
    done # pb
    echo "end algo $(date -Iseconds) "
    algoid=$(($algoid+1))
done



echo "------------------------------------Done $mevals $(date -Iseconds) "

