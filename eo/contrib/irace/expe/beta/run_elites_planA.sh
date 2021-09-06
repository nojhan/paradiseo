#!/bin/bash
#instance = seed
echo "-----------------Start $(date)"
. /etc/profile.d/modules.sh
export MODULEPATH=${MODULEPATH}${MODULEPATH:+:}/opt/dev/Modules/Anaconda:/opt/dev/Modules/Compilers:/opt/dev/Modules/Frameworks:/opt/dev/Modules/Libraries:/opt/dev/Modules/Tools:/opt/dev/Modules/IDEs:/opt/dev/Modules/MPI
module load LLVM/clang-llvm-10.0

csv_file=$1 #contains all the configs of all the problems of one experiments
mexp=$2
mevals=$3
path=$4
# Number of runs (=seeds).
runs=50

# You most probably want to run on release builds.
exe="/home/zhenga/fastga"
plan=$(echo ${csv_file} | sed "s/results_irace_plan//")
outdir="${path}/plan4_maxExp=${mexp}_maxEv=${mevals}_$(date --iso-8601=minutes)_results_elites_all"
mkdir -p ${outdir}
mkdir -p ${outdir}/raw
mkdir -p ${outdir}/raw/data
mkdir -p ${outdir}/raw/logs

n=0
algoid=0
for line in $(cat ${csv_file}|  sed 1,1d | cut -s -d"," -f3-11 ); do
    a=($(echo $line | sed "s/,/ /g"))
    algo="--crossover-rate=${a[0]} --cross-selector=${a[1]} --crossover=${a[2]} --mutation-rate=${a[3]} --mut-selector=${a[4]} --mutation=${a[5]} --replacement=${a[6]} --pop-size=${a[7]} --offspring-size=${a[8]}"
   
    #perc=$(echo "scale=3;${n}/(285)*100.0" | bc)
    #echo "${perc}% : algo ${algoid}/285"
    # echo -n "Runs: "
    for pb in $(seq 0 18) ; do
        name_dir="pb=${pb}_$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
        mkdir -p ${outdir}/raw/data/${name_dir}
        mkdir -p ${outdir}/raw/logs/${name_dir}
        for seed in $(seq ${runs}) ; do # Iterates over runs/seeds.
            # This is the command to be ran.
            #cmd="${exe} --full-log=1 --problem=${pb} --seed=${seed} ${algo}"
            cmd="${exe} --problem=${pb} --seed=${seed} --instance=${seed} ${algo}"
            #echo ${cmd} # Print the command.
            # Forge a directory/log file name
            # (remove double dashs and replace spaces with underscore).
            name_run="pb=${pb}_seed=${seed}_$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"
            # echo $name_run
            # Actually start the command.
        
            ${cmd} > "${outdir}/raw/data/${name_dir}/${name_run}.dat" 2> "${outdir}/raw/logs/${name_dir}/${name_run}.log"
            # Check for the most common problem in the log file.
            #cat "${outdir}/raw/logs/${name_run}.log" | grep "illogical performance"
        done # seed
        n=$(($n+1))
     done 
    algoid=$(($algoid+1))
done

# Move IOH logs in the results directory.
#mv ./FastGA_* ${outdir}

echo "Done $(date) -----------------------"
#date
