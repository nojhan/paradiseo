#!/bin/bash

date -Iseconds
echo "STARTS"
myhome=$1
scratchpath=$2
#dir=${HOME}/plan2/${name}
mexp=$3 #budget irace
mevals=$4 #budget fastga
name="dataF_maxExp=${mexp}_maxEv=${mevals}_$(date --iso-8601=seconds)"
dir=${scratchpath}/dataFAR/dataF/${name}
mkdir -p ${dir}

for r in $(seq 2); do
    echo "Run $r/15";
    #date -Iseconds 
    #cmd="qsub -N irace_${runs}_${buckets}" -q beta -l select=1:ncpus=1 -l walltime=00:04:00 --${HOME}/run_irace.sh ${dir}
    cmd="qsub -N iraceF_${mevals}_run=${r} -q beta -l select=1:ncpus=1 -l walltime=00:30:00 -- ${scratchpath}/planF/r_iF.sh ${dir} ${r} ${mexp} ${mevals} ${myhome}"
    #time -p bash ${HOME}/plan2/run_irace2.sh ${dir} ${r} &> ${dir}/erreur_${r}.txt
    #bash ${HOME}/test/r_i.sh
    echo $cmd
    $cmd
    #date -Iseconds
done

#echo "DONE"
#date -Iseconds
#echo $(pwd)
