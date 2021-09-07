#!/bin/bash

date -Iseconds
echo "STARTS"
myhome=$1
scratchpath=$2
mexp=$3
mevals=$4
name="dataO_maxExp=${mexp}_maxEv=${mevals}_$(date --iso-8601=seconds)"
dir=${scratchpath}/dataFAR/dataO/${name}
mkdir -p ${dir}

for r in $(seq 2); do
    echo "Run $r/15";
    cmd="qsub -N iraceO_maxExp=${exp}_maxEv=${evals}_${r} -q beta -l select=1:ncpus=1 -l walltime=00:10:00 -- ${scratchpath}/planO/r_iO.sh ${dir} ${r} ${mexp} ${mevals} ${myhome}"
    echo $cmd
    $cmd
    #time (p=2; while [[ ${p} > 1 ]] ; do p=$(qqueue -u $USER | wc -l); echo "$r: $p"; sleep 300; done)
done

#echo "DONE"
#date -Iseconds

