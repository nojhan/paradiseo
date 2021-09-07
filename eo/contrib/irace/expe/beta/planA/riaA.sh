#!/bin/bashi
myhome=$1
scratchpath=$2
mexp=$3
mevals=$4
date -Iseconds
echo "STARTS"
dir=${scratchpath}/dataFAR/dataA
#dir=${HOME}/plan4/${name}
#cat ${HOME}/irace_files_pA/example.scen |sed "s/maxExperiments = 0/maxExperiments = ${mexp}/" > ${HOME}/irace_files_pA/example.scen

mkdir -p ${dir}
outdir="${dir}/dataA_maxExp=${mexp}_maxEv=${mevals}_$(date --iso-8601=seconds)"
mkdir -p ${outdir}
for r in $(seq 2); do
    echo "Run $r/15";
    cmd="qsub -N iraceA_maxEv_${r} -q beta -l select=1:ncpus=1 -l walltime=00:25:00 -- ${scratchpath}/planA/r_iA.sh ${outdir} ${r} ${mexp} ${mevals} ${myhome}"
    #cmd="bash ./r_iA_buckets.sh ${outdir} ${r} ${mexp} ${mevals}"
    echo $cmd
    time -p $cmd 
done
echo "DONE"
#cat ${HOME}/irace_files_pA/example.scen |sed "s/maxExperiments = ${mexp}/maxExperiments = 0/" > ${HOME}/irace_files_pA/example.scen
date -Iseconds

