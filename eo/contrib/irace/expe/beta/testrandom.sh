#!/bin/bash

#tab=(15000 20000 30000 40000)
#tab=(100 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000)
myhome=$1
scratchpath=$2
tab=${@:3} 
#echo ${tab[@]}
outdir="/scratchbeta/$USER/fastga_results_all/fastga_results_random"
mkdir -p ${outdir} #results of random experiment
for evals in ${tab[@]}; do
    #evalsdir="${name}/maxEv=${evals}"
    #mkdir -p ${evalsdir}
    #{ time -p bash /home/$USER/run_random.sh ${name} ${i} 50 ; } &> "${name}/sortie5_${j}_maxExp=${i}.txt"
    #cmd="qsub -N iraceR_maxEv=${evals} -q beta -l select=1:ncpus=1 -l walltime=00:30:00 -- /scratchbeta/$USER/run_random.sh ${outdir} ${evals}"
    $cmd
   
done
