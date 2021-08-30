#!/bin/bash
ldata=$1 # eg : ./csv_plan2/ don t forget to end the path with /
file_py=$2
ldir=$(echo $(ls ${ldata}))
fastga_dir="fastga_results_all"
mkdir -p /scratchbeta/${USER}/${fatga_dir}
#mkdir -p "/home/${USER}/${fastga_dir}/fastga_results_plan1"
mkdir -p "/scratchbeta/${USER}/${fastga_dir}/fastga_results_planF"
mkdir -p "/scratchbeta/${USER}/${fastga_dir}/fastga_results_planA"

for data in ${ldir[@]} ; do 
    path_csv="${ldata}${data}"
    plan_name=$(echo ${data} | sed "s/results_irace_plan//")
    mexp=$(echo ${data[@]} | cut -d _ -f4)
    mexp_id=$(echo ${mexp} | cut -d = -f2)
    mevals=$(echo ${data[@]} | cut -d _ -f5)
    mevals_id=$(echo ${mevals} | cut -d = -f2)
    path="/scratchbeta/${USER}/${fastga_dir}/fastga_results_plan${plan_name[@]:0:1}"
    cmd="bash ${file_py} ${path_csv} ${mexp_id} ${mevals_id} ${path}"
    name="fastga${plan_name[@]:0:1}_${mexp}_${mevals}_$(date -Iseconds)_results_elites_all"
    ${cmd} &> "${path}/output${plan_name[@]:0:1}_fastga_${mexp}_${mevals}_$(date -Iseconds).txt"
done
