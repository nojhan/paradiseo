#!/bin/bash
ldata=$1
file_py=$2
csvdir="csv_FA"
ldir=$(echo $(ls ${ldata}))
for data in ${ldir[@]} ; do 
    path="${ldata}/${data}"
    cmd="python3 ${file_py} ${path}"
    plan_name=$(echo ${data} | sed "s/data//")
    mexp=$(echo ${data[@]} | cut -d _ -f2)
    mevals=$(echo ${data[@]} | cut -d _ -f3)
    ddate=$(echo ${data[@]} | cut -d _ -f4)
    name="results_irace_plan${plan_name[@]:0:1}_${mexp}_${mevals}_${ddate}"
    mkdir -p "${csvdir}/csv_plan${plan_name[@]:0:1}"
    ${cmd} > "${csvdir}/csv_plan${plan_name[@]:0:1}/${name}.csv"
done
