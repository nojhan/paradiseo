#!/bin/bash

#get csv file, parse dataF in a csv file
dir=/scratchbeta/$USER/dataFAR
listdir=$(echo $(ls ${dir}))

for data in ${listdir[@]} ; do
    file_py="parse${data: -1}_irace_bests.py"
    path="${dir}/${data}"
    cmd="bash ./csv_all_bests.sh ${path} ${file_py}"
    echo $cmd
    $cmd
done

#get validation run of each config 

dir=/scratchbeta/$USER/csv_FA
listdir=$(echo $(ls ${dir}))
echo ${listdir[@]}
for csvdir in ${listdir[@]} ; do
    csvpath="${dir}/${csvdir}"
    file_py="./run_elites_plan${csvdir: -1}.sh"
    cmd="bash ./fastga_elites_all.sh ${csvpath} ${file_py}"
    echo $cmd
    $cmd
done
