#!/bin/bash

ldata="./fastga_results_all"  #fastga_results_all
figpath="./hist_and_csv"  #hist_and_csv

ldir=$(echo $(ls ${ldata})) #list of directory of each plan
for plan in ${ldir[@]} ; do #get the directory of each plan
    #------------hist by budget of a Plan (O,R or F)
    #path="${ldata}/${plan}"   
    #cmd="python3 hist_join.py ${path} ${figpath}"
    #echo $cmd
    #$cmd

    #---------------------------hist by pb by budget---------------
    path="${ldata}/${plan}"   
    cmd="python3 hist_by_pb_budget_plan.py ${path} ${figpath}"
    echo $cmd
    $cmd
done

#---------------random------------------
#rpath=${ldata}/fastga_results_random
#cmd="python3 hist_join_random.py ${rpath} ${figpath}"
#---------------random---------------

#--------------------Choose a Budget irace and a budget fastga
mexp=100000
mevals=1000
#-------------------histogram join each plan F,A,R,O and join all algorithms for the budget chosen
cmd="python3 hist_by_FARO.py ${ldata} ${figdir} ${mexp} ${mevals}"
$cmd
#-------------------histogram by pb join each plan F,A,R,O and join all algorithms for the budget chosen
cmd="python3 hist_by_FARO_pb.py ${ldata} ${figdir} ${mexp} ${mevals}"
$cmd
