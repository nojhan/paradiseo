#!/bin/bash

ldata="./fastga_results_all/"  #fastga_results_all
figpath="./hist_and_csv/"  #hist_and_csv

#get distribution of operators variants of all problems of each plan of fastga_results_all
#fastga_results_all contains all experiments of all plans 

ldir=$(echo $(ls ${ldata})) #list of directory of each plan
for plan in ${ldir[@]} ; do #get the directory of each plan
    lexperiment=$(echo $(ls ${ldata}/${plan}))

    for experiments in ${lexperiment[@]} ; do
        path="${ldata}/${plan}/${experiments}"

        #----------------average aucs of each algo for each pb only for plan A,F,O ---------------
        #myfig=${figpath}/auc_average_${plan}
        #mkdir -p ${myfig}
        #cmd="python3 parse_auc_average.py ${path} "
        #$cmd > "${myfig}/auc_average_${experiments}.csv"
        #--------------distribution of operators by pb and for all pb only for plan A,F,O ------
        #myfig=${figpath}/distribution_op_${plan}
        #mkdir -p ${myfig}
        #cmd="python3 distribution_op_all.py ${path} ${myfig} "
        #$cmd 
        #--------------best out csv--------
        cmd="python3 best_out_of_elites.py ${path}"
        myfig=${figpath}/best_out_${plan}
        mkdir -p ${myfig}
        $cmd > ${myfig}/best_out_all_pb_${experiments}.csv
        echo ${cmd}

    done
done

#---------------distribution of operators of randoma algo------------------
#rpath=${ldata}/fastga_results_random
#cmd="python3 dist_op_random.py ${rpath} ${figpath}"
#$cmd
#---------------random---------------