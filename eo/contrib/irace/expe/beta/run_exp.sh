#!/bin/bash
lexp=(300 600 1000 10000)
levals=(100 500 1000)
myscratchpath=/scratchbeta/$USER
myhome=${HOME}
for exp in ${lexp[@]} ; do
    for evals in ${levals[@]} ; do
        bash ./planF/riaF.sh ${myhome} ${myscratchpath} ${exp} ${evals} 
        bash ./planA/riaA.sh ${myhome} ${myscratchpath} ${exp} ${evals} 
    done
done
bash testrandom.sh ${myhome} ${scratchpath} ${levals[@]}    
