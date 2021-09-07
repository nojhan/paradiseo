#!/bin/bash
lexp=(300 600)
levals=(100 500)
myscratchpath=/scratchbeta/$USER
myhome=${HOME}
for exp in ${lexp[@]} ; do
    for evals in ${levals[@]} ; do
        bash ./planF/riaF.sh ${myhome} ${myscratchpath} ${exp} ${evals} 
        bash ./planO/riaO.sh ${myhome} ${myscratchpath} ${exp} ${evals}
        bash ./planA/riaA.sh ${myhome} ${myscratchpath} ${exp} ${evals} 
    done
done
bash testrandom.sh ${myhome} ${scratchpath} ${levals[@]}    
