#!/bin/bash

outdir="$(date --iso-8601=ns)_results_irace"
mkdir ${outdir}
cd ${outdir}

run(){
    p="$1"

    echo "Problem ${p}"
    res="results_problem_${p}"
    mkdir ${res}
    cd ${res}

    # Fore some reason, irace absolutely need those files...
    cp ../../irace-config/example.scen .
    cp ../../irace-config/default.instances .
    cp ../../release/fastga .
    cat ../../irace-config/target-runner | sed "s/{{PROBLEM}}/${p}/" > ./target-runner
    chmod u+x target-runner

    # Generate the parameter list file.
    ./fastga -h > fastga.param 2>/dev/null
    # /usr/lib/R/site-library/irace/bin/irace --scenario example.scen 2>&1 | tee irace.log
    /usr/lib/R/site-library/irace/bin/irace --scenario example.scen &> irace.log

    cd ..
    echo "Done problem ${p}"
}

N=5 # Somehow 5 is the fastest on my 4-cores machine.
(
for pb in $(seq 0 18); do 
    ((i=i%N)); ((i++==0)) && wait
    run "$pb" &
done
wait
)
