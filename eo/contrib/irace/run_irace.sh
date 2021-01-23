#!/bin/bash

outdir="$(date --iso-8601=minutes)_results_irace"
mkdir ${outdir}
cd ${outdir}

for p in $(seq 0 18) ; do
    echo -n "Problem ${p}... "
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
    echo " done"
done
