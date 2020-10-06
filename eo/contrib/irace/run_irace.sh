#!/bin/bash

if [[ $# != 1 ]] ; then
    echo "ERROR: build dir not indicated"
    exit 1
fi

cd $1
pwd

# Fore some reason, irace absolutely need those files...
cp ../irace-config/example.scen .
cp ../irace-config/target-runner .
cp ../irace-config/default.instances .

# Generate the parameter list file.
./fastga -h > fastga.param 2>/dev/null
/usr/lib/R/site-library/irace/bin/irace --scenario example.scen

