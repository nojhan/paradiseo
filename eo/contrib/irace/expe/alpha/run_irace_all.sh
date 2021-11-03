#!/bin/bash

date -Iseconds
echo "STARTS"

for r in $(seq 15); do
    echo "Run $r/15";
    date -Iseconds
    ./run_irace_parallel-batch.sh
    date -Iseconds
done

echo "DONE"
date -Iseconds
