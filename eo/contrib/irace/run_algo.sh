#!/bin/bash

# Number of runs (=seeds).
runs=50

# Array of problems.
# You may set something like: (0 2 5 17)
problems=($(seq 0 18))

# Capture anything passed to the script
algo="$@"

# You most probably want to run on release builds.
exe="./release/fastga"

i=1 # Loop counter.
for pb in "${problems[@]}" ; do # Iterate over the problems array.
    for seed in $(seq ${runs}) ; do # Iterates over runs/seeds.
        # Forge a directory/log file name
        # (remove double dashs and replace spaces with underscore).
        name="pb=${pb}_seed=${seed}_$(echo "${algo}" | sed 's/--//g' | sed 's/ /_/g')"

        # This is the command to be ran.
        cmd="${exe} --problem=${pb} --seed=${seed} ${algo}"
        # echo ${cmd} # Print the command.

        # Progress print.
        echo "problem ${pb}, run ${seed}"

        # Actually start the command.
        ${cmd} > "${name}.dat" 2> "${name}.log"

        # Check for the most common problem in the log file.
        cat "${name}.log" | grep "illogical performance"

        perc=$(echo "scale=2;${i}/(${#problems[@]}*${runs})*100" | bc)
        echo -e "${perc}%\n"
        i=$((i+1))
    done
done

echo "Done $((${#problems[@]}*${runs})) runs"
