#!/usr/bin/env python3

import os
import re
import sys

print("pb,ecdf,id,crossover-rate,cross-selector,crossover,mutation-rate,mut-selector,mutation,replacement")
for datadir in sys.argv[1:]:

    for pb_dir in os.listdir(datadir):
        if "results_problem" in pb_dir:
            pb_id=pb_dir.replace("results_problem_","")
            with open(os.path.join("./",datadir,pb_dir,"irace.log")) as fd:
                data = fd.readlines()

                # Find the last best configuration
                bests = [line.strip() for line in data if "Best-so-far" in line]
                best = bests[-1].split()
                best_id, best_perf = best[2], best[5]
                # print(best_id,best_perf)

                # Filter the config detail
                configs = [line.strip() for line in data if "--crossover-rate=" in line and best_id in line]
                # print(configs)

                # Format as CSV
                for config in configs:
                    algo = re.sub("\-\-\S*=", ",", config)
                    csv_line = pb_id + "," + best_perf + "," + algo
                    print(csv_line.replace(" ",""))
