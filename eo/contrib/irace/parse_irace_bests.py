#!/usr/bin/env python3

import os
import re
import sys

print("pb,ecdf,crossover-rate,cross-selector,crossover,mutation-rate,mut-selector,mutation,replacement")
for datadir in sys.argv[1:]:

    for pb_dir in os.listdir(datadir):
        if "results_problem" in pb_dir:
            pb_id=pb_dir.replace("results_problem_","")
            with open(os.path.join("./",datadir,pb_dir,"irace.log")) as fd:
                data = [line.strip() for line in fd.readlines() if "--crossover-rate=" in line]
                for line in data:
                    algo=re.sub("\-\-\S*=", ",", line)
                    csv_line=pb_id+","+algo
                    print(csv_line.replace(" ",""))
