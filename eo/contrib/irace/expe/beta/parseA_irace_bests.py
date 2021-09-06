#!/usr/bin/env python3
#parse data1
import os
import re
import sys

print("ecdf,id,crossover-rate,cross-selector,crossover,mutation-rate,mut-selector,mutation,replacement,pop-size,offspring-size")


#give the path of one experiment
argv=sys.argv[1]
for datadir in os.listdir(argv):
    #if(os.path.isdir(os.path.join(argv,datadir))): check if argv/datadir is a directory
    if(datadir.find("results_irace")>=0): #check if the directory is one JOB
        with open(os.path.join("./",argv,datadir,"irace.log")) as fd:
            data = fd.readlines()

                    # Find the last best configuration
            bests = [line.strip() for line in data if "Best-so-far" in line]
                    #print(datadir,bests)
            best = bests[-1].split()
            best_id, best_perf = best[2], best[5]
                    # print(best_id,best_perf)

                    # Filter the config detail
            configs = [line.strip() for line in data if "--crossover-rate=" in line and best_id in line]
                    # print(configs)
                    # Format as CSV
            algo = re.sub("\-\-\S*=", ",", configs[0])
            csv_line = best_perf + "," + algo
            print(csv_line.replace(" ",""))