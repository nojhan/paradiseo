#!/usr/bin/env python3
#parse data1
import os
import re
import sys
#print("pb,ecdf,id,crossover-rate,cross-selector,crossover,mutation-rate,mut-selector,mutation,replacement") #plan1
print("pb,ecdf,id,crossover-rate,cross-selector,crossover,mutation-rate,mut-selector,mutation,replacement,pop-size,offspring-size")


#give the path of one experiment 
argv=sys.argv[1]
for datadir in os.listdir(argv):
    #if(os.path.isdir(os.path.join(argv,datadir))): check if argv/datadir is a directory
    if(datadir.find("results_irace")>=0): #check if the directory is one JOB
        for pb_dir in os.listdir(os.path.join(argv,datadir)):
            if "results_problem" in pb_dir:
                pb_id=pb_dir.replace("results_problem_","")
                with open(os.path.join("./",argv,datadir,pb_dir,"irace.log")) as fd:
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
                    csv_line = pb_id + "," + best_perf + "," + algo
                    print(csv_line.replace(" ",""))
