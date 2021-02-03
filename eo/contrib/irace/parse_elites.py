#!/usr/bin/env python3

import sys

print("algo,problem,seed,ECDF_AUC")

for fname in sys.argv[1:]:

    run = {}
    for f in fname.strip(".dat").split("_"):
        kv = f.split("=")
        assert(len(kv)==2),str(kv)+" "+str(len(kv))
        key,idx = kv[0], int(kv[1])
        run[key] = idx

    with open(fname) as fd:
        auc = int(fd.readlines()[0]) * -1

    algo = "pc={}_c={}_C={}_pm={}_m={}_M={}_R={}".format(run["crossover-rate"],run["cross-selector"],run["crossover"],run["mutation-rate"],run["mut-selector"],run["mutation"],run["replacement"])

    print(algo, run["pb"], run["seed"], auc, sep=",")
