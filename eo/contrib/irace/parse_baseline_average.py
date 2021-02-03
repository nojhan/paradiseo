#!/usr/bin/env python3

import sys

print("problem,EA,fEA,xGA,1ptGA")

algos_names = {
    str({"crossover-rate":0, "cross-selector":0, "crossover":0, "mutation-rate":0, "mut-selector":0, "mutation":1, "replacement":0}) : "EA",
    str({"crossover-rate":0, "cross-selector":0, "crossover":0, "mutation-rate":0, "mut-selector":0, "mutation":5, "replacement":0}) : "fEA",
    str({"crossover-rate":2, "cross-selector":2, "crossover":2, "mutation-rate":2, "mut-selector":2, "mutation":1, "replacement":0}) : "xGA",
    str({"crossover-rate":2, "cross-selector":2, "crossover":5, "mutation-rate":2, "mut-selector":2, "mutation":1, "replacement":0}) : "1ptGA",
}

data = {}

# Parse
for fname in sys.argv[1:]:

    run = {}
    for f in fname.strip(".dat").split("_"):
        kv = f.split("=")
        assert(len(kv)==2),str(kv)+" "+str(len(kv))
        key,idx = kv[0], int(kv[1])
        run[key] = idx

    with open(fname) as fd:
        auc = int(fd.readlines()[0]) * -1

    algo = str({"crossover-rate":run["crossover-rate"], "cross-selector":run["cross-selector"], "crossover":run["crossover"], "mutation-rate":run["mutation-rate"], "mut-selector":run["mut-selector"], "mutation":run["mutation"], "replacement":run["replacement"]})

    if run["pb"] in data:
        data[run["pb"]][algos_names[algo]].append(auc)
    else:
        data[run["pb"]] = {
                "EA"   : [],
                "fEA"  : [],
                "xGA"  : [],
                "1ptGA": [],
            }
        data[run["pb"]][algos_names[algo]].append(auc)


# Print CSV
for pb in sorted(data.keys()):
    print(pb, end="")
    for algo in ["EA","fEA","xGA","1ptGA"]:
        res = data[pb][algo]
        print(",", sum(res)/len(res), end="", sep="")
    print()
