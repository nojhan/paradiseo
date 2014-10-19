README
------

To launch a set of experiments with t-mpi-distrib-exp:

0) Compile it:

    mpic++ -o distrib-exp t-mpi-distrib-exp.cpp -I../../src/ -I../../src/mpi/ -DWITH_MPI -L ../../../build/eo/lib/ -leoutils -leo -leompi -leoserial

1) Generate the experiments, thanks to the script gen-xp.py
This script will guide you and ask you for all experiments. The prefix is used in the results filenames.
You may want to modify the name of the experiments file (default value: "experiments.json") or
the pattern of the results files. However, you have to ensure that the pattern is an one-to-one
function of the parameters, otherwise some results could be lost.

2) Launch the t-mpi-distrib-exp program with mpirun:
For 4 cores (= 1 master + 3 workers)
mpirun -np 4 ./t-mpi-distrib-exp --use-experiment-file=1 --experiment-file=/home/eodev/eo/test/mpi/experiments.json

For 16 cores (= 1 master + 15 workers)
mpirun -np 5 ./t-mpi-distrib-exp --use-experiment-file=1 --experiment-file=/home/eodev/eo/test/mpi/experiments.json

3) The program will generate the results of the experiments, as txt files. There is one result file for each run of each
experiment.

