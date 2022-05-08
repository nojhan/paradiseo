# Lesson3 - How to use Simulated Annealing and Checkpointing
In this lesson, a simple simulated annealing is presented, using an order neighborhood based on a shift operator, to solve the Queen problem. Then, a checkpoint will be used to save some informations during the search.

1. Simulating Annealing on the Queen problem.
2. Checkpointing
3. Avalaible statistics in MO
4. Exercise

## 1. Simulating Annealing (example on the Queen problem)

First you have to define the representation of a Queen, how to initialize and evaluate it. So you have to declare three classes:
```c++
queenFullEval<Queen> fullEval;
eoInitPermutation<Queen> init(vecSize);
Queen solution1;
```

Then, you have to ramdomly intialize and evaluate the solution:
```c++
init(solution1);
fullEval(solution1);
```

Let see the most simple constructor of a Simulated Annealing (in algo/moSA.h). You need three parameters:
* a neighborhood
* a full evaluation function (declared before)
* a neighbor's evaluation function
```c++
moFullEvalByCopy<shiftNeighbor> shiftEval(fullEval);
rndShiftNeighborhood rndShiftNH(pow(vecSize-1, 2));
```

You can now declare the Simulated Annealing:
```c++
moSA<shiftNeighbor> localSearch1(rndShiftNH, fullEval, shiftEval);
```
This simple constructor uses by default three components:
* moSimpleCoolingSchedule (with default parameters)
* moSolNeighborComparator
* moTrueContinuator

More flexible constructors exist in which you can change these components. In the following, the "moTrueContinuator" is replaced by a "moCheckpoint".

You can try this first algorithm with different problem sizes (use parameter file or the option --vecSize=X on command line to execute "testSimulatedAnnealing"). It prints the initial and final solution1.

## 2. Checkpointing (example on the Queen problem)

The class "moCheckpoint" inherits of the abstract class "moContinuator" and allows to incorporate one or many "moContinuator" classes (Composite pattern). It also allows to incorporate many "eoMonitor", "eoUpdater" and "moStatBase" classes.

Here, an example of checkpointing is presented, including:
* a continuator returning always true (moTrueContinuator)
* a monitor saving information in a file (eoFileMonitor)
* an updater using the file monitor with a determinated frequency (moCounterMonitorSaver)
* a very simple statistical operator giving only the fitness of the current solution (moFitnessStat)

First, you have to define the "moTrueContinuator" and build the "moCheckpoint":
```c++
moTrueContinuator<shiftNeighbor> continuator;
moCheckpoint<shiftNeighbor> checkpoint(continuator);
```

Then, create the "moFitnessStat" and add it in the checkpoint:
```c++
moFitnessStat<Queen> fitStat;
checkpoint.add(fitStat);
```

Finally, create the "eoFileMonitor" to write fitness values in the file fitness.out and the "moCounterMonitorSaver" to use the file monitor only for each 100 iterations.
```c++
eoFileMonitor monitor("fitness.out", "");
moCounterMonitorSaver countMon(100, monitor);
checkpoint.add(countMon);
monitor.add(fitStat);
```

So you can create a Simulated Annealing with this checkpoint:
```c++
moSA<shiftNeighbor> localSearch2(rndShiftNH, fullEval, shiftEval, coolingSchedule, solComparator, checkpoint);
```

Try this second algorithm with different problem sizes (use parameter file or the option --vecSize=X on command line to execute "testSimulatedAnnealing"). It prints the initial and final solution2 and you can see the evolution of fitness values in the file fitness.out (only 1 value each 100 iterations).

## 3. Avalaible statistics

A lot of statistics are avalaible to have informations during the search:

* moCounterStat
* moMinusOneCounterStat
* moStatFromStat
* moFitnessStat
* moNbInfNeighborStat
* moNbSupNeighborStat
* moNeutralDegreeNeighborStat
* moSizeNeighborStat
* moNeighborhoodStat
* moDistanceStat
* moSolutionStat
* moBestSoFarStat
* moSecondMomentNeighborStat
* moMaxNeighborStat
* moMinNeighborStat
* moNeighborBestStat
* moNeighborFitnessStat
* moAverageFitnessNeighborStat
* moStdFitnessNeighborStat

## 4. Exercise

1. Try to add the cooling schedule parameters into the parameters file. Then, try the simulated annealing with different parameters to see theirs impacts on the search.
2. Add an existed operator (in continuator directory) to print the solution each 100 iterations.