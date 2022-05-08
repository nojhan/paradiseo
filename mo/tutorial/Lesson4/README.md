# How to use Tabu Search
In this lesson, a simple tabu search is presented, using an order neighborhood based on a shift operator, to solve the Queen problem.
1. Tabu Search on the Queen problem.
2. Exercise

## 1. Tabu Search (example on the Queen problem)

First you have to define the representation of a Queen, how to initialize and how to evaluate it. So you have to declare three classes:
```c++
queenFullEval<Queen> fullEval;
eoInitPermutation<Queen> init(vecSize);
Queen sol1;
```

Then, you have to ramdomly intialize a solution:
```c++
init(sol1);
fullEval(sol1);
```

Let see the most simple constructor of a Tabu Search (in mo/src/algo/moTS.h). You need five parameters:

* a neighborhood
```c++
orderShiftNeighborhood orderShiftNH(pow(vecSize-1, 2));
```
* a full evaluation function (declared before)
* a neighbor evaluation function*
```c++
moFullEvalByCopy<shiftNeighbor> shiftEval(fullEval);
```
* a time limit for the search (in seconds)
* a size for the tabu list

You can now declare the Tabu Search:
```c++
moTS<shiftNeighbor> localSearch1(orderShiftNH, fullEval, shiftEval, 2, 7);
// 2 is the time limit, 7 is the size of the tabu List
```

This simple constructor uses by default seven components:
* moTimeContinuator
* moNeighborComparator
* moSolNeighborComparator
* moNeighborVectorTabuList
* moDummyIntensification
* moDummyDiversification
* moBestImprAspiration

More flexible constructors exist as you can change these components:
```c++
moNeighborVectorTabuList<shiftNeighbor> tl(sizeTabuList,0);
moTS<shiftNeighbor> localSearch2(orderShiftNH, fullEval, shiftEval, 3, tl);
// 3 is the time limit
```
In this one, the tabuList has been specified.
```c++
moTS<shiftNeighbor> localSearch3(orderShiftNH, fullEval, shiftEval,
    comparator, solComparator, continuator, tl, inten, div, asp);
```
In this one, comparators, continuator, tabu list, intensification strategy, diversification strategy and aspiration criteria have been specified.

You can test these three algorithms by changing problem sizes, time limit and the size of tabu list (use parameters file or the option --vecSize=X, --timeLimit=Y and --sizeTabuList=Z on command line to execute "testSimpleTS"). It prints the initial and final solutions.

## 2. Exercise

1. Try to implement and use a diversification strategy in 'testSimpleTS". You can also use a predifined strategy: moMonOpDiversification (in "memory" directory)