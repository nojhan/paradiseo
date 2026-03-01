# Variable Neighborhood Search (VNS)

VNS on the N-Queens problem, using multiple neighborhood structures to
escape local optima.

The N-Queens problem places N queens on an N x N chessboard so that no two
queens attack each other. The representation is a permutation (one queen
per column) and the fitness counts conflicts (minimized).

## Running

From the `build/mo/tutorial/Lesson9` directory:

```shell
./VNS -V=12
```

This runs VNS on a 12-queens instance with a 3-second time limit.

## How it works

VNS alternates between perturbation (shaking) and local search using
different neighborhood structures. Here, two neighborhoods are used:

```c++
shiftNeighborhood shiftNH((vecSize-1) * (vecSize-1));
swapNeighborhood swapNH(vecSize * (vecSize-1) / 2);
```

Each has its own hill-climber and mutation operator:

```c++
moSimpleHC<shiftNeighbor> ls1(shiftNH, fullEval, shiftEval);
moSimpleHC<swapNeighbor> ls2(swapNH, fullEval, swapEval);

moRndVectorVNSelection<Queen> selectNH(ls1, shiftMut, true);
selectNH.add(ls2, swapMut);
```

`moRndVectorVNSelection` randomly picks which neighborhood/local search
to apply at each iteration. Forward (`moForwardVectorVNSelection`) and
backward (`moBackwardVectorVNSelection`) selection strategies are also
available. The search runs until the `moTimeContinuator` time limit:

```c++
moTimeContinuator<shiftNeighbor> cont(3);  // 3 seconds
moVNS<shiftNeighbor> vns(selectNH, acceptCrit, fullEval, cont);
```
