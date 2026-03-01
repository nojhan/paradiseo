# Dominance-based multi-objective local search (DMLS)

Same flow-shop problem as Lessons 2-3, but solved with local search instead
of an evolutionary algorithm. This combines ParadisEO-MO local search
components with the MOEO multi-objective framework.

## Running

From the `build/moeo/tutorial/Lesson4` directory:

```shell
./FlowShopDMLS @FlowShopDMLS.param
```

## How it works

The program uses `moShiftNeighbor` and `moOrderNeighborhood` from
ParadisEO-MO, with `moeoFullEvalByCopy` to adapt the full solution
evaluator for neighbor evaluation:

```c++
typedef moShiftNeighbor<FlowShop, FlowShopObjectiveVector> Neighbor;

moOrderNeighborhood<Neighbor> neighborhood((nhSize-1) * (nhSize-1));
moeoFullEvalByCopy<Neighbor> moEval(eval);
```

At each iteration, DMLS explores all non-dominated solutions in the
archive via local search moves and updates the archive with any improving
neighbors:

```c++
moeoFirstImprovingNeighborhoodExplorer<Neighbor> explor(neighborhood, moEval);
moeoUnifiedDominanceBasedLS<Neighbor> algo(checkpoint, eval, arch, explor, select);
```
