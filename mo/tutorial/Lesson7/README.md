# How to hybrid an evolutionary algorithm and a local search
In this lesson, a hybridization between an evolutionary algorithm(EA) and a local search is presented. It will be illustrated by an example on the Queen problem. Here, the hybridization consists in replacing the mutation operator of the EA by a first improvement hill climber.

1. hybridization
2. Exercise

## 1. Hybridization (example on the Queen problem)

First, you have to define the represenation of a Queen, how to initialize and how to evaluate a population of solutions:
```c++
queenFullEval<Queen> fullEval;

eoInitPermutation<Queen> init(vecSize);

eoPop<Queen> pop;
Queen tmp;
for(unsigned int i=0; i<20; i++){ //population size is fixed to 20
    init(tmp);
    fullEval(tmp);
    pop.push_back(tmp);
}
```

As in previous lessons, a local search is declared (first improvement hill climber):
```c++
moFullEvalByCopy<shiftNeighbor> shiftEval(fullEval);

orderShiftNeighborhood orderShiftNH(pow(vecSize-1, 2));

moFirstImprHC<shiftNeighbor> hc(orderShiftNH, fullEval, shiftEval);
```
To hybrid this local search with an EA, you just have to use it instead of a classical mutation:
```c++
eoOrderXover<Queen> cross;
eoSGATransform<Queen> transform(cross, 0.3, hc, 0.7); // cross and mutation probabilities are fixed
```

Others components of the "eoEasyEA" have to be declared:
```c++
eoGenContinue<Queen> EAcont(50); //nb generations is fixed to 50
eoDetTournamentSelect<Queen> selectOne(2); //size of tournament is fixed to 2
eoSelectMany<Queen> select(selectOne, 1); //rate of selection is fixed to 1
eoGenerationalReplacement<Queen> repl;
```
More details are available in EO lessons.

Finally, the hybrid algorithm is declared as:
```c++
eoEasyEA<Queen> hybridAlgo(EAcont, fullEval, select, transform, repl);
```
and should be applied on the population with:
```c++
hybridAlgo(pop);
```

You can test this hybrid algorithm by changing problem size (use parameters file or the option --vecSize=X on command line to execute "hybridAlgo"). It prints the initial and final population.

## 2. Exercise

Try to use a hybridization at the checkpointing step rather than at the mutation step. You have to implement an "eoUpdater" which applies a local search. This updater should be added in a "eoCheckpoint".