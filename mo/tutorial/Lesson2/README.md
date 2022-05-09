# How to implement and use neighborhoods
In this lesson, you will learn how to implement a neighbor, neighborhood and the evaluation function. Two ways will be show, one generic and one using an indexed neighborhoods. As an example, it will be illustrated on the Queens problem.

1. Classical neighborhoods (example with a swap operator)
2. Indexed neighbordhoods (example with a shift operator)
3. Evaluation of neighbors
4. Exercise 

## 1. Classical neighborhoods (example with a swap operator)

### Implementation
To implement a neighborhood for your problem, you must have a class that inherits from "moNeighborhood" and a class that inherits from "moNeighbor" for the corresponding neighbors. As a consequence, in the neighborhood class, you have to implement the following methods:

hasNeighbor (test if there is at least one valid neighbor)
init (init the first neighbor)
cont (test if there is again a valid neighbor)
next (compute the next valid neighbor)
And in the neighbor class:

move (how to apply the move corresponding to the neighbor on a solution)
### Example
In the "paradiseo-mo/src/problems/permutation" directory, classical neighborhood and neighbor for swap operator (moSwapNeighborhood.h and moSwapNeighbor.h) are defined. Some methods are specific to the swap operator and you can see a "move_back" methods that is explained at the end of this tutorial.

In "mo/tutorial/Lesson2" directory, open the source file "testNeighborhood.cpp". You can see how to use this first neighborhood...

After inclusion, useful types are defined for more lisibility:

Define type of representation
```c++
typedef eoInt<unsigned int> Queen;
```
Define type of a swap neighbor
```c++
typedef moSwapNeighbor<Queen> swapNeighbor;
```
Define type of the swap neighborhood
```c++
typedef moSwapNeighborhood<Queen> swapNeighborhood;
```
And in the "main" fonction, a neighborhood, a solution and a neighbor are declared:
```c++
swapNeighborhood swapNH;
Queen solution;
swapNeighbor n1;
```

Then they are used to explore and print all the neighbors of the neighborhood for a Queen problem of size 8 (swapEval is the evaluation function declared previously)
```c++
swapNH.init(solution, n1);
swapEval(solution,n1);
n1.print();
while(swapNH.cont(solution)){
    swapNH.next(solution, n1);
    swapEval(solution,n1);
    n1.print();
}
```

You can run the executable on the lesson 2 directory and see the output (the beginning).

## 2. Indexed neighbordhoods (example with a shift operator)

### Implementation
Three indexed neighborhoods are already defined in Paradiseo-MO. To use them you have to know the size of your neighborhoods and define a mapping that associates a neighbor from a known key, in your class neighbor. This neighbor must inherit from "moIndexNeighbor".

### Example
In the mo/src/problems/permutation" directory, a neighbor for shift operator (moShiftNeighbor.h) is defined. In this class, the mapping is done in the method "translate".

After inclusion useful types are defined for more lisibility:

Define type of a shift neighbor
```c++
typedef moShiftNeighbor<Queen> shiftNeighbor;
```
Define three different indexed neighborhoods for shift operator
```c++
typedef moOrderNeighborhood<shiftNeighbor> orderShiftNeighborhood;
typedef moRndWithoutReplNeighborhood<shiftNeighbor> rndWithoutReplShiftNeighborhood;
typedef moRndWithReplNeighborhood<shiftNeighbor> rndWithReplShiftNeighborhood;
```

And in the "main" fonction, a shift neighbor and the three indexed neighborhoods are declared:
```c++
shiftNeighbor n2;
orderShiftNeighborhood orderShiftNH(pow(vecSize-1, 2));
rndWithoutReplShiftNeighborhood rndNoReplShiftNH(pow(vecSize-1, 2));
rndWithReplShiftNeighborhood rndReplShiftNH(pow(vecSize-1, 2));
```

Exploration of the neighborhoods is done like with a classical neighborhood.

You can run the executable on the lesson 2 directory and see the output.

## 3. Evaluation of neighbors

There are three ways to evaluate a neighbor:

1. Incremental evaluation
2. Full evaluation by modification
3. Full evaluation by copy

In terms of performance, it is more efficient to use incremental evaluation and if it cannot be defined, full evaluation by modification is better than that one by copy.

### Incremental evaluation
To implement an incremental evaluation, you have to create a class which inherits of "**moEval**". So you have to define the method:
```c++
void operator()(EOT&, Neighbor&){ ... }
```
EOT and Neighbor are respectively the templates for a solution and a neighbor.

### Full evaluation
The two full evaluations are already defined in Paradiseo-MO. The full evaluation by modification applies the move on the initial solution, evaluates the obtained solution and affects the fitness value to the neighbor. Then the "moveBack" is applied to come back to the initial solution. On the other hand, the full evaluation by copy applies the move on a temporary copy of the solution, evaluates it and affects the fitness value to the neighbor.

To use these evaluations, you need your classical full evaluation function ("eoEvalFunc") in the constructors:
```c++
moFullEvalByCopy(eoEvalFunc<EOT>& _eval)
moFullEvalByModif(eoEvalFunc<EOT>& _eval) 
```

Be carefull, if you want to use the class "moFullEvalByModif", your neighbor must be "backable" and so it has to inherit of the class "**moBackableNeighbor**" and consequently to have a method "moveBack".

## 4. Exercise

Try to define an indexed swap neighbor like in the file "moShiftNeighbor.h". Then explore and print the neighborhood randomly.