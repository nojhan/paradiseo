# How to use Iterated Local Search
In this lesson, an Iterated Local Search is presented. The Tabu Search of the Lesson 4 is used with an order neighborhood based on a shift operator, to solve the Queen problem.

1. Iterated Tabu Search on the Queen problem.
2. Exercise

## 1. Iterated Tabu Search (example on the Queen problem)

As in Lesson 4, you have to define a Solution, the method to initialize and evaluate it. Then you have to define a Tabu Search.

Declaration of the Tabu Search:
```c++
moTS<shiftNeighbor> ts(orderShiftNH, fullEval, shiftEval, 1, 7);
```

To use a simple Iterated Local Search, a mutation operator is needed. So the swap mutation defined in EO is used:
```c++
eoSwapMutation<Queen> mut;
```

Now, a simple Iterated Tabu Search can be declared as follow:
```c++
moILS<shiftNeighbor> localSearch1(ts, fullEval, mut, 3);
```
This constructor has got 4 parameters:
1. a local search (ts)
2. a full evaluation function (fullEval)
3. a mutation operator (mut)
4. a number of iterations (3)

**localSearch1** performs the Tabu Search 3 times. The first solution of each iteration(except the first one) is obtained by applying the mutation operator on the last visited solution.

A constructor allows to specify the continuator. **_Be carefull_**, the continuator must be templatized by a "moDummyNeighbor":
```c++
moIterContinuator<moDummyNeighbor<Queen> > cont(4, false);
```
The explorer of the Iterated local search don't use its own neighborhood. Here, the neighborhood of the Tabu Search is used. But to respect the conception, we create a "moDummyNeighbor" using as template for Iterated Local Search.

An Iterated Tabu Search with this continuator can be declared as:
```c++
moILS<shiftNeighbor> localSearch2(ts, fullEval, mut, cont);
```

A general constructor is available allowing to specify the perturbation operator and the acceptance criteria. First, you have to declare a perturbation operator:
```c++
moMonOpPerturb<shiftNeighbor> perturb(mut, fullEval);
```
And, the acceptance criteria:
```c++
moSolComparator<Queen> solComp;
moBetterAcceptCrit<shiftNeighbor> accept(solComp);
```
Finally, the Iterated Local Search can be declared as:
```c++
moILS<shiftNeighbor> localSearch3(ts, fullEval, cont, perturb, accept);
```

You can test these three algorithms by changing problem sizes(use parameter file or the option --vecSize=X on command line to execute "testILS"). It prints the initial and the final solutions.

## 2. Exercise

* Try to implement an Iterated Hill Climbing on the Queen problem with these caracteristics:
  1. Hill Climbing with a "moShiftNeighborhood" and a "moTrueContinuator"
  2. Iterated Local Search using a "moIterContinuator" and a "moNeighborhoodPerturb" with a "moSwapNeighborhood".