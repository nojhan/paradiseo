# Multi-objective EA on the flow-shop scheduling problem

This lesson solves a more realistic combinatorial problem: the bi-objective
permutation flow-shop, which minimizes both makespan and total flow time.

## Running

From the `build/moeo/tutorial/Lesson2` directory:

```shell
./FlowShopEA @FlowShopEA.param
```

The parameter file specifies the problem instance, population size, operator
rates, and fitness assignment strategy.

## How it works

The code uses ParadisEO's `do_make_*` factory functions to build all
components from command-line parameters:

```c++
eoEvalFuncCounter<FlowShop>& eval = do_make_eval(parser, state);
eoInit<FlowShop>& init = do_make_genotype(parser, state);
eoGenOp<FlowShop>& op = do_make_op(parser, state);
```

The flow-shop representation is defined in `FlowShop.h` — a
permutation-based `moeoVector` with a bi-objective vector. Different
Pareto-based fitness assignment strategies (NSGA-II, SPEA2, IBEA, etc.)
can be selected via parameters.
