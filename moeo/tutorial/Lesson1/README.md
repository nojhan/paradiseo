# How to run your first multi-objective EA?

In this lesson you will solve the Schaffer SCH1 bi-objective problem with
NSGA-II using the ParadisEO-MOEO framework.

The SCH1 problem minimizes two objectives over a single real-valued
variable x in [0, 2]:

- f1(x) = x^2
- f2(x) = (x - 2)^2

## 1. Running

From the `build/moeo/tutorial/Lesson1` directory:

```shell
./Sch1 --help
./Sch1 --popSize=100 --maxGen=100
```

Parameters can also be read from a file:

```shell
./Sch1 @Sch1.param
```

## 2. Browsing the code

Open `Sch1.cpp` and follow along.

First, define the objective vector traits. This tells MOEO how many
objectives the problem has and whether each is minimizing or maximizing:

```c++
class Sch1ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int) { return true; }
    static bool maximizing (int) { return false; }
    static unsigned int nObjectives () { return 2; }
};
```

The solution class `Sch1` inherits from `moeoRealVector` — a real-valued
decision vector that also carries an objective vector:

```c++
class Sch1 : public moeoRealVector < Sch1ObjectiveVector >
{
public:
    Sch1() : moeoRealVector < Sch1ObjectiveVector > (1) {}
};
```

The evaluator computes both objectives and stores them via
`objectiveVector()`:

```c++
void operator () (Sch1 & _sch1)
{
    if (_sch1.invalidObjectiveVector()) {
        Sch1ObjectiveVector objVec;
        double x = _sch1[0];
        objVec[0] = x * x;
        objVec[1] = (x - 2.0) * (x - 2.0);
        _sch1.objectiveVector(objVec);
    }
}
```

The algorithm itself is a single line — `moeoNSGAII` takes the generation
limit, evaluator, crossover, and mutation as constructor arguments:

```c++
moeoNSGAII < Sch1 > nsgaII (MAX_GEN, eval, xover, P_CROSS, mutation, P_MUT);
```

After the run, extract the Pareto front with `moeoUnboundedArchive`:

```c++
moeoUnboundedArchive < Sch1 > arch;
arch(pop);
arch.sortedPrintOn(cout);
```
