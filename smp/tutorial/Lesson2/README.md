# Homogeneous island model

Three islands running the same algorithm (`eoEasyEA`) with a complete
topology — all islands exchange individuals with all others.

Same QAP problem as Lesson 1.

## Running

From the `build/smp/tutorial/Lesson2` directory:

```shell
./lesson2_homogeneous
```

The problem instance is loaded from `../lessonData.dat`.

## How it works

All islands share the same operators (evaluation, selection, crossover,
mutation, replacement) but can have different parameters. The topology
and island model are set up with:

```c++
Topology<Complete> topo;
IslandModel<Indi> model(topo);
```

Each island gets its own population and generation limit, then the model
runs them in parallel and handles migration.
