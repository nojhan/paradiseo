# Dynamic island topology

Three islands start with a ring topology and switch to a complete topology
after 10 seconds of computation.

Same QAP problem as Lessons 1-2.

## Running

From the `build/smp/tutorial/Lesson4` directory:

```shell
./lesson4_topology
```

## How it works

A callback function registered with the islands triggers the topology
change:

```c++
void changeTopo(IslandModel<Indi>* _model, AbstractTopology& _topo)
{
    // ... after 10 seconds:
    _model->setTopology(_topo);
}
```

The topology can be changed at runtime via `model.setTopology()`, so you
can implement adaptive communication patterns based on time, convergence,
or any other criterion.
