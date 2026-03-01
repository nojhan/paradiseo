# Master/Workers wrapping with eoEasyEA

Wraps a standard eoEasyEA with the ParadisEO-SMP parallel evaluation model,
applied to the Quadratic Assignment Problem (QAP).

The QAP assigns facilities to locations to minimize cost (flow x distance).
The representation is a permutation.

## Running

From the `build/smp/tutorial/Lesson1` directory:

```shell
./lesson1_eoEasyEA lesson1_eoEasyEA.param
```

## How it works

The key idea: ParadisEO-SMP parallelizes evaluation by wrapping an existing
sequential algorithm. The algorithm code itself stays the same. `QAP.h`
defines the problem representation and evaluation, `QAPGA.h` defines the GA
operators, and `parserStruct.h` + `utils.h` handle parameter parsing and
instance loading. See also `lesson1_data.dat` for the problem instance.
