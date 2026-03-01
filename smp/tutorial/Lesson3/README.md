# Heterogeneous island model

Two islands running different algorithms: an `eoEasyEA` on the QAP problem
and a PSO on a bitstring problem, connected via conversion functions.

## Running

From the `build/smp/tutorial/Lesson3` directory:

```shell
./lesson3_heterogeneous
```

## How it works

When islands use different representations, you need conversion functions
to translate individuals during migration. In this example, `fromBase()`
and `toBase()` convert between the QAP permutation and the PSO bitstring:

```c++
Indi2 fromBase(Indi& i, unsigned size) { ... }
Indi toBase(Indi2& i) { ... }
```

The conversions here are dummy placeholders (they just create random
individuals), but in a real application you would implement meaningful
mappings between representations.
