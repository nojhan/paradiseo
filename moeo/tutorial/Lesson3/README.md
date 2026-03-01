# User-friendly multi-objective EA

Same flow-shop problem as Lesson 2, but with a complete algorithmic pipeline
built from reusable `do_make_*` helpers: stopping criteria, checkpointing,
statistics, and the full evolution engine.

## Running

From the `build/moeo/tutorial/Lesson3` directory:

```shell
./FlowShopEA2 @FlowShopEA2.param
```

## How it works

The main program is essentially a sequence of factory calls:

```c++
eoEvalFuncCounter<FlowShop>& eval = do_make_eval(parser, state);
eoInit<FlowShop>& init = do_make_genotype(parser, state);
eoPop<FlowShop>& pop = do_make_pop(parser, state, init);
eoContinue<FlowShop>& term = do_make_continue_moeo(parser, state, eval);
eoCheckPoint<FlowShop>& checkpoint = do_make_checkpoint_moeo(parser, state, ...);
eoAlgo<FlowShop>& algo = do_make_ea_moeo(parser, state, ...);
do_make_run(parser, state, algo, pop);
```

`do_make_continue_moeo` builds continuators (generation limit, fitness
target, etc.) from parameters. `do_make_checkpoint_moeo` adds statistics,
population snapshots, and archive tracking. `do_make_ea_moeo` constructs
the selection and replacement strategy.
