/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

The above line is usefulin Emacs-like editors
 */

/*
Template for creating a new representation in EO
================================================

This is the template file that allows separate compilation of
everything that is representation independant (evolution engine and
general output) for an Evolutionary Algorithm with scalar fitness.

It includes of course the definition of the genotype (eoMyStruct.h) and
is written like the make_xxx.cpp files in dirs src/ga (for bitstrings)
and src/es (for real vectors).

*/

// Miscilaneous include and declaration
#include <iostream>
using namespace std;

// eo general include
#include "eo"
// the real bounds (not yet in general eo include)
#include "utils/eoRealVectorBounds.h"

// include here whatever specific files for your representation
// Basically, this should include at least the following

/** definition of representation:
 * class eoMyStruct MUST derive from EO<FitT> for some fitness
 */
#include "eoMyStruct.h"

// create an initializer: this is NOT representation-independent
// and will be done in the main file
// However, should you decide to freeze that part, you could use the
// following (and remove it from the main file, of course!!!)
//------------------------------------------------------------------
// #include "make_genotype_MyStruct.h"
// eoInit<eoMyStruct<double>> & make_genotype(eoParser& _parser, eoState&_state, eoMyStruct<double> _eo)
// {
//   return do_make_genotype(_parser, _state, _eo);
// }

// eoInit<eoMyStruct<eoMinimizingFitness>> & make_genotype(eoParser& _parser, eoState&_state, eoMyStruct<eoMinimizingFitness> _eo)
// {
//   return do_make_genotype(_parser, _state, _eo);
// }

// same thing for the variation operaotrs
//---------------------------------------
// #include "make_op_MyStruct.h"
// eoGenOp<eoMyStruct<double>>&  make_op(eoParser& _parser, eoState& _state, eoInit<eoMyStruct<double>>& _init)
// {
//   return do_make_op(_parser, _state, _init);
// }

// eoGenOp<eoMyStruct<eoMinimizingFitness>>&  make_op(eoParser& _parser, eoState& _state, eoInit<eoMyStruct<eoMinimizingFitness>>& _init)
// {
//   return do_make_op(_parser, _state, _init);
// }

// The following modules use ***representation independent*** routines

// how to initialize the population
// it IS representation independent if an eoInit is given
#include <make_pop.h>
eoPop<eoMyStruct<double> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoMyStruct<double> > & _init)
{
  return do_make_pop(_parser, _state, _init);
}

eoPop<eoMyStruct<eoMinimizingFitness> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoMyStruct<eoMinimizingFitness> > & _init)
{
  return do_make_pop(_parser, _state, _init);
}

// the stopping criterion
#include <make_continue.h>
eoContinue<eoMyStruct<double> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoMyStruct<double> > & _eval)
{
  return do_make_continue(_parser, _state, _eval);
}

eoContinue<eoMyStruct<eoMinimizingFitness> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoMyStruct<eoMinimizingFitness> > & _eval)
{
  return do_make_continue(_parser, _state, _eval);
}

// outputs (stats, population dumps, ...)
#include <make_checkpoint.h>
eoCheckPoint<eoMyStruct<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoMyStruct<double> >& _eval, eoContinue<eoMyStruct<double> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}

eoCheckPoint<eoMyStruct<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoMyStruct<eoMinimizingFitness> >& _eval, eoContinue<eoMyStruct<eoMinimizingFitness> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}

// evolution engine (selection and replacement)
#include <make_algo_scalar.h>
eoAlgo<eoMyStruct<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoMyStruct<double> >& _eval, eoContinue<eoMyStruct<double> >& _continue, eoGenOp<eoMyStruct<double> >& _op)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

eoAlgo<eoMyStruct<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoMyStruct<eoMinimizingFitness> >& _eval, eoContinue<eoMyStruct<eoMinimizingFitness> >& _continue, eoGenOp<eoMyStruct<eoMinimizingFitness> >& _op)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

// simple call to the algo. stays there for consistency reasons
// no template for that one
#include <make_run.h>
void run_ea(eoAlgo<eoMyStruct<double> >& _ga, eoPop<eoMyStruct<double> >& _pop)
{
  do_run(_ga, _pop);
}

void run_ea(eoAlgo<eoMyStruct<eoMinimizingFitness> >& _ga, eoPop<eoMyStruct<eoMinimizingFitness> >& _pop)
{
  do_run(_ga, _pop);
}
