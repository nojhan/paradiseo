/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

The above line is usefulin Emacs-like editors
 */

/*
Template for creating a new representation in EO
================================================

This is the template file that llows separate compilation of
everything that is representation independant (evolution engine and
general output) for an Evolutionary Algorithm with scalar fitness.

It includes of course the definition of the genotype (eoOneMax.h) and
is written like the make_xxx.cpp files in dirs src/ga (for bitstrings)
and src/es (for real vectors).

*/

// Miscilaneous include and declaration
#include <iostream>
using namespace std;

// eo general include
#include <paradiseo/eo.h>
// the real bounds (not yet in general eo include)
#include <paradiseo/eo/utils/eoRealVectorBounds.h>

// include here whatever specific files for your representation
// Basically, this should include at least the following

/** definition of representation:
 * class eoOneMax MUST derive from EO<FitT> for some fitness
 */
#include "eoOneMax.h"

// create an initializer: this is NOT representation-independent
// and will be done in the main file
// However, should you decide to freeze that part, you could use the
// following (and remove it from the main file, of course!!!)
//------------------------------------------------------------------
// #include "make_genotype_OneMax.h"
// eoInit<eoOneMax<double>> & make_genotype(eoParser& _parser, eoState&_state, eoOneMax<double> _eo)
// {
//   return do_make_genotype(_parser, _state, _eo);
// }

// eoInit<eoOneMax<eoMinimizingFitness>> & make_genotype(eoParser& _parser, eoState&_state, eoOneMax<eoMinimizingFitness> _eo)
// {
//   return do_make_genotype(_parser, _state, _eo);
// }

// same thing for the variation operaotrs
//---------------------------------------
// #include "make_op_OneMax.h"
// eoGenOp<eoOneMax<double>>&  make_op(eoParser& _parser, eoState& _state, eoInit<eoOneMax<double>>& _init)
// {
//   return do_make_op(_parser, _state, _init);
// }

// eoGenOp<eoOneMax<eoMinimizingFitness>>&  make_op(eoParser& _parser, eoState& _state, eoInit<eoOneMax<eoMinimizingFitness>>& _init)
// {
//   return do_make_op(_parser, _state, _init);
// }

// The following modules use ***representation independent*** routines

// how to initialize the population
// it IS representation independent if an eoInit is given
#include <paradiseo/eo/do/make_pop.h>
eoPop<eoOneMax<double> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoOneMax<double> > & _init)
{
  return do_make_pop(_parser, _state, _init);
}

eoPop<eoOneMax<eoMinimizingFitness> >&  make_pop(eoParser& _parser, eoState& _state, eoInit<eoOneMax<eoMinimizingFitness> > & _init)
{
  return do_make_pop(_parser, _state, _init);
}

// the stopping criterion
#include <paradiseo/eo/do/make_continue.h>
eoContinue<eoOneMax<double> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoOneMax<double> > & _eval)
{
  return do_make_continue(_parser, _state, _eval);
}

eoContinue<eoOneMax<eoMinimizingFitness> >& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoOneMax<eoMinimizingFitness> > & _eval)
{
  return do_make_continue(_parser, _state, _eval);
}

// outputs (stats, population dumps, ...)
#include <paradiseo/eo/do/make_checkpoint.h>
eoCheckPoint<eoOneMax<double> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoOneMax<double> >& _eval, eoContinue<eoOneMax<double> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}

eoCheckPoint<eoOneMax<eoMinimizingFitness> >& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<eoOneMax<eoMinimizingFitness> >& _eval, eoContinue<eoOneMax<eoMinimizingFitness> >& _continue)
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}

// evolution engine (selection and replacement)
#include <paradiseo/eo/do/make_algo_scalar.h>
eoAlgo<eoOneMax<double> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoOneMax<double> >& _eval, eoContinue<eoOneMax<double> >& _continue, eoGenOp<eoOneMax<double> >& _op)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

eoAlgo<eoOneMax<eoMinimizingFitness> >&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<eoOneMax<eoMinimizingFitness> >& _eval, eoContinue<eoOneMax<eoMinimizingFitness> >& _continue, eoGenOp<eoOneMax<eoMinimizingFitness> >& _op)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

// simple call to the algo. stays there for consistency reasons
// no template for that one
#include <paradiseo/eo/do/make_run.h>
void run_ea(eoAlgo<eoOneMax<double> >& _ga, eoPop<eoOneMax<double> >& _pop)
{
  do_run(_ga, _pop);
}

void run_ea(eoAlgo<eoOneMax<eoMinimizingFitness> >& _ga, eoPop<eoOneMax<eoMinimizingFitness> >& _pop)
{
  do_run(_ga, _pop);
}
