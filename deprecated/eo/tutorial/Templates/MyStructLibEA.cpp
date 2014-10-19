/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

The above line is usefulin Emacs-like editors
 */

/*
Template for creating a new representation in EO
================================================

This is the template main file for compiling after creating a
"library", i.e. putting everything but the fitness in a separate file
(make_MyStruct.cpp) and compiling it once and for all.
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

/** definition of initilizqtion:
 * class eoMyStructInit MUST derive from eoInit<eoMyStruct>
 */
#include "eoMyStructInit.h"

/** definition of evaluation:
 * class eoMyStructEvalFunc MUST derive from eoEvalFunc<eoMyStruct>
 * and should test for validity before doing any computation
 * see tutorial/Templates/evalFunc.tmpl
 */
#include "eoMyStructEvalFunc.h"

// GENOTYPE   eoMyStruct ***MUST*** be templatized over the fitness

//
// START fitness type: double or eoMaximizingFitness if you are maximizing
//                     eoMinimizingFitness if you are minimizing
typedef eoMinimizingFitness MyFitT ;	// type of fitness
// END fitness type
//

// Then define your EO objects using that fitness type
typedef eoMyStruct<MyFitT> Indi;      // ***MUST*** derive from EO

// create an initializer - done here and NOT in make_MyStruct.cpp
// because it is NOT representation independent
#include "make_genotype_MyStruct.h"
eoInit<Indi> & make_genotype(eoParser& _parser, eoState&_state, Indi _eo)
{
  return do_make_genotype(_parser, _state, _eo);
}

// same thing for the variation operaotrs
#include "make_op_MyStruct.h"
eoGenOp<Indi>&  make_op(eoParser& _parser, eoState& _state, eoInit<Indi>& _init)
{
  return do_make_op(_parser, _state, _init);
}

// The representation independent routines are simply declared here

// how to initialize the population
// it IS representation independent if an eoInit is given
eoPop<Indi >&  make_pop(eoParser& _parser, eoState& _state, eoInit<Indi> & _init);

// the stopping criterion
eoContinue<Indi>& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<Indi> & _eval);

// outputs (stats, population dumps, ...)
eoCheckPoint<Indi>& make_checkpoint(eoParser& _parser, eoState& _state, eoEvalFuncCounter<Indi>& _eval, eoContinue<Indi>& _continue);

// evolution engine (selection and replacement)
eoAlgo<Indi>&  make_algo_scalar(eoParser& _parser, eoState& _state, eoEvalFunc<Indi>& _eval, eoContinue<Indi>& _continue, eoGenOp<Indi>& _op);

// simple call to the algo. stays there for consistency reasons
// no template for that one
void run_ea(eoAlgo<Indi>& _ga, eoPop<Indi>& _pop);

// checks for help demand, and writes the status file
// and make_help; in libutils - just a declaration, code in libeoutils.a
void make_help(eoParser & _parser);

// now use all of the above, + representation dependent things
// from here on, no difference with eoMyStruct.cpp
int main(int argc, char* argv[])
{

  try
  {
  eoParser parser(argc, argv);  // for user-parameter reading

  eoState state;    // keeps all things allocated

    // The fitness
    //////////////
   eoMyStructEvalFunc<Indi> plainEval/* (varType  _anyVariable) */;
   // turn that object into an evaluation counter
   eoEvalFuncCounter<Indi> eval(plainEval);

  // the genotype - through a genotype initializer
  eoInit<Indi>& init = make_genotype(parser, state, Indi());

  // Build the variation operator (any seq/prop construct)
  eoGenOp<Indi>& op = make_op(parser, state, init);


  //// Now the representation-independent things
  //
  // YOU SHOULD NOT NEED TO MODIFY ANYTHING BEYOND THIS POINT
  // unless you want to add specific statistics to the checkpoint
  //////////////////////////////////////////////

  // initialize the population
  // yes, this is representation indepedent once you have an eoInit
  eoPop<Indi>& pop   = make_pop(parser, state, init);

  // stopping criteria
  eoContinue<Indi> & term = make_continue(parser, state, eval);
  // output
  eoCheckPoint<Indi> & checkpoint = make_checkpoint(parser, state, eval, term);
  // algorithm (need the operator!)
  eoAlgo<Indi>& ga = make_algo_scalar(parser, state, eval, checkpoint, op);

  ///// End of construction of the algorithm

  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(parser);

  //// GO
  ///////
  // evaluate intial population AFTER help and status in case it takes time
  apply<Indi>(eval, pop);
  // if you want to print it out
   cout << "Initial Population\n";
   pop.sortedPrintOn(cout);
   cout << endl;

  run_ea(ga, pop); // run the ga

  cout << "Final Population\n";
  pop.sortedPrintOn(cout);
  cout << endl;

  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
  return 0;
}
