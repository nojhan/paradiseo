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

/** definition of mutation: 
 * class eoMyStructMonop MUST derive from eoMonOp<eoMyStruct>
 */
#include "eoMyStructMutation.h"

/** definition of crossover (either as eoBinOp (2->1) or eoQuadOp (2->2): 
 * class eoMyStructBinCrossover MUST derive from eoBinOp<eoMyStruct>
 * OR 
 * class eoMyStructQuadCrossover MUST derive from eoQuadOp<eoMyStruct>
 */
// #include "eoMyStructBinOp.h"
// OR
#include "eoMyStructQuadCrossover.h"

/** definition of evaluation: 
 * class eoMyStructEvalFunc MUST derive from eoEvalFunc<eoMyStruct>
 * and should test for validity before doing any computation
 * see tutorial/Templates/evalFunc.tmpl
 */
#include "eoMyStructEvalFunc.h"

// GENOTYPE   eoMyStruct ***MUST*** be templatized over the fitness

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
// START fitness type: double or eoMaximizingFitness if you are maximizing
//                     eoMinimizingFitness if you are minimizing
typedef eoMyStruct<double> Indi;      // ***MUST*** derive from EO 
// END fitness type
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


// Use existing modules to define representation independent routines
// These are parser-based definitions of objects

// how to initialize the population 
// it IS representation independent if an eoInit is given
#include <do/make_pop.h>
eoPop<Indi >&  make_pop(eoParser& _parser, eoState& _state, eoInit<Indi> & _init)
{
  return do_make_pop(_parser, _state, _init);
}

// the stopping criterion
#include <do/make_continue.h>
eoContinue<Indi>& make_continue(eoParser& _parser, eoState& _state, eoEvalFuncCounter<Indi> & _eval)
{
  return do_make_continue(_parser, _state, _eval);
}

// outputs (stats, population dumps, ...)
#include <do/make_checkpoint.h>
eoCheckPoint<Indi>& make_checkpoint(eoParameterLoader& _parser, eoState& _state, eoEvalFuncCounter<Indi>& _eval, eoContinue<Indi>& _continue) 
{
  return do_make_checkpoint(_parser, _state, _eval, _continue);
}

// evolution engine (selection and replacement)
#include <do/make_algo_scalar.h>
eoAlgo<Indi>&  make_algo_scalar(eoParameterLoader& _parser, eoState& _state, eoEvalFunc<Indi>& _eval, eoContinue<Indi>& _continue, eoGenOp<Indi>& _op)
{
  return do_make_algo_scalar(_parser, _state, _eval, _continue, _op);
}

// simple call to the algo. stays there for consistency reasons 
// no template for that one
#include <do/make_run.h>
// the instanciating fitnesses
#include <eoScalarFitness.h>
void run_ea(eoAlgo<Indi>& _ga, eoPop<Indi>& _pop)
{
  do_run(_ga, _pop);
}

// checks for help demand, and writes the status file
// and make_help; in libutils
void make_help(eoParser & _parser);

// now use all of the above, + representation dependent things
int main(int argc, char* argv[])
{

  try
  {
  eoParser parser(argc, argv);  // for user-parameter reading

  eoState state;    // keeps all things allocated

  ///// FIRST, problem or representation dependent stuff
  //////////////////////////////////////////////////////

    //////////////////////////////////////////////
  // the genotype - through a genotype initializer

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
// START read parameters (if any) and create the initializer directly

  // As far as memory management is concerned, the parameters 
  //    AND the values are owned by the parser (handled through references)

  // example of parameter reading using the most compact syntax
  // varType var = parser.createParam(varType defaultValue, 
  //                                  string keyword, 
  //                                  string comment,
  //                                  char shortKeyword, 
  //                                  string section,).value();

   // an unsigned parameter
    unsigned sampleUnsigned = parser.createParam(unsigned(10), "anUnsigned", "An unsigned parameter",'V', "Representation").value();

   // a double parameter
    double sampleDouble = parser.createParam(0.3, "aDouble", "A double parameter", 's', "Representation" ).value();

    // some real bounds: [-1,1]x[-1,1] by default
  eoRealVectorBounds & sampleBounds = parser.createParam(eoRealVectorBounds(2,-1,1), "someBounds", "Bounds of some real variables", 'B', "Representation").value();
// END   read parameters (if any) and create the initializer directly
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
// START Modify definitions of objects by eventually add parameters

  /**  HINTS 
   *
   * The following declare variables that are objects defined 
   * in the customized files.
   * You shoudl only modify the arguments passed onto their constructors
   * ("varType  _anyVariable") in their definition
   *
   * and you can optionally uncomment and modify the lines commented between
   *    /* and */

   // the initializer: will be used in make_pop
  ////////////////////
  eoMyStructInit<Indi> init/* (varType  _anyVariable) */;

    // The fitness
    //////////////
   eoMyStructEvalFunc<Indi> plainEval/* (varType  _anyVariable) */;
   // turn that object into an evaluation counter
   eoEvalFuncCounter<Indi> eval(plainEval);

    /////////////////////////////
    // Variation operators
    ////////////////////////////
    // read crossover and mutations, combine each in a proportional Op
    // and create the eoGenOp that calls crossover at rate pCross 
    // then mutation with rate pMut

    // the crossovers
    /////////////////

    // here we can have eoQuadOp (2->2) only - no time for the eoBinOp case

    // you can have more than one - combined in a proportional way
    
    // first, define the crossover objects and read their rates from the parser
    
    // A first crossover   
    eoMyStructQuadCrossover<Indi> cross1/* (varType  _anyVariable) */;
  // its relative rate in the combination
    double cross1Rate = parser.createParam(1.0, "cross1Rate", "Relative rate for crossover 1", '1', "Variation Operators").value();
  // and the creation of the combined operator with this one
  eoPropCombinedQuadOp<Indi> propXover(cross1, cross1Rate);

    // Optional: A second(and third, and ...)  crossover   
    //   of course you must create the corresponding classes
    // and all ***MUST*** derive from eoQuadOp<Indi>

  /* Uncomment if necessary - and replicate as many time as you need
      eoMyStructSecondCrossover<Indi> cross2(varType  _anyVariable); 
          double cross2Rate = parser.createParam(1.0, "cross2Rate", "Relative rate for crossover 2", '2', "Variation Operators").value(); 
      propXover.add(cross2, cross2Rate); 
  */
  // if you want some gentle output, the last one shoudl be like
  //  propXover.add(crossXXX, crossXXXRate, true);


  // the mutation: same story
  ////////////////
  // you can have more than one - combined in a proportional way

  // for each mutation, 
  // - define the mutator object
  // - read its rate from the parser
  // - add it to the proportional combination

  // a first mutation  
  eoMyStructMutation<Indi> mut1/* (varType  _anyVariable) */;
  // its relative rate in the combination
  double mut1Rate = parser.createParam(1.0, "mut1Rate", "Relative rate for mutation 1", '1', "Variation Operators").value();
  // and the creation of the combined operator with this one
  eoPropCombinedMonOp<Indi> propMutation(mut1, mut1Rate);

    // Optional: A second(and third, and ...)  mutation with their rates
    //   of course you must create the corresponding classes
    // and all ***MUST*** derive from eoMonOp<Indi>

  /* Uncomment if necessary - and replicate as many time as you need
      eoMyStructSecondMutation<Indi> mut2(varType  _anyVariable);
      double mut2Rate = parser.createParam(1.0, "mut2Rate", "Relative rate for mutation 2", '2', "Variation Operators").value(); 
       propMutation.add(mut2, mut2Rate); 
  */
  // if you want some gentle output, the last one shoudl be like
  //  propMutation.add(mutXXX, mutXXXRate, true);

  // end of crossover and mutation definitions
  ////////////////////////////////////////////

// END Modify definitions of objects by eventually add parameters
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

// from now on, you do not need to modify anything
// though you CAN add things to the checkpointing (see tutorial)

  // now build the eoGenOp:
  // to simulate SGA (crossover with proba pCross + mutation with proba pMut
  // we must construct
  //     a sequential combination of
  //          with proba 1, a proportional combination of 
  //                        a QuadCopy and our crossover
  //          with proba pMut, our mutation

  // but of course you're free to use any smart combination you could think of
  // especially, if you have to use eoBinOp rather than eoQuad Op youùll have
  // to modify that part

  // First read the individual level parameters
    eoValueParam<double>& pCrossParam = parser.createParam(0.6, "pCross", "Probability of Crossover", 'C', "Variation Operators" );
    // minimum check
    if ( (pCrossParam.value() < 0) || (pCrossParam.value() > 1) )
      throw runtime_error("Invalid pCross");

    eoValueParam<double>& pMutParam = parser.createParam(0.1, "pMut", "Probability of Mutation", 'M', "Variation Operators" );
    // minimum check
    if ( (pMutParam.value() < 0) || (pMutParam.value() > 1) )
      throw runtime_error("Invalid pMut");


  // the crossover - with probability pCross
  eoProportionalOp<Indi> * cross = new eoProportionalOp<Indi> ;
  state.storeFunctor(cross);
  eoQuadOp<Indi> *ptQuad = new eoQuadCloneOp<Indi>;
  state.storeFunctor(ptQuad);
  cross->add(propXover, pCrossParam.value()); // crossover, with proba pcross
  cross->add(*ptQuad, 1-pCrossParam.value()); // nothing, with proba 1-pcross

  // now the sequential
  eoSequentialOp<Indi> *op = new eoSequentialOp<Indi>;
  state.storeFunctor(op);
  op->add(*cross, 1.0);	 // always do combined crossover
  op->add(propMutation, pMutParam.value()); // then mutation, with proba pmut

  // that's it! (beware op is a pointer - for lazy cut-and-paste reasons!

  // end of operator definition
  ////////////////////////////

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
  eoAlgo<Indi>& ga = make_algo_scalar(parser, state, eval, checkpoint, (*op));

  ///// End of construction of the algorithm

  /////////////////////////////////////////
  // to be called AFTER all parameters have been read!!!
  make_help(parser);

  //// GO
  ///////
  // evaluate intial population AFTER help and status in case it takes time
  apply(eval, pop);
  // if you want to print it out
//   cout << "Initial Population\n";
//   pop.sortedPrintOn(cout);
//   cout << endl;

  run_ea(ga, pop); // run the ga

  cout << "Final Population\n";
  pop.sortedPrintOn(cout);
  cout << endl;

  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
}
