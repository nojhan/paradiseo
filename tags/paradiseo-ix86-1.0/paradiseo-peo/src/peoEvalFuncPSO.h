// "peoEvalFuncPSO.h"

// (c) OPAC Team, October 2007

/* 
   Contact: clive.canape@inria.fr
*/

#ifndef PEOEVALFUNCPSO_H
#define PEOEVALFUNCPSO_H

#include <eoEvalFunc.h>

/** peoEvalFuncPSO: This class
 * takes an existing function pointer and converts it into a evaluation
 * function class. 
 */

#ifdef _MSC_VER
template< class POT, class FitT = POT::Fitness, class FunctionArg = const POT& >
#else
template< class POT, class FitT = typename POT::Fitness, class FunctionArg = const POT& >
#endif
struct peoEvalFuncPSO: public eoEvalFunc<POT> {

  peoEvalFuncPSO( FitT (* _eval)( FunctionArg ) )
    : eoEvalFunc<POT>(), evalFunc( _eval ) {};
  
    //Applies the evaluation function to a PEO
  virtual void operator() ( POT & _peo ) 
  {
      _peo.fitness((*evalFunc)( _peo ));
  };
    
  private:
    FitT (* evalFunc )( FunctionArg );
};

#endif

