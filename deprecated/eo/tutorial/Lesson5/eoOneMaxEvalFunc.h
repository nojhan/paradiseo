/** -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

The above line is usefulin Emacs-like editors
 */

/*
Template for evaluator in EO, a functor that computes the fitness of an EO
==========================================================================
*/

#ifndef _eoOneMaxEvalFunc_h
#define _eoOneMaxEvalFunc_h

// include whatever general include you need
#include <stdexcept>
#include <fstream>

// include the base definition of eoEvalFunc
#include "eoEvalFunc.h"

/**
  Always write a comment in this format before class definition
  if you want the class to be documented by Doxygen
*/
template <class EOT>
class eoOneMaxEvalFunc : public eoEvalFunc<EOT>
{
public:
	/// Ctor - no requirement
// START eventually add or modify the anyVariable argument
  eoOneMaxEvalFunc()
  //  eoOneMaxEvalFunc( varType  _anyVariable) : anyVariable(_anyVariable)
// END eventually add or modify the anyVariable argument
  {
    // START Code of Ctor of an eoOneMaxEvalFunc object
    // END   Code of Ctor of an eoOneMaxEvalFunc object
  }

  /** Actually compute the fitness
   *
   * @param EOT & _eo the EO object to evaluate
   *                  it should stay templatized to be usable
   *                  with any fitness type
   */
  void operator()(EOT & _eo)
  {
    // test for invalid to avoid recomputing fitness of unmodified individuals
    if (_eo.invalid())
      {
	double fit;		   // to hold fitness value
    // START Code of computation of fitness of the eoOneMax object
	const vector<bool> & b = _eo.B();
	fit = 0;
	for (unsigned i=0; i<b.size(); i++)
	  fit += (b[i]?0:1);
    // END   Code of computation of fitness of the eoOneMax object
	_eo.fitness(fit);
      }
  }

private:
// START Private data of an eoOneMaxEvalFunc object
  //  varType anyVariable;		   // for example ...
// END   Private data of an eoOneMaxEvalFunc object
};


#endif
