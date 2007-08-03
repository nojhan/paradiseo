// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moGenSolContinue.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moGenSolContinue_h
#define __moGenSolContinue_h

#include "moSolContinue.h"

//! One possible stopping criterion for a solution-based heuristic.
/*!
  The stopping criterion corresponds to a maximum number of iteration.
 */
template < class EOT > class moGenSolContinue:public moSolContinue < EOT >
{

public:

  //! Basic constructor.
  /*!
     \param __maxNumGen the maximum number of generation.
   */
  moGenSolContinue (unsigned int __maxNumGen):maxNumGen (__maxNumGen), numGen (0)
  {}

  //! Function that activates the stop criterion.
  /*!
     Increments the counter and returns true if the
     current number of iteration is lower than the given
     maximum number of iterations.

     \param __sol the current solution.
     \return true or false according to the current generation number.
   */
  bool operator () (const EOT & __sol)
  {
    return (++numGen < maxNumGen);
  }

  //! Procedure which allows to initialise all the stuff needed.
  /*!
    It can be also used to reinitialize the counter all the needed things.
  */
  void init ()
  {
    numGen=0;
  }

private:

  //! Iteration maximum number.
  unsigned int maxNumGen;

  //! Iteration current number.
  unsigned int numGen;
};

#endif
