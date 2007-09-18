// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoGenSolContinue.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moGenSolContinue_h
#define __moGenSolContinue_h

#include "moSolContinue.h"

//! One possible stop criterion for a solution-based heuristic.
/*!
  The stop criterion corresponds to a maximum number of iteration.
 */
template < class EOT > class moGenSolContinue:public moSolContinue < EOT >
{

public:

  //! Simple constructor.
  /*!
     \param __maxNumGen the maximum number of generation.
   */
  moGenSolContinue (unsigned int __maxNumGen):maxNumGen (__maxNumGen), numGen (0)
  {

  }

  //! Function that activates the stop criterion.
  /*!
     Increments the counter and returns TRUE if the
     current number of iteration is lower than the given
     maximum number of iterations.

     \param __sol the current solution.
     \return TRUE or FALSE according to the current generation number.
   */
  bool operator   () (const EOT & __sol)
  {

    return (++numGen < maxNumGen);
  }

  //! Procedure which allows to initialise the generation counter.
  /*!
     It can also be used to reset the iteration counter.
   */
  void init ()
  {

    numGen = 0;
  }

private:

  //! Iteration maximum number.
  unsigned int maxNumGen;

  //! Iteration current number.
  unsigned int numGen;
};

#endif
