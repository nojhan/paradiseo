// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moFitSolContinue.h"

// (c) OPAC Team (LIFL), Dolphin project (INRIA), 2003-2007

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moFitSolContinue_h
#define __moFitSolContinue_h

#include "moSolContinue.h"

//! One possible stop criterion for a solution-based heuristic.
/*!
  The stop criterion corresponds to a fitness threshold gained.
 */
template < class EOT > class moFitSolContinue:public moSolContinue < EOT >
{

public:

  //! Alias for the fitness.
  typedef typename EOT::Fitness Fitness;

  //! Basic constructor.
  /*!
     \param __fitness The fitness to reach.
     \param __maximization Indicate if the the aim is to maximize or minimize the fitness.
   */
  moFitSolContinue (Fitness __fitness, bool __maximization=true): fitness (__fitness), maximization(__maximization)
  {}

  //! Function that activates the stopping criterion.
  /*!
    Indicates if the fitness threshold has not been yet reached.

     \param __sol the current solution.
     \return true or false according to the value of the fitness.
   */
  bool operator   () (const EOT & __sol)
  {
    if(__sol.invalid())
      {
	return true;
      }

    if(maximization)
      {
	return __sol.fitness()<fitness;
      }
    return __sol.fitness()>fitness;
  }

  //! Procedure which allows to initialise all the stuff needed.
  void init ()
  {}

private:

  //! Fitness target.
  Fitness fitness;

  //! Flag that indicate if there is a maximization (true) or a minimization (false) of the fitness value.
  /*!
    It can be interesting to know this information because some solution-based metaheuristics can generate solution with a fitness that
    is worse that the best known fitness (in this case, the counter is not reinitialized).
   */
  bool maximization;
};

#endif
