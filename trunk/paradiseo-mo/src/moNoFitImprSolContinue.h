// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moNoFitImprSolContinue.h"

// (c) OPAC Team (LIFL), Dolphin project (INRIA), 2003-2007

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moNoFitImprSolContinue_h
#define __moNoFitImprSolContinue_h

#include "moSolContinue.h"

//! One possible stop criterion for a solution-based heuristic.
/*!
  The stop criterion corresponds to a maximum number of iterations without improevement.
 */
template < class EOT > class moNoFitImprSolContinue:public moSolContinue < EOT >
{

public:

  //! Alias for the fitness.
  typedef typename EOT::Fitness Fitness;

  //! Basic constructor.
  /*!
     \param __maxNumberOfIterationWithoutImprovement The number of iterations without fitness improvement to reach for stop.
   */
  moNoFitImprSolContinue (unsigned int __maxNumberOfIterationWithoutImprovement)
    : maxNumberOfIterationsWithoutImprovement(__maxNumberOfIterationWithoutImprovement), firstFitnessSaved(true), counter(0) 
  {}

  //! Function that activates the stopping criterion.
  /*!
    Indicates if the fitness has not been improved since a given number of iterations (after a minimum of iterations).
    \param __sol the current solution.
    \return true or false.
   */
  bool operator   () (const EOT & __sol)
  {
    if(__sol.invalid())
      {
	return true;
      }

    if(firstFitnessSaved)
      {
	fitness=__sol.fitness();
	counter=0;
	firstFitnessSaved=false;
	return true;
      }
    
    counter++;

    if( __sol.fitness() > fitness)
      {
	fitness=__sol.fitness();
	counter=0;
      }
  
    if(counter==maxNumberOfIterationsWithoutImprovement)
      {
	std::cout << "moNoFitImrpSolContinue: Done [" << counter  << "] iterations without improvement." << std::endl;
      }
    return counter!=maxNumberOfIterationsWithoutImprovement;
  }

  //! Procedure which allows to initialise all the stuff needed.
  /*!
    It can be also used to reinitialize the counter all the needed things.
  */
  void init ()
  {
    firstFitnessSaved=true;
    counter=0;
  }

private:

  //! Maximum number of iterations without improvement allowed.
  unsigned int maxNumberOfIterationsWithoutImprovement;

  //! Flag that this is the first time that the fitness is used.
  bool firstFitnessSaved;

  //! Current Fitness.
  Fitness fitness;

  //! The iteration couter.
  unsigned int counter;
};

#endif
