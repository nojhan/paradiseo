// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moSteadyFitSolContinue.h"

// (c) OPAC Team (LIFL), Dolphin project (INRIA), 2003-2007

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moSteadyFitSolContinue_h
#define __moSteadyFitSolContinue_h

#include "moSolContinue.h"

//! One possible stopping criterion for a solution-based heuristic.
/*!
  The stop criterion corresponds to a maximum number of iterations without improvement (after a minimum number of iterations).
 */
template < class EOT > class moSteadyFitSolContinue:public moSolContinue < EOT >
{

public:

  //! Alias for the fitness.
  typedef typename EOT::Fitness Fitness;

  //! Basic constructor.
  /*!
     \param __maxNumberOfIterations The number of iterations to reach before looking for the fitness.
     \param __maxNumberOfIterationWithoutImprovement The number of iterations without fitness improvement to reach for stop.
   */
  moSteadyFitSolContinue (unsigned int __maxNumberOfIterations, unsigned int __maxNumberOfIterationWithoutImprovement)
    : maxNumberOfIterations (__maxNumberOfIterations), maxNumberOfIterationsWithoutImprovement(__maxNumberOfIterationWithoutImprovement),
      maxNumberOfIterationsReached(false), firstFitnessSaved(true), counter(0) 
  {}

  //! Function that activates the stopping criterion.
  /*!
    Indicates if the fitness has not been improved since a number of iterations (after a minimum of iterations).

     \param __sol the current solution.
     \return true or false.
   */
  bool operator   () (const EOT & __sol)
  {
    if(!maxNumberOfIterationsReached)
      {
	maxNumberOfIterationsReached=((++counter)==maxNumberOfIterations);
	if(maxNumberOfIterationsReached)
	  {
	    std::cout << "moSteadyFitSolContinue: Done the minimum number of iterations [" << counter << "]." << std::endl;
	  }
	return true;
      }

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

    if( __sol.fitness() > fitness )
      {
	fitness=__sol.fitness();
	counter=0;
      }
  
    if(counter==maxNumberOfIterationsWithoutImprovement)
      {
	std::cout << "moSteadyFitSolContinue: Done [" << counter  << "] iterations without improvement." << std::endl;
      }
    return counter!=maxNumberOfIterationsWithoutImprovement;
  }

  //! Procedure which allows to initialise the stuff needed.
  /*!
    It can be also used to reinitialize the counter all the needed things.
  */
  void init ()
  {
    maxNumberOfIterationsReached=false;
    counter=0;
    firstFitnessSaved=true;
  }

private:

  //! Maximum number of iterations before considering the fitness.
  unsigned int maxNumberOfIterations;

   //! Maximum number of iterations without improvement allowed.
  unsigned int maxNumberOfIterationsWithoutImprovement;

  //! Flag that indicates that the maxNumberIteration have been reached.
  bool maxNumberOfIterationsReached;

  //! Flag that this is the first time that the fitness is used.
  bool firstFitnessSaved;

  //! Current Fitness.
  Fitness fitness;

  //! The iteration couter.
  unsigned int counter;
};

#endif
