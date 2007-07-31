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
  The stop criterion corresponds to a maximum number of iterations without improvement.
 */
template < class EOT > class moNoFitImprSolContinue:public moSolContinue < EOT >
{

public:

  //! Alias for the fitness.
  typedef typename EOT::Fitness Fitness;

  //! Basic constructor.
  /*!
     \param __maxNumberOfIterationWithoutImprovment The number of iterations without fitness improvment to reach for stop.
     \param __maximization Indicate if the the aim is to maximize or minimize the fitness.
   */
  moNoFitImprSolContinue (unsigned int __maxNumberOfIterationWithoutImprovment, bool __maximization=true)
    : maxNumberOfIterationsWithoutImprovment(__maxNumberOfIterationWithoutImprovment),maximization(__maximization), 
      firstFitnessSaved(true), counter(0) 
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
	FirstFitnessSaved=false;
	return true;
      }
    
    counter++;

    if( ((maximization) && (__sol.fitness() > fitness)) || 
	((!maximization) && (__sol.fitness() < fitness)) )
      {
	fitness=__sol.fitness();
	counter=0;
      }
  
    if(counter==maxNumberOfIterationsWithoutImprovment)
      {
	std::cout << "moNoFitImrpSolContinue: Done [" << counter  << "] iterations without improvement." << std::endl;
      }
    return counter!=maxNumberOfIterationsWithoutImprovment;
  }

  //! Procedure which allows to initialise all the stuff needed.
  void init ()
  {}

private:

  //! Maximum number of iterations without improvment allowed.
  unsigned int maxNumberOfIterationsWithoutImprovment;

  //! Flag that this is the first time that the fitness is used.
  bool firstFitnessSaved;

  //! Current Fitness.
  Fitness fitness;

  //! Flag that indicate if there is a maximization (true) or a minimization (false) of the fitness value.
  /*!
    It can be interesting to know this information because some solution-based metaheuristics can generate solutions wiht a fitness that
    is worse that the best known fitness (in this case, the counter is not reinitialized).
   */
  bool maximization;

  //! The iteration couter.
  unsigned int counter;
};

#endif
