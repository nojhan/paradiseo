// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moSA.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moSA_h
#define __moSA_h

#include <eoOp.h>
#include <eoEvalFunc.h>

#include "moAlgo.h"
#include "moRandMove.h"
#include "moMoveIncrEval.h"
#include "moCoolingSchedule.h"
#include "moSolContinue.h"

#include <math.h>

//! Simulated Annealing (SA)
/*!
  Class that describes a Simulated Annealing algorithm.
*/
template < class M > class moSA:public moAlgo < typename M::EOType >
{

  //! Alias for the type
  typedef typename  M::EOType  EOT;

  //! Alias for the fitness
  typedef typename  EOT::Fitness  Fitness;

public:

  //! SA constructor
  /*!
     All the boxes used by a SA need to be given.

     \param __move_rand a move generator (generally randomly).
     \param __incr_eval a (generaly) efficient evaluation function 
     \param __cont a stopping criterion.
     \param __init_temp the initial temperature.
     \param __cool_sched a cooling schedule, describes how the temperature is modified.
     \param __full_eval a full evaluation function.
   */
  moSA (moRandMove < M > &__move_rand,
	moMoveIncrEval < M > &__incr_eval,
	moSolContinue < EOT > &__cont,
	double __init_temp,
	moCoolingSchedule & __cool_sched, eoEvalFunc < EOT > &__full_eval):
  move_rand (__move_rand),
  incr_eval (__incr_eval),
  cont (__cont),
  init_temp (__init_temp),
  cool_sched (__cool_sched),
  full_eval (__full_eval)
  {

  }

  //! function that launches the SA algorithm.
  /*!
     As a moTS or a moHC, the SA can be used for HYBRIDATION in an evolutionary algorithm.

     \param __sol a solution to improve.
     \return TRUE.
   */
  bool operator   ()(EOT & __sol)
  {

    if (__sol.invalid ())
      {
	full_eval (__sol);
      }

    double temp = init_temp;

    M move;

    EOT best_sol = __sol;

    Fitness current_fitness, delta;
    double exp1, exp2;

    do
      {
	cont.init ();
	do
	  {
	    move_rand (move);

	    current_fitness= incr_eval (move, __sol);

	    delta = current_fitness - __sol.fitness();

	    if(((long double)delta) < 0.0)
	      {
		delta=-delta;
	      }

	    if ((current_fitness > __sol.fitness()) || ((rng.uniform ()) < (exp (-delta/ temp))))
	      {
		__sol.fitness (current_fitness);
		move (__sol);
		
		/* Updating the best solution found  until now ? */
		if (__sol.fitness () > best_sol.fitness ())
		  {
		    best_sol = __sol;
		  }
	      }
	  }
	while (cont (__sol));
      }
    while (cool_sched (temp));

    __sol = best_sol;

    return true;
  }

private:

  //! A move generator (generally randomly)
  moRandMove < M > &move_rand;

  //! A (generally) efficient evaluation function.
  moMoveIncrEval < M > &incr_eval;

  //! Stopping criterion before temperature update
  moSolContinue < EOT > &cont;

  //! Initial temperature
  double  init_temp;

  //! The cooling schedule
  moCoolingSchedule & cool_sched;

  //! A full evaluation function.
  eoEvalFunc < EOT > &full_eval;	// Full evaluator.
};

#endif
