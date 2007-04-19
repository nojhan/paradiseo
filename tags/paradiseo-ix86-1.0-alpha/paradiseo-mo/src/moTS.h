// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moTS.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
      
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moTS_h
#define __moTS_h

#include <eoOp.h>
#include <eoEvalFunc.h>

#include "moAlgo.h"
#include "moSolContinue.h"

#include "moMoveExpl.h"
#include "moTSMoveLoopExpl.h"

#include <pthread.h>

//! Tabu Search (TS)
/*!
  Generic algorithm that describes a tabu search.
 */
template < class M > class moTS:public moAlgo < typename M::EOType >
{

  //!Alias for the type
  typedef
    typename
    M::EOType
    EOT;

  //!Alias for the fitness
  typedef
    typename
    EOT::Fitness
    Fitness;

public:

  //!Constructor of a moTS specifying all the boxes
  /*!
     In this constructor, a moTSMoveLoopExpl is instanciated.

     \param __move_init move initialisation
     \param __next_move neighborhood explorer
     \param __incr_eval efficient evaluation
     \param __tabu_list tabu list
     \param __aspir_crit aspiration criterion
     \param __cont stop criterion
     \param __full_eval full evaluation function
   */
moTS (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval, moTabuList < M > &__tabu_list, moAspirCrit < M > &__aspir_crit, moSolContinue < EOT > &__cont, eoEvalFunc < EOT > &__full_eval):move_expl (*new moTSMoveLoopExpl < M >
	     (__move_init, __next_move, __incr_eval, __tabu_list,
	      __aspir_crit)), cont (__cont), full_eval (__full_eval)
  {

    if (first_time)
      {

	pthread_mutex_init (&mutex, 0);

	first_time = false;
      }
  }

  //! Constructor with less parameters
  /*!
     The explorer is given in the parameters.

     \param __move_expl the explorer (generally different that a moTSMoveLoopExpl)
     \param __cont stop criterion
     \param __full_eval full evaluation function
   */
moTS (moMoveExpl < M > &__move_expl, moSolContinue < EOT > &__cont, eoEvalFunc < EOT > &__full_eval):move_expl (__move_expl),
    cont (__cont),
    full_eval (__full_eval)
  {

    if (first_time)
      {

	pthread_mutex_init (&mutex, 0);

	first_time = false;
      }
  }

  //! Function which launchs the Tabu Search
  /*!
     Algorithm of the tabu search.
     As a moSA or a moHC, it can be used for HYBRIDATION in an evolutionary algorithm.
     For security a lock (pthread_mutex_t) is closed during the algorithm. 

     \param __sol a solution to improve.
     \return TRUE.
   */
  bool operator   ()(EOT & __sol)
  {

    pthread_mutex_lock (&mutex);

    if (__sol.invalid ())
      {
	full_eval (__sol);
      }

    M move;

    EOT best_sol = __sol, new_sol;

    cont.init ();

    do
      {

	new_sol = __sol;

	try
	{

	  move_expl (__sol, new_sol);

	}
	catch (EmptySelection & __ex)
	{

	  break;
	}

	/* Updating the best solution
	   found until now ? */
	if (new_sol.fitness () > __sol.fitness ())
	  {
	    best_sol = new_sol;
	  }

	__sol = new_sol;

      }
    while (cont (__sol));

    __sol = best_sol;

    pthread_mutex_unlock (&mutex);

    return true;
  }

private:

  //! Boolean allowing to initialise the ptread_mutex_t in the constructor
  static
    bool
    first_time;

  //! The lock
  static
    pthread_mutex_t
    mutex;

  //! Neighborhood explorer
  moMoveExpl < M > &move_expl;

  //! Stop criterion
  moSolContinue < EOT > &cont;

  //! Full evaluation function
  eoEvalFunc < EOT > &full_eval;
};

//! declaration of the mutex variable
template < class EOT > pthread_mutex_t moTS < EOT >::mutex;

//! by default, first_time must have the value true
template < class EOT > bool moTS < EOT >::first_time = true;

#endif
