// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moHC.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moHC_h
#define __moHC_h

#include <eoOp.h>
#include <eoEvalFunc.h>

#include "moAlgo.h"
#include "moMoveExpl.h"
#include "moHCMoveLoopExpl.h"

//! Hill Climbing (HC)
/*!
  Class which describes the algorithm for a hill climbing.
 */
template < class M > class moHC:public moAlgo < typename M::EOType >
{

  //! Alias for the type.
  typedef
    typename
    M::EOType
    EOT;

  //! Alias for the fitness.
  typedef
    typename
    EOT::Fitness
    Fitness;

public:

  //! Full constructor.
  /*!
     All the boxes are given in order the HC to use a moHCMoveLoopExpl.

     \param __move_init a move initialiser.
     \param __next_move a neighborhood explorer.
     \param __incr_eval a (generally) efficient evaluation function.
     \param __move_select a move selector.
     \param __full_eval a full evaluation function.
   */
moHC (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval, moMoveSelect < M > &__move_select, eoEvalFunc < EOT > &__full_eval):move_expl (*new moHCMoveLoopExpl < M >
	     (__move_init, __next_move, __incr_eval, __move_select)),
    full_eval (__full_eval)
  {

  }

  //! Light constructor.
  /*!
     This constructor allow to use another moMoveExpl (generally not a moHCMoveLoopExpl).

     \param __move_expl a complete explorer.
     \param __full_eval a full evaluation function.
   */
moHC (moMoveExpl < M > &__move_expl, eoEvalFunc < EOT > &__full_eval):move_expl (__move_expl),
    full_eval
    (__full_eval)
  {

  }

  //! Function which launches the HC
  /*!
     The HC has to improve a current solution.
     As the moSA and the mo TS, it can be used for HYBRIDATION in an evolutionnary algorithm.

     \param __sol a current solution to improve.
     \return TRUE.
   */
  bool operator   ()(EOT & __sol)
  {

    if (__sol.invalid ())
      {
	full_eval (__sol);
      }

    EOT new_sol;

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

	if (new_sol.fitness () > __sol.fitness ())
	  {
	    __sol = new_sol;
	  }
	else
	  {
	    break;
	  }

      }
    while (true);

    return true;
  }

private:

  //! Complete exploration of the neighborhood.
  moMoveExpl < M > &move_expl;

  //! A full evaluation function.
  eoEvalFunc < EOT > &full_eval;
};

#endif
