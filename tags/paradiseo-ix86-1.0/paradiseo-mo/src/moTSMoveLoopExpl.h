// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moTSMoveLoopExpl.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moTSMoveLoopExpl_h
#define __moTSMoveLoopExpl_h

#include "moMoveLoopExpl.h"

#include "moMoveInit.h"
#include "moNextMove.h"
#include "moMoveIncrEval.h"
#include "moMoveSelect.h"

#include "moTabuList.h"
#include "moAspirCrit.h"
#include "moBestImprSelect.h"

//! Explorer for a Tabu Search algorithm
/*!
  It is used by a moTS.
 */
template < class M > class moTSMoveLoopExpl:public moMoveLoopExpl < M >
{

  //!Alias for the type
  typedef typename M::EOType EOT;

  //!Alias for the fitness
  typedef typename M::EOType::Fitness Fitness;

public:

  //!Constructor
  /*!
     \param __move_init move initialisation
     \param __next_move neighborhood explorer
     \param __incr_eval efficient evaluation
     \param __tabu_list tabu list
     \param __aspir_crit aspiration criterion
   */
moTSMoveLoopExpl (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval, moTabuList < M > &__tabu_list, moAspirCrit < M > &__aspir_crit):
  move_init (__move_init),
    next_move (__next_move),
    incr_eval (__incr_eval),
    tabu_list (__tabu_list), aspir_crit (__aspir_crit)
  {

    tabu_list.init ();
    aspir_crit.init ();
  }

  //!Procedure which lauches the exploration
  /*!
     The exploration continues while the chosen move is not in the tabu list 
     or the aspiration criterion is true. If these 2 conditions are not true, the
     exploration stops if the move selector update function returns false.

     \param __old_sol the initial solution
     \param __new_sol the new solution
   */
  void operator   () (const EOT & __old_sol, EOT & __new_sol)
  {

    M move;


    move_init (move, __old_sol);	/* Restarting the exploration of 
					   of the neighborhood ! */

    move_select.init (__old_sol.fitness ());

    do
      {

	Fitness fit = incr_eval (move, __old_sol);

	if (!tabu_list (move, __old_sol) || aspir_crit (move, fit))
	  {
	    if (!move_select.update (move, fit))
	      break;
	  }

      }
    while (next_move (move, __old_sol));

    M best_move;

    Fitness best_move_fit;

    move_select (best_move, best_move_fit);

    __new_sol.fitness (best_move_fit);
    best_move (__new_sol);

    /* Removing moves that are
       no more tabu */
    tabu_list.update ();

    // Updating the tabu list
    tabu_list.add (best_move, __new_sol);
  }

private:

  //!Move initialisation
  moMoveInit < M > &move_init;

  //!Neighborhood explorer
  moNextMove < M > &next_move;

  //!Efficient evaluation
  moMoveIncrEval < M > &incr_eval;

  //!Move selector
  moBestImprSelect < M > move_select;

  //!Tabu list
  moTabuList < M > &tabu_list;

  //!Aspiration criterion
  moAspirCrit < M > &aspir_crit;
};

#endif
