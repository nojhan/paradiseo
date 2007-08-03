// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moHCMoveLoopExpl.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moHCMoveLoopExpl_h
#define __moHCMoveLoopExpl_h

#include "moMoveLoopExpl.h"

#include "moMoveInit.h"
#include "moNextMove.h"
#include "moMoveIncrEval.h"
#include "moMoveSelect.h"

//! Iterative explorer used by a moHC.
template < class M > class moHCMoveLoopExpl:public moMoveLoopExpl < M >
{

  //! Alias for the type.
  typedef typename M::EOType EOT;

  //! Alias for the fitness.
  typedef typename M::EOType::Fitness Fitness;

public:

  //! Constructor.
  /*!
     All the boxes have to be specified.

     \param __move_init the move initialiser.
     \param __next_move the neighborhood explorer.
     \param __incr_eval (generally) efficient evaluation function.
     \param __move_select the move selector.
   */
moHCMoveLoopExpl (moMoveInit < M > &__move_init, moNextMove < M > &__next_move, moMoveIncrEval < M > &__incr_eval, moMoveSelect < M > &__move_select):

  move_init (__move_init),
    next_move (__next_move),
    incr_eval (__incr_eval), move_select (__move_select)
  {

  }

  //!  Procedure which launches the explorer.
  /*!
     The exploration starts from an old solution and provides a new solution.

     \param __old_sol the current solution.
     \param __new_sol the new_sol (result of the procedure).
   */
  void operator   () (const EOT & __old_sol, EOT & __new_sol)
  {

    M move;

    //
    move_init (move, __old_sol);	/* Restarting the exploration of 
					   of the neighborhood ! */
    
    move_select.init (__old_sol.fitness ());
    
    while (move_select.update (move, incr_eval (move, __old_sol))
	   && next_move (move, __old_sol));
    
    try
      {
	
	M best_move;
	
	Fitness best_move_fit;
	
	move_select (best_move, best_move_fit);
	__new_sol.fitness (best_move_fit);
	best_move (__new_sol);
	
      }
    catch (EmptySelection & __ex)
      {
	
	// ?
      }
  }
  
private:

  //! Move initialiser.
  moMoveInit < M > &move_init;

  //! Neighborhood explorer.
  moNextMove < M > &next_move;

  //! (generally) Efficient evaluation.
  moMoveIncrEval < M > &incr_eval;

  //! Move selector.
  moMoveSelect < M > &move_select;

};

#endif
