// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moFirstImprSelect.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moFirstImprSelect_h
#define __moFirstImprSelect_h

#include "moMoveSelect.h"

//! One possible moMoveSelect.
/*!
  The neighborhood is explored until
  a move enables an improvement of the
  current solution.
*/
template < class M > class moFirstImprSelect:public moMoveSelect < M >
{

public:

  //! Alias for the fitness.
  typedef typename M::EOType::Fitness Fitness;

  //! Procedure which initialise the exploration.
  /*!
     It save the current fitness as the initial value for the fitness.
   */
  virtual void init (const Fitness & __fit)
  {

    valid = false;
    init_fit = __fit;
  }


  //!Function that indicates if the current move has not improved the fitness.
  /*!
     If the given fitness enables an improvement,
     the move (moMove) should be applied to the current solution.

     \param __move a move.
     \param __fit a fitness linked to the move.
     \return TRUE if the move does not improve the fitness.
   */
  bool update (const M & __move, const typename M::EOType::Fitness & __fit)
  {

    if (__fit > init_fit)
      {

	best_fit = __fit;
	best_move = __move;
	valid = true;

	return false;
      }
    else
      {
	return true;
      }
  }

  //! Procedure which saved the best move and fitness.
  /*!
     \param __move the current move (result of the procedure).
     \param __fit the current fitness (result of the procedure).
     \throws EmptySelection if no move has improved the fitness.
   */
  void operator   () (M & __move, Fitness & __fit) throw (EmptySelection)
  {

    if (valid)
      {
	__move = best_move;
	__fit = best_fit;
      }
    else
      throw EmptySelection ();
  }

private:

  //! Allow to know if at least one move has improved the solution.
  bool valid;

  //! Best stored movement.
  M best_move;

  //! Initial fitness.
  Fitness init_fit;

  //! Best stored fitness.
  Fitness best_fit;

};

#endif
