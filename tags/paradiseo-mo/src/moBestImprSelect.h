// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moBestImprSelect.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moBestImprSelect_h
#define __moBestImprSelect_h

#include "moMoveSelect.h"

//! One of the possible moMoveSelect.
/*!
  All neighbors are considered, and the movement
  which enables the best improvement is selected.
*/
template < class M > class moBestImprSelect:public moMoveSelect < M >
{

public:

  //! Alias for the fitness.
  typedef typename M::EOType::Fitness Fitness;

  //! Procedure which initialise the exploration
  void init (const Fitness & __fit)
  {

    first_time = true;
  }


  //!Function that indicates if the current move has not improved the fitness.
  /*!
     If the given fitness enables an improvment,
     the move (moMove) and the fitness linked to this move are saved.

     \param __move a move.
     \param __fit a fitness linked to the move.
     \return TRUE if the move does not improve the fitness.
   */
  bool update (const M & __move, const Fitness & __fit)
  {

    if (first_time || __fit > best_fit)
      {

	best_fit = __fit;
	best_move = __move;

	first_time = false;
      }

    return true;
  }

  //! Procedure which saved the best move and fitness.
  /*!
     \param __move the current move (result of the procedure).
     \param __fit the current fitness (result of the procedure).
     \throws EmptySelection if no move has improved the fitness.
   */
  void operator   () (M & __move, Fitness & __fit) throw (EmptySelection)
  {

    if (!first_time)
      {
	__move = best_move;
	__fit = best_fit;
      }
    else
      throw EmptySelection ();
  }

private:

  //! Allowing to know if at least one move has been generated. 
  bool first_time;

  //! The best move.
  M best_move;

  //! The best fitness.
  Fitness best_fit;

};

#endif
