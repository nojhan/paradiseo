// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moNextMove.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moItRandNextMove_h
#define __moItRandNextMove_h

#include "moNextMove.h"
#include "moRandMove.h"

//! One of the possible moNextMove.
/*!
  This class is a move (moMove) generator with a bound for the maximum number of iterations.
*/
template < class M > class moItRandNextMove:public moNextMove < M >
{

  //! Alias for the type.
  typedef typename M::EOType EOT;

public:

  //! The constructor.
  /*!
     Parameters only for initialising the attributes.

     \param __rand_move the random move generator.
     \param __max_iter the iteration maximum number.
   */
  moItRandNextMove (moRandMove < M > &__rand_move,
		    unsigned __max_iter):rand_move (__rand_move),
    max_iter (__max_iter), num_iter (0)
  {

  }

  //! Generation of a new move
  /*!
     If the maximum number is not already reached, the current move is forgotten and remplaced by another one.

     \param __move the current move.
     \param __sol the current solution.
     \return FALSE if the maximum number of iteration is reached, else TRUE.
   */
  bool operator   () (M & __move, const EOT & __sol)
  {

    if (num_iter++ > max_iter)
      {

	num_iter = 0;
	return false;
      }
    else
      {

	/* The given solution is discarded here */
	rand_move (__move);
	num_iter++;
	return true;
      }
  }

private:

  //! A move generator (generally randomly).
  moRandMove < M > &rand_move;

  //! Iteration maximum number.
  unsigned max_iter;

  //! Iteration current number.
  unsigned num_iter;

};

#endif
