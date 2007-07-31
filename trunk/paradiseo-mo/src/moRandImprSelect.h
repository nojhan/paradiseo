// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moRandImprSelect.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moRandImprSelect_h
#define __moRandImprSelect_h

#include <vector>

#include <utils/eoRNG.h>
#include "moMoveSelect.h"

//! One of the possible moMove selector (moMoveSelect)
/*!
  All the neighbors are considered. 
  One of them that enables an improvement of the objective function is choosen.
*/
template < class M > class moRandImprSelect:public moMoveSelect < M >
{

public:

  //! Alias for the fitness
  typedef typename M::EOType::Fitness Fitness;

  //!Procedure which all that needs a moRandImprSelect
  /*!
     Give a value to the initialise fitness.
     Clean the move and fitness vectors.

     \param __fit the current best fitness
   */
  void init (const Fitness & __fit)
  {
    init_fit = __fit;
    vect_better_fit.clear ();
    vect_better_moves.clear ();
  }

  //! Function that updates the fitness and move vectors
  /*!
     if a move give a better fitness than the initial fitness, 
     it is saved and the fitness too.

     \param __move a new move.
     \param __fit a new fitness associated to the new move.
     \return TRUE.
   */
  bool update (const M & __move, const Fitness & __fit)
  {

    if (__fit > init_fit)
      {

	vect_better_fit.push_back (__fit);
	vect_better_moves.push_back (__move);
      }

    return true;
  }

  //! The move selection
  /*!
     One the saved move is randomly chosen.

     \param __move the reference of the move that can be initialised by the function.
     \param __fit the reference of the fitness that can be initialised by the function.
     \throws EmptySelection If no move which improves the current fitness are found.
   */
  void operator   () (M & __move, Fitness & __fit) throw (EmptySelection)
  {

    if (!vect_better_fit.empty ())
      {

	unsigned n = rng.random (vect_better_fit.size ());

	__move = vect_better_moves[n];
	__fit = vect_better_fit[n];
      }
    else
      throw EmptySelection ();
  }

private:

  //! Fitness of the current solution.
  Fitness init_fit;

  //! Candidate fitnesse vector.
  std::vector < Fitness > vect_better_fit;

  //! Candidate move vector.
  std::vector < M > vect_better_moves;
};

#endif
