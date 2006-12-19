// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moMoveSelect.h"

// (c) OPAC Team, LIFL, 2003-2006

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moMoveSelect_h
#define __moMoveSelect_h

#include <eoFunctor.h>

//! Special class that describes the case of no selection.
/*!
  This class is used as an exception that can be thrown if a solution selector has completly failed.
 */
class EmptySelection
{

};

//! Class that describes a move selector (moMove).
/*! 
  It iteratively considers some moves (moMove) and their
  associated fitnesses. The best move is so regularly updated.
  At any time, it could be accessed.
*/
template < class M > class moMoveSelect:public eoBF < M &, typename M::EOType::Fitness &,
  void >
{
public:
  //! Alias for the fitness
  typedef
    typename
    M::EOType::Fitness
    Fitness;

  //! Procedure which initialises all that the move selector needs including the initial fitness.
  /*! 
     In order to know the fitness of the solution,
     for which the neighborhood will
     be soon explored

     \param __fit the current fitness.
   */
  virtual void
  init (const Fitness & __fit) = 0;

  //! Function which updates the best solutions.
  /*! 
     \param __move a new move.
     \param __fit a fitness linked to the new move.
     \return a boolean that expresses the need to resume the exploration.
   */
  virtual
    bool
  update (const M & __move, const Fitness & __fit) = 0;

};

#endif
