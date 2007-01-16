// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moTabuList.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT 
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moTabuList_h
#define __moTabuList_h

#include <eoFunctor.h>

//! Class describing a tabu list that a moTS uses
/*!
  It is only a description, does nothing... A new object that herits from this class has to be defined in order
  to be used in a moTS.
 */
template < class M > class moTabuList:public eoBF < const M &, const typename
  M::EOType &,
  bool >
{

public:
  //! Alias for the type
  typedef
    typename
    M::EOType
    EOT;


  //! Procedure to add a move in the tabu list
  /*!
     The two parameters have not to be modified so they are constant parameters

     \param __move a new tabu move
     \param __sol the solution associated to this move
   */
  virtual void
  add (const M & __move, const EOT & __sol) = 0;

  //! Procedure that updates the tabu list content
  /*!
     Generally, a counter associated to each saved move is decreased by one.
   */
  virtual void
  update () = 0;

  //! Procedure which initialises the tabu list
  /*!
     Can be useful if the data structure needs to be allocated before being used.
   */
  virtual void
  init () = 0;
};

#endif
