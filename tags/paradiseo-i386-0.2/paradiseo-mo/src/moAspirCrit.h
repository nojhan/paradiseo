// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "moAspirCrit.h"

// (c) OPAC Team, LIFL, 2003-2006

/* TEXT LICENCE
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __moAspirCrit_h
#define __moAspirCrit_h

#include <eoFunctor.h>

//! Description of the conditions in which a tabu move could be accepted
/*!
  It is only a description... An object that herits from this class is needed to be used in a moTS.
  See moNoAspriCrit for example.
 */
template < class M > class moAspirCrit:public eoBF < const M &, const typename
  M::EOType::Fitness &,
  bool >
{

public:
  //! Procedure which initialises all that needs a aspiration criterion.
  /*!
     It can be possible that this procedure do nothing...
   */
  virtual void
  init () = 0;

};

#endif
