// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "partial_mapped_xover.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef partial_mapped_xover_h
#define partial_mapped_xover_h

#include <eoOp.h>

#include "route.h"

/** Partial Mapped Crossover */
class PartialMappedXover : public eoQuadOp <Route> {
  
public :
  
  bool operator () (Route & __route1, Route & __route2) ;

private :
  
  void repair (Route & __route, unsigned __cut1, unsigned __cut2) ;
} ;

#endif
