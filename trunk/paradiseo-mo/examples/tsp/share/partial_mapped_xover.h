// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "partial_mapped_xover.h"

// (c) OPAC Team, LIFL, 2003

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: cahon@lifl.fr
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
