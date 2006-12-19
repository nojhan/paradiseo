// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "partial_mapped_xover.cpp"

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

#include <assert.h>

#include <utils/eoRNG.h>

#include "partial_mapped_xover.h"
#include "route_valid.h"
#include "mix.h"

void PartialMappedXover :: repair (Route & __route, unsigned __cut1, unsigned __cut2) {
  
  unsigned v [__route.size ()] ; // Number of times a cities are visited ...
  
  for (unsigned i = 0 ; i < __route.size () ; i ++)
    v [i] = 0 ;
  
  for (unsigned i = 0 ; i < __route.size () ; i ++)
    v [__route [i]] ++ ;
  
  std :: vector <unsigned> vert ;

  for (unsigned i = 0 ; i < __route.size () ; i ++)
    if (! v [i])
      vert.push_back (i) ;
  
  mix (vert) ;

  for (unsigned i = 0 ; i < __route.size () ; i ++)
    if (i < __cut1 || i >= __cut2)
      if (v [__route [i]] > 1) {
	__route [i] = vert.back () ;
	vert.pop_back () ;
      }
}

bool PartialMappedXover :: operator () (Route & __route1, Route & __route2) {
    
  unsigned cut1 = rng.random (__route1.size ()), cut2 = rng.random (__route2.size ()) ;
  
  if (cut2 < cut1)
    std :: swap (cut1, cut2) ;
  
  // Between the cuts
  for (unsigned i = cut1 ; i < cut2 ; i ++)
    std :: swap (__route1 [i], __route2 [i]) ;
  
  // Outside the cuts
  repair (__route1, cut1, cut2) ;
  repair (__route2, cut1, cut2) ;
  
  // Debug
  assert (valid (__route1)) ;
  assert (valid (__route2)) ;

  __route1.invalidate () ;
  __route2.invalidate () ;

  return true ;
}
