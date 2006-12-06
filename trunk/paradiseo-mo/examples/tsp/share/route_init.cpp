// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "route_init.cpp"

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

#include <utils/eoRNG.h>

#include "route_init.h"
#include "graph.h"

void RouteInit :: operator () (Route & __route) {

  // Init.
  __route.clear () ;
  for (unsigned i = 0 ; i < Graph :: size () ; i ++)
    __route.push_back (i) ;
  
  // Swap. cities

  for (unsigned i = 0 ; i < Graph :: size () ; i ++) {
    //unsigned j = rng.random (Graph :: size ()) ;
    
    unsigned j = (unsigned) (Graph :: size () * (rand () / (RAND_MAX + 1.0))) ;
    unsigned city = __route [i] ;
    __route [i] = __route [j] ;
    __route [j] = city ;
  }   
}
