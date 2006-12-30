// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "order_xover.cpp"

// (c) OPAC Team, LIFL, 2002

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
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <assert.h>

#include <utils/eoRNG.h>

#include "order_xover.h"

void OrderXover :: cross (const Route & __par1, const Route & __par2, Route & __child) {

  unsigned cut2 = 1 + rng.random (numNodes) ;    
  unsigned cut1 = rng.random (cut2);
  unsigned l = 0;

  /* To store vertices that have already been crossed */
  std :: vector <bool> v (numNodes, false);

  /* Copy of the left partial route of the first parent */ 
  for (unsigned i = cut1 ; i < cut2 ; i ++) {
    __child [l ++] = __par1 [i] ; 
    v [__par1 [i]] = true ;
  }
   
  /* Searching the vertex of the second path, that ended the previous first one */
  unsigned from = 0 ;
  for (unsigned i = 0; i < numNodes; i ++)
    if (__par2 [i] == __child [cut2 - 1]) { 
      from = i ;
      break ;
    }
  
  /* Selecting a direction (Left or Right) */
  char direct = rng.flip () ? 1 : -1 ;
      
  for (unsigned i = 0; i < numNodes + 1; i ++) {
    unsigned bidule = (direct * i + from + numNodes) % numNodes;
    if (! v [__par2 [bidule]]) {
      __child [l ++] = __par2 [bidule] ;
      v [__par2 [bidule]] = true ;
    }
  }
} 

bool OrderXover :: operator () (Route & __route1, Route & __route2) {
  
  // Init. copy
  Route par [2] ;
  par [0] = __route1 ;
  par [1] = __route2 ;
  
  cross (par [0], par [1], __route1) ;
  cross (par [1], par [0], __route2) ;
  
  __route1.invalidate () ;
  __route2.invalidate () ;

  return true ;
}
