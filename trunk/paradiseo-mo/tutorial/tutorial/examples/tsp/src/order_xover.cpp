// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "order_xover.cpp"

// (c) OPAC Team, LIFL, 2002-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <assert.h>
#include <vector.h>

#include <utils/eoRNG.h>

#include "order_xover.h"
#include "route_valid.h"

void OrderXover :: cross (const Route & __par1, const Route & __par2, Route & __child) {
  
  unsigned cut = rng.random (__par1.size ()) ;
      
  /* To store vertices that have
     already been crossed */
  vector<bool> v;
  v.resize(__par1.size());
  
  for (unsigned i = 0 ; i < __par1.size () ; i ++)
    {
      v [i] = false ;
    }

  /* Copy of the left partial
     route of the first parent */ 
  for (unsigned i = 0 ; i < cut ; i ++) {
    __child [i] = __par1 [i] ; 
    v [__par1 [i]] = true ;
  }
   
  /* Searching the vertex of the second path, that ended
     the previous first one */
  unsigned from = 0 ;
  for (unsigned i = 0 ; i < __par2.size () ; i ++)
    {
      if (__par2 [i] == __child [cut - 1]) {
	from = i ;
	break ;
      }
    }
  
  /* Selecting a direction
     Left or Right */
  char direct = rng.flip () ? 1 : -1 ;
    
  /* Copy of the left vertices from
     the second parent path */
  unsigned l = cut ;
  
  for (unsigned i = 0 ; i < __par2.size () ; i ++) 
    {
      unsigned bidule /* :-) */ = (direct * i + from + __par2.size ()) % __par2.size () ;
      if (! v [__par2 [bidule]]) 
	{
	  __child [l ++] = __par2 [bidule] ;
	  v [__par2 [bidule]] = true ;
	}
    }
  
  v.clear();
} 

bool OrderXover :: operator () (Route & __route1, Route & __route2) {
  
  // Init. copy
  Route par [2] ;
  par [0] = __route1 ;
  par [1] = __route2 ;
  
  cross (par [0], par [1], __route1) ;
  cross (par [1], par [0], __route2) ;
  
  assert (valid (__route1)) ;
  assert (valid (__route2)) ;

  __route1.invalidate () ;
  __route2.invalidate () ;

  return true ;
}
