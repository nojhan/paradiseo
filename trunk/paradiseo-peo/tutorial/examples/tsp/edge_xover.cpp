// "edge_xover.cpp"

// (c) OPAC Team, LIFL, 2003

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <assert.h>
#include <values.h>

#include <utils/eoRNG.h>

#include "edge_xover.h"

void EdgeXover :: build_map (const Route & __par1, const Route & __par2) {
  
  unsigned len = __par1.size () ;
  
  /* Initialization */
  _map.clear () ;
  _map.resize (len) ;
  
  for (unsigned i = 0 ; i < len ; i ++) {
    _map [__par1 [i]].insert (__par1 [(i + 1) % len]) ;
    _map [__par2 [i]].insert (__par2 [(i + 1) % len]) ;
    _map [__par1 [i]].insert (__par1 [(i - 1 + len) % len]) ;
    _map [__par2 [i]].insert (__par2 [(i - 1 + len) % len]) ;
  }
  
  visited.clear () ;
  visited.resize (len, false) ;
}

void EdgeXover :: remove_entry (unsigned __vertex, std :: vector <std :: set <unsigned> > & __map) {
  
  std :: set <unsigned> & neigh = __map [__vertex] ;

  for (std :: set <unsigned> :: iterator it = neigh.begin () ;
       it != neigh.end () ;
       it ++)
    __map [* it].erase (__vertex) ; 
      
}

void EdgeXover :: add_vertex (unsigned __vertex, Route & __child) {
  
  visited [__vertex] = true ;
  __child.push_back (__vertex) ;    
  remove_entry (__vertex, _map) ; /* Removing entries */    
}

void EdgeXover :: cross (const Route & __par1, const Route & __par2, Route & __child) {
  
  build_map (__par1, __par2) ;
  
  unsigned len = __par1.size () ;
 
  /* Go ! */
  __child.clear () ;
  
  unsigned cur_vertex = rng.random (len) ;
  
  add_vertex (cur_vertex, __child) ;

  for (unsigned i = 1 ; i < len ; i ++) {
    
    unsigned len_min_entry = MAXINT ;
    
    std :: set <unsigned> & neigh = _map [cur_vertex] ;
    
    for (std :: set <unsigned> :: iterator it = neigh.begin () ;
	 it != neigh.end () ;
	 it ++) {      
      unsigned l = _map [* it].size () ;
      if (len_min_entry > l)
	len_min_entry = l ;
    }
    
    std :: vector <unsigned> cand ; /* Candidates */
    
    for (std :: set <unsigned> :: iterator it = neigh.begin () ;
	 it != neigh.end () ;
	 it ++) {      
      unsigned l = _map [* it].size () ;
      if (len_min_entry == l)
	cand.push_back (* it) ;
    }
       
    if (! cand.size ()) {
      
      /* Oh no ! Implicit mutation */      
      for (unsigned j = 0 ; j < len ; j ++)
	if (! visited [j])
	  cand.push_back (j) ;
    }

    cur_vertex = cand [rng.random (cand.size ())] ;
    
    add_vertex (cur_vertex, __child) ;
  } 
}

bool EdgeXover :: operator () (Route & __route1, Route & __route2) {
  
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
