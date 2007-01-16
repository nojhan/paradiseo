// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "graph.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT 
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef graph_h
#define graph_h

#include <vector>
#include <utility>

namespace Graph {

  void load (const char * __file_name) ;
  /* Loading cities
     (expressed by their coordinates)
     from the given file name */  
  
  float distance (unsigned __from, unsigned __to) ;

  unsigned size () ; // How many cities ?
}

#endif
