// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "edge_xover.h"

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
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef edge_xover_h
#define edge_xover_h

#include <vector>
#include <set>

#include <eoOp.h>

#include "route.h"

/** Edge Crossover */
class EdgeXover : public eoQuadOp <Route> {
  
public :
  
  bool operator () (Route & __route1, Route & __route2) ;

private :
  
  void cross (const Route & __par1, const Route & __par2, Route & __child) ; /* Binary */

  void remove_entry (unsigned __vertex, std :: vector <std :: set <unsigned> > & __map) ;
  /* Updating the map of entries */

  void build_map (const Route & __par1, const Route & __par2) ;

  void add_vertex (unsigned __vertex, Route & __child) ;

  std :: vector <std :: set <unsigned> > _map ; /* The handled map */

  std :: vector <bool> visited ; /* Vertices that are already visited */

} ;

#endif
