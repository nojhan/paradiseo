// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "part_route_eval.cpp"

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

#include "part_route_eval.h"
#include "node.h"

PartRouteEval :: PartRouteEval (float __from,
				float __to
				) : from (__from),
				    to (__to) {
  
}

void PartRouteEval :: operator () (Route & __route) {
  
  
  unsigned len = 0 ;
  
  for (unsigned i = (unsigned) (__route.size () * from) ;
       i < (unsigned) (__route.size () * to) ;
       i ++)
    len += distance (__route [i], __route [(i + 1) % numNodes]) ;
  
  __route.fitness (- (int) len) ;
}
