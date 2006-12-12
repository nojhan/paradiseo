// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "ring_topo.cpp"

// (c) OPAC Team, LIFL, September 2005

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

#include "ring_topo.h"

void RingTopology :: setNeighbors (Cooperative * __mig,
				   std :: vector <Cooperative *> & __from,
				   std :: vector <Cooperative *> & __to) {
  __from.clear () ;
  __to.clear () ;

    int len = mig.size () ;
    
    for (int i = 0 ; i < len ; i ++)      
      if (mig [i] == __mig) {	
	__from.push_back (mig [(i - 1 + len) % len]) ;
	__to.push_back (mig [(i + 1) % len]) ;	
	break;
      }
}
