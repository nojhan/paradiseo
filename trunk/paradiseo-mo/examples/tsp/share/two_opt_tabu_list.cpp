// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_tabu_list.cpp"

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

#include "two_opt_tabu_list.h"
#include "graph.h"

#define TABU_LENGTH 10

void TwoOptTabuList :: init () {
  
  // Size (eventually)
  tabu_span.resize (Graph :: size ()) ;
  for (unsigned i = 0 ; i < tabu_span.size () ; i ++)
    tabu_span [i].resize (Graph :: size ()) ;  

  // Clear
  for (unsigned i = 0 ; i < tabu_span.size () ; i ++)
    for (unsigned j = 0 ; j < tabu_span [i].size () ; j ++)
      tabu_span [i] [j] = 0 ;
}

bool TwoOptTabuList :: operator () (const TwoOpt & __move, const Route & __sol) {
  
  return tabu_span [__move.first] [__move.second] > 0 ;
}

void TwoOptTabuList :: add (const TwoOpt & __move, const Route & __sol) {
  
  tabu_span [__move.first] [__move.second] = tabu_span [__move.second] [__move.first] = TABU_LENGTH ;
}

void TwoOptTabuList :: update () {
  
  for (unsigned i = 0 ; i < tabu_span.size () ; i ++)
    for (unsigned j = 0 ; j < tabu_span [i].size () ; j ++)
      if (tabu_span [i] [j] > 0)
	tabu_span [i] [j] -- ;
}
