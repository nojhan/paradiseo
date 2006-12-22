// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "TwoOptIncrEval.cpp"

// (c) OPAC Team, LIFL, January 2006

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

#include "two_opt_incr_eval.h"
#include "node.h"

int TwoOptIncrEval :: operator () (const TwoOpt & __move, const Route & __route) {
  
  /* From */
  Node v1 = __route [__move.first], v1_left = __route [(__move.first - 1 + numNodes) % numNodes];
  
  /* To */
  Node v2 = __route [__move.second], v2_right = __route [(__move.second + 1) % numNodes];
 
  if (v1 == v2 || v2_right == v1)
    return __route.fitness ();
  else 
    return __route.fitness () - distance (v1_left, v2) - distance (v1, v2_right) + distance (v1_left, v1) + distance (v2, v2_right);
}
