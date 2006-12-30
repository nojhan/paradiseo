// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "node.cpp"

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
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <math.h>
#include <values.h>

#include "node.h"

unsigned numNodes; /* Number of nodes */

//static unsigned * * dist; /* Square matrix of distances */

double * X_coord, * Y_coord;

double X_min = MAXDOUBLE, X_max = MINDOUBLE, Y_min = MAXDOUBLE, Y_max = MINDOUBLE;

void loadNodes (FILE * __f) {

  /* Coord */
  
  X_coord = new double [numNodes];
  
  Y_coord = new double [numNodes];
 
  unsigned num;

  for (unsigned i = 0; i < numNodes; i ++) {
    
    fscanf (__f, "%u%lf%lf", & num, X_coord + i, Y_coord + i);
    
    if (X_coord [i] < X_min)
      X_min = X_coord [i];
    if (X_coord [i] > X_max)
      X_max = X_coord [i];
    if (Y_coord [i] < Y_min)
      Y_min = Y_coord [i];
    if (Y_coord [i] > Y_max)
      Y_max = Y_coord [i];    
  }
  
  /* Allocation */
  /*
  dist = new unsigned * [numNodes];
  
  for (unsigned i = 0; i < numNodes; i ++)
    dist [i] = new unsigned [numNodes];
  */
  /* Computation of the distances */
  
  /*
  for (unsigned i = 0; i < numNodes; i ++) {

    dist [i] [i] = 0;

    for (unsigned j = 0; j < numNodes; j ++) {
      
      double dx = X_coord [i] - X_coord [j], dy = Y_coord [i] - Y_coord [j];
      
      dist [i] [j] = dist [j] [i] = (unsigned) (sqrt (dx * dx + dy * dy) + 0.5) ;
    }
    }*/
}

unsigned distance (Node __from, Node __to) {

  //  return dist [__from] [__to];

  double dx = X_coord [__from] - X_coord [__to], dy = Y_coord [__from] - Y_coord [__to];
  
  return (unsigned) (sqrt (dx * dx + dy * dy) + 0.5) ;
}

