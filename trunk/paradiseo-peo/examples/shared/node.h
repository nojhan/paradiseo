// "node.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __node_h
#define __node_h

#include <stdio.h>

typedef unsigned Node; 

extern double X_min, X_max, Y_min, Y_max;

extern double * X_coord, * Y_coord;

extern unsigned numNodes; /* Number of nodes */

extern void loadNodes (FILE * __f);

extern unsigned distance (Node __from, Node __to);

#endif
