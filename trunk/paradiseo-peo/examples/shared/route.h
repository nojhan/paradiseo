// "route.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __route_h
#define __route_h

#include <eoVector.h>

#include "node.h"

typedef eoVector <int, Node> Route; 

unsigned length (const Route & __route); 

#endif
