// "opt_route.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __opt_route_h
#define __opt_route_h

#include <cassert>
#include <utils/eoParser.h>

#include "route.h"

extern void loadOptimumRoute (const char * __filename);

extern void loadOptimumRoute (eoParser & __parser);

extern Route opt_route; /* Optimum route */

#endif
