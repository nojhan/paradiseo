// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "route.h"

// (c) OPAC Team, LIFL, 2003-2007

/* LICENCE TEXT 
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef route_h
#define route_h

#include <eoVector.h>
#include <eoScalarFitness.h>

typedef eoScalarFitness< float, std::greater< float > > tspFitness ;

typedef eoVector <tspFitness, unsigned int> Route ; // [Fitness (length), Gene (city)]

#endif
