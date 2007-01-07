// "two_opt.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __two_opt_h
#define __two_opt_h

#include <utility>
#include <moMove.h>

#include "route.h"

class TwoOpt : public moMove <Route>, public std :: pair <unsigned, unsigned> {
  
public :
  
  void operator () (Route & __route);

} ;

#endif
