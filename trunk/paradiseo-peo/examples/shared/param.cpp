// "param.cpp"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <utils/eoParser.h>

#include "data.h"
#include "opt_route.h"

void loadParameters (int __argc, char * * __argv) {

  eoParser parser (__argc, __argv);
  
  loadData (parser);

  loadOptimumRoute (parser);
}


