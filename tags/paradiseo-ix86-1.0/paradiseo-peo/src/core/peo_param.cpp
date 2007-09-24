// "peo_param.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <utils/eoParser.h>

#include "peo_param.h"
#include "peo_debug.h"



void peo :: loadParameters (int & __argc, char * * & __argv) {

  eoParser parser (__argc, __argv);

  /* Debug */
  eoValueParam <std :: string> debug_param ("false", "debug", "?");
  parser.processParam (debug_param);
  if (debug_param.value () == "true")
    setDebugMode ();
}
