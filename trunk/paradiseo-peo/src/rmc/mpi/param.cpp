// "param.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <utils/eoParser.h>

#include "schema.h"

void loadRMCParameters (int & __argc, char * * & __argv) {

  eoParser parser (__argc, __argv);

  /* Schema */
  eoValueParam <std :: string> schema_param ("schema.xml", "schema", "?");
  parser.processParam (schema_param);
  loadSchema (schema_param.value ().c_str ());
}
