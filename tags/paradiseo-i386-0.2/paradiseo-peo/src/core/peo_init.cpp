// "peo_init.cpp"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include <stdio.h>

#include "peo_init.h"
#include "peo_param.h"
#include "peo_debug.h"
#include "rmc.h"

namespace peo {

  int * argc;
  
  char * * * argv;

  void init (int & __argc, char * * & __argv) {

    argc = & __argc;
    
    argv = & __argv;
    
    /* Initializing the the Resource Management and Communication */
    initRMC (__argc, __argv);

    /* Loading the common parameters */ 
    loadParameters (__argc, __argv);
    
    /* */
    initDebugging ();
  }
}
