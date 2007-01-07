// "peo_init.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __peo_init_h
#define __peo_init_h

namespace peo {

  extern int * argc;
  
  extern char * * * argv;
  
  extern void init (int & __argc, char * * & __argv);
}

#endif
