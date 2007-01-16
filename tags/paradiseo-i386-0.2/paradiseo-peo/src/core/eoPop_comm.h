// "eoPop_comm.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __eoPop_comm_h
#define __eoPop_comm_h

#include <eoPop.h>

#include "messaging.h"

template <class EOT> void pack (const eoPop <EOT> & __pop) {

  pack ((unsigned) __pop.size ());
  for (unsigned i = 0; i < __pop.size (); i ++)
    pack (__pop [i]);
}

template <class EOT> void unpack (eoPop <EOT> & __pop) {

  unsigned n;
  
  unpack (n);
  __pop.resize (n);
  for (unsigned i = 0; i < n; i ++)
    unpack (__pop [i]);
}
#endif
