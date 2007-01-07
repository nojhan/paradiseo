// "eoVector_comm.h"

// (c) OPAC Team, LIFL, August 2005

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __eoVector_comm_h
#define __eoVector_comm_h

#include <eoVector.h>

#include "messaging.h"

template <class F, class T> void pack (const eoVector <F, T> & __v) {

  pack (__v.fitness ()) ;
  unsigned len = __v.size ();
  pack (len);
  for (unsigned i = 0 ; i < len; i ++)
    pack (__v [i]);  
}

template <class F, class T> void unpack (eoVector <F, T> & __v) {

  F fit; 
  unpack (fit);
  __v.fitness (fit);

  unsigned len;
  unpack (len);
  __v.resize (len);
  for (unsigned i = 0 ; i < len; i ++)
    unpack (__v [i]);
}

#endif
