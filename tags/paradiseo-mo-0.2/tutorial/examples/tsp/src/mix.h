// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "mix.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef mix_h
#define mix_h

#include <utils/eoRNG.h>

template <class T> void mix (std :: vector <T> & __vect) 
{
  for (unsigned int i = 0 ; i < __vect.size () ; i ++)   
    {
      std :: swap (__vect [i], __vect [rng.random (__vect.size ())]) ;
    }
}

#endif
