 

/* 
   file: 'mix.h'      
   author: S. CAHON
   mail: paradiseo-help@lists.gforge.inria.fr
   date: dec. 2005    
*/

#ifndef __mix_h
#define __mix_h

#include <vector>

#include <utils/eoRNG.h>

template <class T> void mix (std :: vector <T> & __v) {
  
  unsigned len = __v.size () ;

  for (unsigned i = 0 ; i < len ; i ++)
    std :: swap (__v [i], __v [rng.random (len)]) ;
}

#endif
