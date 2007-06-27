// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_rand.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef two_opt_rand_h
#define two_opt_rand_h

#include <moRandMove.h>

#include "two_opt.h"

class TwoOptRand : public moRandMove <TwoOpt> 
{
  
public :
  
  void operator () (TwoOpt & __move) ;
  
} ;

#endif
