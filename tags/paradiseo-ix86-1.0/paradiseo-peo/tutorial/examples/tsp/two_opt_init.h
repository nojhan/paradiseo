// "two_opt_init.h"

// (c) OPAC Team, LIFL, January 2006

/* 
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __two_opt_init_h
#define __two_opt_init_h

#include <moMoveInit.h>

#include "two_opt.h"

class TwoOptInit : public moMoveInit <TwoOpt> {
  
public :
  
  void operator () (TwoOpt & __move, const Route & __route) ;
  
} ;

#endif
