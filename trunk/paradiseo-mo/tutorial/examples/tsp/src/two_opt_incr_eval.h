// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "TwoOptIncrEval.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef two_optincr_eval_h
#define two_optincr_eval_h

#include <moMoveIncrEval.h>
#include "two_opt.h"

class TwoOptIncrEval : public moMoveIncrEval <TwoOpt> {

public :
  
  float operator () (const TwoOpt & __move, const Route & __route) ; 

} ;

#endif
