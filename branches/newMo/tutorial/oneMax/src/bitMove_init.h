// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "bitMove_init.h"
#ifndef __bitMove_init
#define __bitMove_init

#include <move/moMoveInit.h>
#include "bitMove.h"

template <class EOT>
class BitMove_init : public moMoveInit < BitMove<EOT> > {
  
public :
  
  void operator () (BitMove<EOT> & __move, const EOT & genome) {
    __move.bit = 0 ;
  };
  
} ;

#endif
