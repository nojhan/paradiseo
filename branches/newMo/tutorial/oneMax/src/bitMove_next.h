// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "bitMove_next.h"

#ifndef __bitMove_next_h
#define __bitMove_next_h

#include <move/moNextMove.h>
#include "bitMove.h"

template <class EOT>
class BitMove_next : public moNextMove < BitMove<EOT> > {
  
public:
  BitMove_next()
  {
  };

  bool operator () (BitMove<EOT> & __move, const EOT & genome) {
  
    if (__move.bit >= (genome.size() - 1)){
      return false ;
}
    else {
      __move.bit++;
      return true ;
    }
  };

  
} ;

#endif
