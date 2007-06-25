// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_init.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt_init.h"

void TwoOptInit :: operator () (TwoOpt & __move, const Route & __route) 
{
  __move.first = 0 ;
  __move.second = 2 ;
}
