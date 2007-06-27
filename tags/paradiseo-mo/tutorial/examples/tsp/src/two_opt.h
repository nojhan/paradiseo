// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef two_opt_h
#define two_opt_h

#include <eoPersistent.h>

#include <utility>
#include <moMove.h>

#include "route.h"

class TwoOpt : public moMove <Route>, public std :: pair <unsigned, unsigned>, public eoPersistent 
{
  
public :
  
  TwoOpt operator ! () const ;
  
  void operator () (Route & __route) ;
  
  void readFrom (std :: istream & __is) ;
  
  void printOn (std :: ostream & __os) const ;
} ;

#endif
