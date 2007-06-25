// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_tabu_list.h"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef two_opt_tabu_list_h
#define two_opt_tabu_list_h

#include <moTabuList.h>
#include "two_opt.h"
#include "route.h"

/** The table of tabu movements, i.e. forbidden edges */
class TwoOptTabuList : public moTabuList <TwoOpt> 
{
public :
  
  bool operator () (const TwoOpt & __move, const Route & __sol) ;
  
  void add (const TwoOpt & __move, const Route & __sol) ;
  
  void update () ;
  
  void init () ;
  
private :
  
  std :: vector <std :: vector <unsigned> > tabu_span ;
  
} ;

#endif
