// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "part_two_opt_next.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "part_two_opt_next.h"
#include "graph.h"

bool TwoOptNext :: operator () (TwoOpt & __move, const Route & __route) 
{
  if (__move.first == Graph :: size () - 4 && __move.second == __move.first + 2)
    {
      return false ;
    }
  else 
    {
      __move.second ++ ;
      if (__move.second == Graph :: size () - 1) 
	{
	  __move.first ++ ;
	  __move.second = __move.first + 2 ;
	}
      
      return true ;
    }
}
