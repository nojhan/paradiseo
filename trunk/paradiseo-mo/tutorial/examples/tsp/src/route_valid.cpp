// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "route_valid.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "route_valid.h"

#include <vector>

bool valid (Route & __route) 
{
  
  std::vector<unsigned int> t;
  t.resize(__route.size());
  
  for (unsigned int i = 0 ; i < __route.size () ; i ++)
    {
      t [i] = 0 ;
    }
  
  for (unsigned int i = 0 ; i < __route.size () ; i ++)
    {
      t [__route [i]] ++ ;
    }
  
  for (unsigned int i = 0 ; i < __route.size () ; i ++)
    {
      if (t [i] != 1)
	{
	  t.clear();
	  return false ;
	}
    }
  
  t.clear();
  return true ; // OK.
}
