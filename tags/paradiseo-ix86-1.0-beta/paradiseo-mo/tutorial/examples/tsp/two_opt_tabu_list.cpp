// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_tabu_list.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt_tabu_list.h"
#include "graph.h"

#define TABU_LENGTH 10

void TwoOptTabuList :: init () 
{
  // Size (eventually)
  tabu_span.resize (Graph :: size ()) ;
  for (unsigned int i = 0 ; i < tabu_span.size () ; i ++)
    {
      tabu_span [i].resize (Graph :: size ()) ;  
    }

  // Clear
  for (unsigned int i = 0 ; i < tabu_span.size () ; i ++)
    {
      for (unsigned int j = 0 ; j < tabu_span [i].size () ; j ++)
	{
	  tabu_span [i] [j] = 0 ;
	}
    }
}

bool TwoOptTabuList :: operator () (const TwoOpt & __move, const Route & __sol) 
{
  return tabu_span [__move.first] [__move.second] > 0 ;
}

void TwoOptTabuList :: add (const TwoOpt & __move, const Route & __sol) 
{
  tabu_span [__move.first] [__move.second] = tabu_span [__move.second] [__move.first] = TABU_LENGTH ;
}

void TwoOptTabuList :: update () 
{
  for (unsigned int i = 0 ; i < tabu_span.size () ; i ++)
    {
      for (unsigned int j = 0 ; j < tabu_span [i].size () ; j ++)
	{
	  if (tabu_span [i] [j] > 0)
	    {
	      tabu_span [i] [j] -- ;
	    }
	}
    }
}
