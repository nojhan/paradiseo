// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt.cpp"

// (c) OPAC Team, LIFL, 2003-2006

/* LICENCE TEXT
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#include "two_opt.h"

TwoOpt TwoOpt :: operator ! () const 
{
  TwoOpt move = * this ;
  std :: swap (move.first, move.second) ;
  
  return move ;
}

void TwoOpt :: operator () (Route & __route) 
{
  
  std :: vector <unsigned int> seq_cities ;
  
  for (unsigned int i = second ; i > first ; i --)
    {
      seq_cities.push_back (__route [i]) ;
    }
  
  unsigned int j = 0 ;
  for (unsigned int i = first + 1 ; i < second + 1 ; i ++)
    {
      __route [i] = seq_cities [j ++] ;
    }
}

void TwoOpt :: readFrom (std :: istream & __is) 
{
  __is >> first >> second ;
}

void TwoOpt :: printOn (std :: ostream & __os) const 
{
  __os << first << ' ' << second ;
}
