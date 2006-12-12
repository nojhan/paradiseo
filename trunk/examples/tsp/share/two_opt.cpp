// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt.cpp"

// (c) OPAC Team, LIFL, 2003

/* This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2 of the License, or (at your option) any later version.
   
   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
   
   Contact: cahon@lifl.fr
*/

#include "two_opt.h"

TwoOpt TwoOpt :: operator ! () const {
  
  TwoOpt move = * this ;
  std :: swap (move.first, move.second) ;
  
  return move ;
}

void TwoOpt :: operator () (Route & __route) {
  
  std :: vector <unsigned> seq_cities ;
  
  for (unsigned i = second ; i > first ; i --)
    seq_cities.push_back (__route [i]) ;
  
  unsigned j = 0 ;
  for (unsigned i = first + 1 ; i < second + 1 ; i ++)
    __route [i] = seq_cities [j ++] ;
}

void TwoOpt :: readFrom (std :: istream & __is) {
  
  __is >> first >> second ;
}

void TwoOpt :: printOn (std :: ostream & __os) const {
  
  __os << first << ' ' << second ;
}
