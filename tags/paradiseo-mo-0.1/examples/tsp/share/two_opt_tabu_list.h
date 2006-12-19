// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_tabu_list.h"

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

#ifndef two_opt_tabu_list_h
#define two_opt_tabu_list_h

#include <moTabuList.h>
#include "two_opt.h"
#include "route.h"

/** The table of tabu movements, i.e. forbidden edges */
class TwoOptTabuList : public moTabuList <TwoOpt> {
  
public :
  
  bool operator () (const TwoOpt & __move, const Route & __sol) ;
  
  void add (const TwoOpt & __move, const Route & __sol) ;
  
  void update () ;

  void init () ;
  
private :
  
  std :: vector <std :: vector <unsigned> > tabu_span ;
  
} ;

#endif
