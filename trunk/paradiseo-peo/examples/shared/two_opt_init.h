
// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "two_opt_init.h"

// (c) OPAC Team, LIFL, January 2006

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
   
   Contact: paradiseo-help@lists.gforge.inria.fr
*/

#ifndef __two_opt_init_h
#define __two_opt_init_h

#include <moMoveInit.h>

#include "two_opt.h"

class TwoOptInit : public moMoveInit <TwoOpt> {
  
public :
  
  void operator () (TwoOpt & __move, const Route & __route) ;
  
} ;

#endif
