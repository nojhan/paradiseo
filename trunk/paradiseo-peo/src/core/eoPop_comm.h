// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoPop_comm.h"

// (c) OPAC Team, LIFL, August 2005

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

#ifndef __eoPop_comm_h
#define __eoPop_comm_h

#include <eoPop.h>

#include "messaging.h"

template <class EOT> void pack (const eoPop <EOT> & __pop) {

  pack ((unsigned) __pop.size ());
  for (unsigned i = 0; i < __pop.size (); i ++)
    pack (__pop [i]);
}

template <class EOT> void unpack (eoPop <EOT> & __pop) {

  unsigned n;
  
  unpack (n);
  __pop.resize (n);
  for (unsigned i = 0; i < n; i ++)
    unpack (__pop [i]);
}
#endif
