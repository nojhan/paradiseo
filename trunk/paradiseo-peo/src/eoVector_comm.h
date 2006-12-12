// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoVector_comm.h"

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
   
   Contact: cahon@lifl.fr
*/

#ifndef __eoVector_comm_h
#define __eoVector_comm_h

#include <eoVector.h>

#include "mess.h"

template <class F, class T> void pack (const eoVector <F, T> & __v) {

  pack (__v.fitness ()) ;
  unsigned len = __v.size ();
  pack (len);
  for (unsigned i = 0 ; i < len; i ++)
    pack (__v [i]);  
}

template <class F, class T> void unpack (eoVector <F, T> & __v) {

  F fit; 
  unpack (fit);
  __v.fitness (fit);

  unsigned len;
  unpack (len);
  __v.resize (len);
  for (unsigned i = 0 ; i < len; i ++)
    unpack (__v [i]);
}

#endif
