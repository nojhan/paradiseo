// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoPublishMessFrom.h"

// (c) OPAC Team, LIFL, 2002

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

#ifndef eoPublishMessFrom_h
#define eoPublishMessFrom_h

#include <string.h>
#include <mpi.h>
#include <paradisEO/comm/messages/eoMessFrom.h>

/**
   To be notified of the lauch of a new kind of process somewhere ...
 */

template <class EOT> class eoPublishMessFrom : public eoMessFrom <EOT> {
  
public :
  
  /**
     Constructor
   */

  eoPublishMessFrom (eoLocalListener <EOT> & _loc_listen) :
    eoMessFrom <EOT> (_loc_listen) {
    
    MPI :: Status stat ;
    comm.Probe (loc_listen.number (), 0, stat) ;
    int len = stat.Get_count (MPI :: CHAR) ;
    char buff [len] ;
    comm.Recv (buff, len, MPI :: CHAR, loc_listen.number (), 0) ;
    label.assign (buff) ;
  }
  
  void operator () () {
    
   loc_listen.label () = label ;
  }

private :

  string label ; // String identifier ...

} ;

#endif



