// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoHeaderMessFrom.h"

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

#ifndef eoHeaderMessFrom_h
#define eoHeaderMessFrom_h

#include <string>
#include <mpi.h>

template <class EOT> class eoLocalListener ;

/**
   A header is an identifier for the kind of
   message to be then received ...
*/

template <class EOT> class eoHeaderMessFrom : public string {
  
public :
  
  /**
     Second constructor. On contrary, the header is
     waited for and built from the local listener.
     It comes before a message.
  */
  
  eoHeaderMessFrom (eoLocalListener <EOT> & loc_listen) :
    comm (MPI :: COMM_WORLD) {
    
    MPI :: Status stat ;
    comm.Probe (loc_listen.number (), 0, stat) ; // Blocking
    int len = stat.Get_count (MPI :: CHAR) ;
    char buff [len] ; // Temp. buffer
    comm.Recv (buff, len, MPI :: CHAR, loc_listen.number (), 0) ;
    assign (buff) ;
  }
  
private :
  
  MPI :: Comm & comm ; // Communicator
  
} ;

#endif

