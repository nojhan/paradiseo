// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoEOSendMessFrom.h"

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

#ifndef eoEOSendMessFrom_h
#define eoEOSendMessFrom_h

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <string>
#ifdef HAVE_SSTREAM
#include <sstream>
#else
#include <strstream.h>
#endif
#include <mpi.h>
#include <paradisEO/comm/messages/eoMessFrom.h>

using namespace std;

/**
   A message embeding a set of immigrants ...
*/

template <class EOT> class eoEOSendMessFrom : public eoMessFrom <EOT> {
  
public :
  
  /**
     Constructor
  */

  eoEOSendMessFrom (eoLocalListener <EOT> & _loc_listen) :
    eoMessFrom <EOT> (_loc_listen) {
    
    MPI :: Status stat ;
    comm.Probe (loc_listen.number (), 0, stat) ;
    int len = stat.Get_count (MPI :: CHAR) ;
    char buff [len] ;
    comm.Recv (buff, len, MPI :: CHAR, loc_listen.number (), 0) ;
    istrstream f (buff) ;
    pop.readFrom (f) ;

  }
  
  void operator () () {
    
    loc_listen.push (pop) ;
    //    std::cout << "Reception de " << pop.size () << "individus " << std::endl ;
  }

private :

  eoPop <EOT> pop ; // New immigrants !

} ;

#endif



