// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoEOSendMessTo.h"

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

/**
   A message embeding immigrants to send to ...
*/

#ifndef eoEOSendMessTo_h
#define eoEOSendMessTo_h

#include <mpi.h>
#include <strstream.h>
#include <eoPop.h>
#include <paradisEO/comm/messages/eoMessTo.h>

template <class EOT> class eoEOSendMessTo : public eoMessTo <EOT> {
  
public :
  
  /**
     Constructor ...
   */
  
  eoEOSendMessTo (eoPop <EOT> & _pop ) 
    : eoMessTo <EOT> ("eoEOSendMessTo"), 
      pop (_pop)
  {}
  
  /**
     To send the given population ...
   */

  void operator () (eoLocalListener <EOT> & loc_listen) {
    
    eoMessTo <EOT> :: operator () (loc_listen) ;
    
    ostrstream f ;
    pop.printOn (f) ;
    comm.Send (f.str (), f.pcount (), MPI :: CHAR, loc_listen.number (), 0) ;
    loc_listen.need_immigration () = false ;
  }
  
private :
  
  eoPop <EOT> & pop ; // The set of EO to send.
  
} ;

#endif
