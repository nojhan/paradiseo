// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoMessTo.h"

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

#ifndef eoMessTo_h
#define eoMessTo_h

#include <string.h>
#include <mpi.h>
#include <paradisEO/comm/messages/to/eoHeaderMessTo.h>

/**
   An abstract class for any sending message.
   Common features are declared.
 */

template <class EOT> class eoLocalListener ;

template <class EOT> class eoMessTo {
  
public :
  
  /**
     Constructor. A std::string identifier, being defined in subclasses
     is given for any kind of messages.
  */

  eoMessTo (std::string _label) :
    
    label (_label),
    comm (MPI :: COMM_WORLD) {
  }
  
  /**
     Must be called in sub-classes ...
  */
  
  void operator () (eoLocalListener <EOT> & loc_listen) {
    
    eoHeaderMessTo <EOT> header (label) ;
    header (loc_listen) ; 
  }
  
protected :
    
  MPI :: Comm & comm ; // MPI Communicator
  
  std::string label ; // String identifier of the message
    
} ;

#endif




