// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoPublishMessTo.h"

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
   To let know from distributed algos the std::string identifier
   of the home process ...
*/

#ifndef eoPublishMessTo_h
#define eoPublishMessTo_h

#include <mpi.h>
#include <string.h>
#include <paradisEO/comm/messages/eoMessTo.h>
#include <paradisEO/comm/eoLocalListener.h>

template <class EOT> class eoPublishMessTo : public eoMessTo <EOT> {
  
public :
  
  /**
     Constructor
   */

  eoPublishMessTo (std::string & _label
		   ) :
    eoMessTo <EOT> ("eoPublishMessTo"),
    label (_label) {
    
  }
  
  void operator () (eoLocalListener <EOT> & loc_listen) {
    
    eoMessTo <EOT> :: operator () (loc_listen) ;
    comm.Send (label.c_str (), label.size () + 1, MPI :: CHAR, loc_listen.number (), 0) ;
  }
  
private :

  std::string label ; // String identifier to send ...

} ;

#endif


