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


#ifndef eoEOSendMessTo_h
#define eoEOSendMessTo_h

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <mpi.h>
#ifdef HAVE_SSTREAM
#include <sstream>
#else
#include <strstream.h>
#endif
#include <eoPop.h>
#include <paradisEO/comm/messages/eoMessTo.h>


/** A message embeding immigrants to send to ... */
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
#ifdef HAVE_SSTREAM
        std::ostringstream f;
        pop.printOn(f);
        comm.Send(f.str().c_str(), f.str().size(), MPI::CHAR, loc_listen.number(), 0);
#else
        std::ostrstream f;
        pop.printOn (f);
        comm.Send (f.str(), f.pcount(), MPI::CHAR, loc_listen.number(), 0);
#endif
        loc_listen.need_immigration () = false;
    }
  

protected:
  
    eoPop <EOT> & pop ; // The set of EO to send.
} ;

#endif
