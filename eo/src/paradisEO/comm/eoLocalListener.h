// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoLocalListener.h"

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

#ifndef eoLocalListener_h
#define eoLocalListener_h

#include <queue>
#include <string>
#include <paradisEO/comm/messages/from/eoHeaderMessFrom.h>
#include <paradisEO/comm/messages/from/eoEOReceiveMessFrom.h>
#include <paradisEO/comm/messages/from/eoEOSendMessFrom.h>
#include <paradisEO/comm/messages/from/eoPublishMessFrom.h>
#include <paradisEO/comm/messages/from/eoKillMessFrom.h>
#include <eoPop.h>
#include <mpi.h>
#include <unistd.h>

/**
   A local listener to pack any coming message or, on contrary, to
   send any data to another distributed process.
 */

// Well, not very nice, but necessary for multiples header inclusions :-/

template <class EOT> class eoHeaderMessFrom ;
template <class EOT> class eoMessFrom ;
template <class EOT> class eoMessTo ;
template <class EOT> class eoEOReceiveMessFrom ;
template <class EOT> class eoEOSendMessFrom ;
template <class EOT> class eoPublishMessFrom ;
template <class EOT> class eoKillMessFrom ;

template <class EOT> class eoLocalListener : public queue <eoPop <EOT> > {
  
public :

  /**
     Constructor
   */
  
  eoLocalListener (int _num_id) : 
    
    num_id (_num_id),
    comm (& MPI :: COMM_WORLD), // Shared communicator ...
    req_EO (false) {
    
    gethostname (host_name, 255) ; // Host ?
  }
  
  /**
     Any distributed algorithm or agent has its own integer
     identifiant, which lets us to distinguish them.
  */

  bool operator == (eoLocalListener <EOT> & loc_listen) {
    
    return loc_listen.num_id == num_id ;
  }
  
  /**
     To import awaiting messages from other algorithms.
     For each one, an action may be performed.
  */
  
  void update () {
   
    while (comm -> Iprobe (num_id, 0)) {
      // While any more messages 
      
      eoHeaderMessFrom <EOT> header (* this) ;
      eoMessFrom <EOT> * mess ;
      
      /* The header identifies the kind of messages.
	 Currently, only four are necessary and so defined */
      
      if (header == "eoEOReceiveMessTo")
	mess = new eoEOReceiveMessFrom <EOT> (* this) ;
      else if (header == "eoEOSendMessTo")
	mess = new eoEOSendMessFrom <EOT> (* this) ;      
      else if (header == "eoPublishMessTo")
	mess = new eoPublishMessFrom <EOT> (* this) ;
      else {
	mess = new eoKillMessFrom <EOT> (* this) ;
      }
      // Any side effects ?
      mess -> operator () () ;
      delete mess ;
    }
  }

  /**
     String identifier of this algo/agent ?
  */
  
  string & label () {
    
    return name_id ;
  }
 
  bool & need_immigration () {
    
    return req_EO ;
  }
  
  int number () {
    
    return num_id ;
  }

  void destroy () {
   
    cout << "Agent [" << name_id << "] stopped ..." << endl ;
    MPI :: Finalize () ;
    exit (0) ;
  }

  char host_name [255] ; // Host string identifier
  
private :

  MPI :: Comm * comm ; // MPI Communicator

  string name_id ; // String id.
  int num_id ; // MPI id.
  bool req_EO ;
  
  // Friendly classes
  friend class eoMessFrom <EOT> ;
  friend class eoMessTo <EOT> ;
 
} ; 

#endif



