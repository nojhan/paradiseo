// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoListener.h"

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
   
   Cont	act: cahon@lifl.fr
*/

#ifndef eoListener_h
#define eoListener_h

#include <vector>
#include <paradisEO/comm/eoLocalListener.h>
#include <paradisEO/comm/messages/to/eoPublishMessTo.h>
#include <paradisEO/comm/messages/to/eoKillMessTo.h>
#include <mpi.h>

/**
   Necessary for any use of a distributed model.
   It makes it possible to have a total state of
   that one. 
*/

// In the near future ... Enabling different kinds of EO to send/receive ..

template <class EOT> class eoListener : public vector <eoLocalListener <EOT> > {
  
public :

  /** 
      Constructor.
  */

  eoListener (int argc, char ** argv) : comm (MPI :: COMM_WORLD) {
    
    // Mpi requires it !
    MPI :: Init (argc, argv) ;
    
    // Who and how many ?
    rank = MPI :: COMM_WORLD.Get_rank () ;
    len = MPI :: COMM_WORLD.Get_size () ;

    // To build local listeners algorithms
    for (int i = 0 ; i < len ; i ++) {
      eoLocalListener <EOT> loc_listen (i) ;
      push_back (loc_listen) ;
    }
  }
  
  /**
     Destructor. 
   */

  ~ eoListener () {
    
    MPI :: Finalize () ;
  }

  /**
     A reference to the current listener
   */

  eoLocalListener <EOT> & here () {
    
    return operator [] (rank) ;
  }
  
  /**
     To import messages ...
   */

  void update () {
    
    for (int i = 0 ; i < size () ; i ++)
      operator [] (i).update () ;
  }
  
  /**
     To broadcast the string identifier of the local process to the
     whole neighbouring ...
   */

  void publish (string label) {
    
    eoPublishMessTo <EOT> mess (label) ;
    for (int i = 0 ; i < size () ; i ++)
      if (i != rank)
	mess (operator [] (i)) ;
    here ().label () = label ; // Nothing to send !
  }
  
  /**
     Blocking. Waits for at least one 'eoLocalListener' to
     receive any eoPop ...
   */

  void wait () {
    
    bool b = false ;
    
    do {
      comm.Probe (MPI :: ANY_SOURCE, 0) ;
      update () ;
      for (int i = 0 ; i < size () ; i ++)
	if (! operator [] (i).empty ())
	  b = true ;
	
    } while (! b) ;
  }
  
  void destroy (string label) {
    
    eoKillMessTo <EOT> mess ;
    for (int i = 0 ; i < len ; i ++) {
      if (operator [] (i).label () == label)
	mess (operator [] (i)) ;
    }
  }
  
private :

  int rank, len ; // Rank of current process, and number of distributed processes
  
  MPI :: Comm & comm ; // Communicator
  
  
} ;

#endif



