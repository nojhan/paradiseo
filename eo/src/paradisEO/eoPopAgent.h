// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoPopAgent.h"

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

#ifndef eoPopAgent_h
#define eoPopAgent_h

#include <eoPopAlgo.h>
#include <paradisEO/comm/eoListener.h>
#include <paradisEO/comm/messages/to/eoEOSendMessTo.h>

#include <string>

/**
   A popAgent is waiting for any populations that are queued.
   For each one, a processing is performed. 
*/

template <class EOT> class eoPopAgent {

public :

  /**
     Constructor
   */ 

  eoPopAgent (string _label,
	      eoListener <EOT> & _listen,
	      eoPopAlgo <EOT> & _algo
	      ) :
    label (_label),
    listen (_listen),
    algo (_algo) {
    
  }
  
  // Waiting for eoPop reception and then ...

  void operator () () {
    
    listen.publish (label) ; // To be known by everyone ...

    while (true) {
     
      listen.wait () ; // While no neighbour sends any eoPop ...
      
      for (int i = 0 ; i < listen.size () ; i ++) {
	
	while (! listen [i].empty ()) {
	  
	  eoPop <EOT> & pop = listen [i].front () ;
	  cout << "Agent [" << label << "] on " << listen.here ().host_name << " : Receiving " << pop.size () << " individuals ..." << endl ;
	  algo (pop) ; 
	  eoEOSendMessTo <EOT> mess (pop) ;
	  mess (listen [i]) ; // Coming back ...
	  listen [i].pop () ;
	}
      }
    }
  }
  
private :
  
  string label ; // string identifier
  eoListener <EOT> & listen ; // EO's listener
  eoPopAlgo <EOT> & algo ; // Local supplied algo
	   
} ;

#endif
