// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoSolAgent.h"

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

#ifndef eoSolAgent_h
#define eoSolAgent_h

#include <eoSolAlgo.h>
#include <paradisEO/comm/eoListener.h>
#include <paradisEO/comm/messages/to/eoEOSendMessTo.h>

#include <string>

/**
   A solAgent is waiting for any solutions that are queued.
   For each one, a processing is performed. 
*/

template <class EOT> class eoSolAgent {

public :

  /**
     Constructor
   */ 

  eoSolAgent (std::string _label,
	      eoListener <EOT> & _listen,
	      eoSolAlgo <EOT> & _algo
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
	  
	  EOT & sol = listen [i].front () ;
	  std::cout << "Agent [" << label << "] on " << listen.here ().host_name << " : Receiving one individual ..." << std::endl ;
	  algo (sol) ;
	  eoPop <EOT> pop ;
	  pop.push_back (sol) ;
	  eoEOSendMessTo <EOT> mess (pop) ;
	  mess (listen [i]) ; // Coming back ...
	  listen [i].pop () ;
	}
      }
    }
  }
  
private :
  
  std::string label ; // std::string identifier
  eoListener <EOT> & listen ; // EO's listener
  eoSolAlgo <EOT> & algo ; // Local supplied algo
	   
} ;

#endif
