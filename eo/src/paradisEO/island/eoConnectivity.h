// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoConnectivity.h"

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


#ifndef eoConnectivity_h
#define eoConnectivity_h

#include <vector>
#include <paradisEO/comm/eoLocalListener.h>
#include <paradisEO/comm/eoListener.h>

/**
   It defines the entering and outgoing channels of communication  
   towards the other considered agents. 
*/

template <class EOT> class eoConnectivity {
  
public :
  
  /**
     Constructor. Names of others processes to consider in the topology are given
     in parameter.
  */
  
  eoConnectivity (eoListener <EOT> & _listen,
		  std::vector <std::string> & _sel_neigh
		  ) :
    listen (_listen),
    sel_neigh (_sel_neigh) {    
  
  }
   
  /**
     Computes the subset of neighbours to receive
     immigrants from ...  
   */

  virtual std::vector <eoLocalListener <EOT> *> from () = 0 ;
  
  /**
     Computes the subset of neighbours to send
     emigrants to ...  
  */
  
  virtual std::vector <eoLocalListener <EOT> *> to () = 0 ;
  
protected :
  
  eoListener <EOT> & listen ;
  
  std::vector <std::string> sel_neigh ;

  bool selected (std::string & id) {
    
    for (int i = 0 ; i < sel_neigh.size () ; i ++)
      if (sel_neigh [i] == id)
	return true ;
    return false ;
  }
  
} ;

#endif




