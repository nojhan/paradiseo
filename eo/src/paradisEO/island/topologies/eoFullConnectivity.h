// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoFullConnectivity.h"

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

#ifndef eoFullConnectivity_h
#define eoFullConnectivity_h

#include <paradisEO/island/eoConnectivity.h>

/**
   Each algorithm is also connected to the whole neighbouring ...
*/

template <class EOT> class eoFullConnectivity : public eoConnectivity <EOT> {
  
public :
  
  /**
     Constructor
   */

  eoFullConnectivity (eoListener <EOT> & _listen,
		      vector <string> & _sel_neigh
		      ) : eoConnectivity <EOT> (_listen, _sel_neigh) {    
  }
    
  virtual vector <eoLocalListener <EOT> *> from () {
    
    listen.update () ;
    
    vector <eoLocalListener <EOT> *> v ;
    
    for (int i = 0 ; i < listen.size () ; i ++)
      if (i != listen.here ().number () && selected (listen [i].label ()))
	v.push_back (& listen [i]) ;
    
    return v ;
  }
    
  virtual vector <eoLocalListener <EOT> *> to () {
    
    listen.update () ;

    vector <eoLocalListener <EOT> *> v ;

    for (int i = 0 ; i < listen.size () ; i ++)
      if (i != listen.here ().number () && selected (listen [i].label ()))
	v.push_back (& listen [i]) ;
       
    return v ;
  }
  
} ;

#endif




























