// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoMigUpdater.h"

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

#ifndef eoMigUpdater_h
#define eoMigUpdater_h

#include <eoPop.h>
#include <eoSelect.h>
#include <eoReplacement.h>
#include <eoContinue.h>
#include <utils/eoUpdater.h>
#include <paradisEO/island/eoConnectivity.h>
#include <paradisEO/comm/messages/to/eoEOSendMessTo.h>
#include <paradisEO/comm/messages/to/eoEOReceiveMessTo.h>
#include <mpi.h>

/**
   An updater. After any evolution of an evolutionnary algorithm, it
   moves some EO from/into the current population from/to a subset
   of the neighbouring algos.
*/

template <class EOT> class eoMigUpdater : public eoUpdater {
  
public :
  
  /**
     Constructor. Some features which should be specified.
     - The neighbouring interface,
     - The communication topology,
     - Which EO instancies from the given population should be sent ?
     - How integrate new immigrants ?
     - Then, an event to determine the time to ask neighbouring sources
       for sending sets of EO.
  */

  eoMigUpdater (eoListener <EOT> & _listen,
		eoConnectivity <EOT> & _conn,
		eoContinue <EOT> & _cont,
		eoSelect <EOT> & _select,
		eoReplacement <EOT> & _replace) :
    listen (_listen),
    conn (_conn),
    cont (_cont),
    select (_select),
    replace (_replace) {
  
  }
  
  /**
     Sets the given population to be the one to receive and/or send EO
  */

  void operator () (eoPop <EOT> & _pop) {
    
    pop = & _pop ;
  }

  /**
     Should be often called. (after each local evolution ?)
   */

  virtual void operator () () {
    
    listen.update () ;
    //    listen.display () ;
    
    std::vector <eoLocalListener <EOT> *> src = conn.from (), dest = conn.to () ;
    
    // Any coming immigrants ?
    for (int i = 0 ; i < src.size () ; i ++) {
      src [i] -> update () ;
      while (! src [i] -> empty ()) {
	replace (* pop, src [i] -> front ()) ;
	std::cout << "[" << listen.here ().host_name << "] Arrival of " << src [i] -> front ().size () << " individuals ..." << std::endl ;
	src [i] -> pop () ;
      }
    }
    
    // Any request ?
    for (int i = 0 ; i < dest.size () ; i ++)
      if (dest [i] -> need_immigration ()) {
	eoPop <EOT> emm ; // Emmigrants
	select (* pop, emm) ;
	eoEOSendMessTo <EOT> mess (emm) ;
	mess (* (dest [i])) ;
      }
    
    // Any request to submit ?
    if (! cont (* pop))
      for (int i = 0 ; i < src.size () ; i ++) {
	eoEOReceiveMessTo <EOT> mess ;
	mess (* (src [i])) ;
    }
  }

private :
  
  eoConnectivity <EOT> & conn ; // The used topology
  
  eoContinue <EOT> & cont ; /* The 'event' which determines
			       need of immigration */
			       
  eoSelect <EOT> & select ; /* In order to select emmigrants
			       from the current population */
  
  eoReplacement <EOT> & replace ; // The replacement procedure

  eoPop <EOT> * pop ; // The population considered
  
  eoListener <EOT> & listen ; // A reference to the neighbouring

} ; 

#endif






