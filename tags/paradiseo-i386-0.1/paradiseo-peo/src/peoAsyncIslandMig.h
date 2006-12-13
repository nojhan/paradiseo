// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peoAsyncIslandMig.h"

// (c) OPAC Team, LIFL, August 2005

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

#ifndef __peoAsyncIslandMig_h
#define __peoAsyncIslandMig_h

#include <queue>

#include <utils/eoUpdater.h>

#include <eoContinue.h>
#include <eoSelect.h>
#include <eoReplacement.h>
#include <eoPop.h>

#include "topo.h"
#include "coop.h"
#include "eoPop_comm.h"
#include "peo_debug.h"

template <class EOT> class peoAsyncIslandMig : public Cooperative, public eoUpdater {

public :
  
  /* Ctor */
  peoAsyncIslandMig (eoContinue <EOT> & __cont,		   
		    eoSelect <EOT> & __select,
		    eoReplacement <EOT> & __replace,
		    Topology & __topo,
		    eoPop <EOT> & __from,
		    eoPop <EOT> & __to);
  
  void operator () ();
  
  void pack ();
  
  void unpack ();
  
private :

  void emigrate ();

  void immigrate ();

  eoContinue <EOT> & cont; /* Continuator */
  eoSelect <EOT> & select; /* The selection strategy */
  eoReplacement <EOT> & replace; /* The replacement strategy */
  Topology & topo; /* The neighboring topology */

  /* Source and target populations */
  eoPop <EOT> & from, & to; 
  
  /* Immigrants & emigrants in the queue */ 
  std :: queue <eoPop <EOT> > imm, em;

  std :: queue <Cooperative *> coop_em;
};

template <class EOT> peoAsyncIslandMig <EOT> :: peoAsyncIslandMig (eoContinue <EOT> & __cont,
								   eoSelect <EOT> & __select,
								   eoReplacement <EOT> & __replace,
								   Topology & __topo,
								   eoPop <EOT> & __from,
								   eoPop <EOT> & __to
								   ) : cont (__cont),
								     select (__select),
								       replace (__replace),
								       topo (__topo),
								       from (__from),
								       to (__to) {
  
  __topo.add (* this);
}

template <class EOT> void peoAsyncIslandMig <EOT> :: pack () {

  lock ();
  :: pack (coop_em.front () -> getKey ());
  :: pack (em.front ());
  coop_em.pop ();
  em.pop ();
  unlock ();
}

template <class EOT> void peoAsyncIslandMig <EOT> :: unpack () {

  lock ();
  eoPop <EOT> mig;
  :: unpack (mig);
  imm.push (mig);  
  unlock ();
}

template <class EOT> void peoAsyncIslandMig <EOT> :: emigrate () {
  
  std :: vector <Cooperative *> in, out ;	
  topo.setNeighbors (this, in, out) ;  
  for (unsigned i = 0; i < out.size (); i ++) {
    eoPop <EOT> mig;
    select (from, mig);
    em.push (mig);
    coop_em.push (out [i]);
    send (out [i]);
    printDebugMessage ("sending some emigrants.");
  }  
}


template <class EOT> void peoAsyncIslandMig <EOT> :: immigrate () {
  
  lock ();
  while (! imm.empty ()) {
    replace (to, imm.front ()) ;
    imm.pop ();
    printDebugMessage ("receiving some immigrants.");
  }
  unlock ();  
}

template <class EOT> void peoAsyncIslandMig <EOT> :: operator () () {

  if (! cont (from)) {
    /* Sending emigrants */
    emigrate ();
    /* Immigrants */
    immigrate ();
  }
}
 
#endif
