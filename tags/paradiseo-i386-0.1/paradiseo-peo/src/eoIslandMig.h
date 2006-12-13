// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "eoIslandMig.h"

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

#ifndef __eoIslandMig_h
#define __eoIslandMig_h

#include <queue>

#include <utils/eoUpdater.h>

#include <eoContinue.h>
#include <eoSelect.h>
#include <eoReplacement.h>
#include <eoPop.h>

#include "eoTopology.h"
#include "coop.h"
#include "eoPop_comm.h"

extern int getNodeRank ();

template <class EOT> class  eoIslandMig : public eoUpdater, public Cooperative {

public :
  
  /* Ctor */
  eoIslandMig (eoSelect <EOT> & __select,
	       eoReplacement <EOT> & __replace,
	       eoTopology <EOT> & __topo,
	       eoPop <EOT> & __from,
	       eoPop <EOT> & __to);
  
  void pack ();

  void unpack ();

protected :

  void emigrate ();

  void immigrate ();

  eoSelect <EOT> & select; /* The selection strategy */
  eoReplacement <EOT> & replace; /* The replacement strategy */
  eoTopology <EOT> & topo; /* The neighboring topology */

  /* Source and target populations */
  eoPop <EOT> & from, & to; 
  
  /* Immigrants in the queue */ 
  std :: queue <eoPop <EOT> > imm;
  /* Emigrants */
  eoPop <EOT> em;

  sem_t sem_imm;
};

template <class EOT> eoIslandMig <EOT> :: eoIslandMig (eoSelect <EOT> & __select,
						       eoReplacement <EOT> & __replace,
						       eoTopology <EOT> & __topo,
						       eoPop <EOT> & __from,
						       eoPop <EOT> & __to
						       ) : select (__select),
							   replace (__replace),
							   topo (__topo),
							   from (__from),
							   to (__to) {
  sem_init (& sem_imm, 0, 0);
}

template <class EOT> void eoIslandMig <EOT> :: pack () {

  :: pack (em);
}

template <class EOT> void eoIslandMig <EOT> :: unpack () {

  printf ("Av lock unpack en %d\n", getNodeRank ());
  lock ();
  eoPop <EOT> mig;
  :: unpack (mig);
  imm.push (mig);  
  unlock ();
  printf ("Ap lock unpack en %d\n", getNodeRank ());
  sem_post (& sem_imm);
}

template <class EOT> void eoIslandMig <EOT> :: emigrate () {
  
  std :: vector <eoIslandMig <EOT> *> in, out ;	
  topo.setNeighbors (this, in, out) ;
  
  for (unsigned i = 0; i < out.size (); i ++) {
    
    select (from, em);      
    send (out [i]);
    printf ("On a reussi une emigration !!!!!!!!!!!!!\n");
  }  
}

template <class EOT> void eoIslandMig <EOT> :: immigrate () {

}

#endif
