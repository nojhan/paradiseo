// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

// "peoSyncMultiStart.h"

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

#ifndef __peoSyncMultiStart_h
#define __peoSyncMultiStart_h

#include <utils/eoUpdater.h>
#include <moAlgo.h>

#include <eoSelect.h>
#include <eoReplacement.h>
#include <eoContinue.h>

#include "service.h"
#include "mess.h"
#include "peo_debug.h"

extern int getNodeRank ();

template <class EOT> class peoSyncMultiStart : public Service, public eoUpdater {

public :

  /* Ctor */
  peoSyncMultiStart (eoContinue <EOT> & __cont,
		     eoSelect <EOT> & __select,
		     eoReplacement <EOT> & __replace,
		     moAlgo <EOT> & __ls,
		     eoPop <EOT> & __pop);

  void operator () ();

  void packData ();

  void unpackData ();
  
  void execute ();
  
  void packResult ();

  void unpackResult ();
  
  void notifySendingData ();
  void notifySendingAllResourceRequests ();

private :

  eoContinue <EOT> & cont;
  eoSelect <EOT> & select;
  eoReplacement <EOT> & replace;
  moAlgo <EOT> & ls;
  eoPop <EOT> & pop;
  eoPop <EOT> sel;
  eoPop <EOT> impr_sel;
  EOT sol;
  unsigned idx;
  unsigned num_term;
};

template <class EOT> 
peoSyncMultiStart <EOT> :: peoSyncMultiStart (eoContinue <EOT> & __cont,
					      eoSelect <EOT> & __select,
					      eoReplacement <EOT> & __replace,
					      moAlgo <EOT> & __ls,
					      eoPop <EOT> & __pop
					      ) : cont (__cont),
						  select (__select),
						  replace (__replace),
						  ls (__ls),					  
						  pop (__pop) {
}

template <class EOT> 
void peoSyncMultiStart <EOT> :: packData () {

  :: pack (sel [idx ++]);
}

template <class EOT> 
void peoSyncMultiStart <EOT> :: unpackData () {

  unpack (sol);
}

template <class EOT> 
void peoSyncMultiStart <EOT> :: execute () {
  
  ls (sol);
}

template <class EOT> 
void peoSyncMultiStart <EOT> :: packResult () {

  pack (sol);
}

template <class EOT> 
void peoSyncMultiStart <EOT> :: unpackResult () {
  
  unpack (sol);
  impr_sel.push_back (sol);
  num_term ++;
  
  if (num_term == sel.size ()) {
    getOwner () -> setActive ();
    replace (pop, impr_sel);
    printDebugMessage ("replacing the improved individuals in the population.");  
    resume ();
  }
}

template <class EOT> 
void peoSyncMultiStart <EOT> :: operator () () {

  printDebugMessage ("performing the parallel multi-start hybridization.");  
  select (pop, sel);
  impr_sel.clear ();
  idx = num_term = 0;
  requestResourceRequest (sel.size ());
  stop ();
}

template <class EOT>
void peoSyncMultiStart <EOT> :: notifySendingData () {

  
}

template <class EOT>
void peoSyncMultiStart <EOT> :: notifySendingAllResourceRequests () {

  getOwner () -> setPassive ();
}

#endif
