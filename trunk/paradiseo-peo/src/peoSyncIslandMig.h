/*
* <peoSyncIslandMig.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Sebastien Cahon, Alexandru-Adrian Tantar, Clive Canape
*
* This software is governed by the CeCILL license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL license and that you accept its terms.
*
* ParadisEO WebSite : http://paradiseo.gforge.inria.fr
* Contact: paradiseo-help@lists.gforge.inria.fr
*
*/

#ifndef __peoSyncIslandMig_h
#define __peoSyncIslandMig_h


#include <queue>
#include <cassert>

#include <eoPeriodicContinue.h>

#include <utils/eoUpdater.h>

#include <eoContinue.h>
#include <eoSelect.h>
#include <eoReplacement.h>
#include <eoPop.h>
#include "peoData.h"

#include "core/messaging.h"
#include "core/eoPop_mesg.h"
#include "core/eoVector_mesg.h"

#include "core/topology.h"
#include "core/thread.h"
#include "core/cooperative.h"
#include "core/peo_debug.h"

#include "rmc/mpi/synchron.h"


//! @class peoSyncIslandMig
//! @brief Specific class for a synchronous migration 
//! @see Cooperative eoUpdater
//! @version 2.0
//! @date january 2008
template< class TYPESELECT, class TYPEREPLACE  > class peoSyncIslandMig : public Cooperative, public eoUpdater
  {

  public:

    //! @brief Constructor
	//! @param unsigned __frequency
	//! @param selector <TYPESELECT> & __select 
	//! @param replacement <TYPEREPLACE> & __replace
	//! @param Topology& __topology
    peoSyncIslandMig(
      unsigned __frequency,
      selector <TYPESELECT> & __select,
      replacement <TYPEREPLACE> & __replace,
      Topology& __topology
    );

    //! @brief operator
    void operator()();
	//! @brief Function realizing packages
    void pack();
    //! @brief Function reconstituting packages
    void unpack();
    //! @brief Function packSynchronizeReq
    void packSynchronizeReq();
	//! @brief Function notifySending
    void notifySending();
	//! @brief Function notifyReceiving
    void notifyReceiving();
	//! @brief notifySendingSyncReq
    void notifySendingSyncReq();
	//! @brief notifySynchronized
    void notifySynchronized();

  private:

    void emigrate();
    void immigrate();


  private:
	//! @param eoSyncContinue cont
	//! @param selector <TYPESELECT> & select
	//! @param replacement <TYPEREPLACE> & replace
	//! @param Topology& topology
	//! @param std :: queue< TYPEREPLACE > imm
	//! @param std :: queue< TYPESELECT > em
	//! @param std :: queue< Cooperative* > coop_em
	//! @param sem_t sync
	//! @param bool explicitPassive
	//! @param bool standbyMigration
	//! @param std :: vector< Cooperative* > in, out, all
	//! @param unsigned nbMigrations
    eoSyncContinue cont;	
    selector <TYPESELECT> & select;	
    replacement <TYPEREPLACE> & replace;	
    Topology& topology;		
    std :: queue< TYPEREPLACE > imm;
    std :: queue< TYPESELECT > em;
    std :: queue< Cooperative* > coop_em;
    sem_t sync;
    bool explicitPassive;
    bool standbyMigration;
    std :: vector< Cooperative* > in, out, all;
    unsigned nbMigrations;
  };


template< class TYPESELECT, class TYPEREPLACE > peoSyncIslandMig< TYPESELECT,TYPEREPLACE > :: peoSyncIslandMig(

  unsigned __frequency,
  selector <TYPESELECT> & __select,
  replacement <TYPEREPLACE> & __replace,
  Topology& __topology
) : cont( __frequency ), select( __select ), replace( __replace ), topology( __topology )
{

  __topology.add( *this );
  sem_init( &sync, 0, 0 );
}


template< class TYPESELECT, class TYPEREPLACE > void peoSyncIslandMig< TYPESELECT, TYPEREPLACE > :: pack()
{
  ::pack( coop_em.front()->getKey() );
  ::pack(em.front());
  coop_em.pop();
  em.pop();
}

template< class TYPESELECT, class TYPEREPLACE > void peoSyncIslandMig< TYPESELECT, TYPEREPLACE > :: unpack()
{
  TYPEREPLACE mig;
  ::unpack(mig);
  imm.push( mig );
  explicitPassive = true;
}

template< class TYPESELECT, class TYPEREPLACE > void peoSyncIslandMig< TYPESELECT,TYPEREPLACE > :: packSynchronizeReq()
{

  packSynchronRequest( all );
}

template< class TYPESELECT, class TYPEREPLACE > void peoSyncIslandMig< TYPESELECT , TYPEREPLACE > :: emigrate()
{

  for ( unsigned i = 0; i < out.size(); i ++ )
    {

      TYPESELECT mig;
      select( mig );
      em.push( mig );
      coop_em.push( out[ i ] );
      send( out[ i ] );
      printDebugMessage( "peoSyncIslandMig: sending some emigrants." );
    }
}

template< class TYPESELECT, class TYPEREPLACE > void peoSyncIslandMig< TYPESELECT , TYPEREPLACE > :: immigrate()
{
  assert( imm.size() );

  while ( imm.size() )
    {
      replace( imm.front() ) ;
      imm.pop();
    }

  printDebugMessage( "peoSyncIslandMig: receiving some immigrants." );
}

template< class TYPESELECT, class TYPEREPLACE > void peoSyncIslandMig< TYPESELECT , TYPEREPLACE > :: operator()()
{

  if (! cont.check() )
    {
      explicitPassive = standbyMigration = false;
      topology.setNeighbors( this, in, out );
      all = topology;
      nbMigrations = 0;
      synchronizeCoopEx();
      stop();
      // sending emigrants
      emigrate();
      // synchronizing
      sem_wait( &sync );
      // receiving immigrants
      immigrate();
      synchronizeCoopEx();
      stop();
    }
}

template< class TYPESELECT, class TYPEREPLACE > void peoSyncIslandMig< TYPESELECT , TYPEREPLACE > :: notifySending()
{
  if ( !explicitPassive ) getOwner()->setPassive();
}

template< class TYPESELECT, class TYPEREPLACE > void peoSyncIslandMig< TYPESELECT , TYPEREPLACE > :: notifyReceiving()
{
  nbMigrations++;

  if ( nbMigrations == in.size() )
    {

      if ( standbyMigration ) getOwner()->setActive();
      sem_post( &sync );
    }
}

template< class TYPESELECT, class TYPE > void peoSyncIslandMig< TYPESELECT, TYPE > :: notifySendingSyncReq ()
{

  getOwner()->setPassive();
}

template< class TYPESELECT, class TYPE > void peoSyncIslandMig< TYPESELECT, TYPE > :: notifySynchronized ()
{

  standbyMigration = true;
  getOwner()->setActive();
  resume();
}


#endif
