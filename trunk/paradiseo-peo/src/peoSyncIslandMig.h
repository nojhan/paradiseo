/*
* <peoSyncIslandMig.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
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
template< class EOT, class TYPE  > class peoSyncIslandMig : public Cooperative, public eoUpdater
  {

  public:

    //! @brief Constructor
	//! @param unsigned __frequency
	//! @param selector <TYPE> & __select 
	//! @param replacement <TYPE> & __replace
	//! @param Topology& __topology
	//! @param peoData & __source
	//! @param eoData & __destination
    peoSyncIslandMig(
      unsigned __frequency,
      selector <TYPE> & __select,
      replacement <TYPE> & __replace,
      Topology& __topology,
      peoData & __source,
      peoData & __destination
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
	//! @param selector <TYPE> & select
	//! @param replacement <TYPE> & replace
	//! @param Topology& topology
	//! @param peoData & source
	//! @param peoData & destination
	//! @param std :: queue< TYPE > imm
	//! @param std :: queue< TYPE > em
	//! @param std :: queue< Cooperative* > coop_em
	//! @param sem_t sync
	//! @param bool explicitPassive
	//! @param bool standbyMigration
	//! @param std :: vector< Cooperative* > in, out, all
	//! @param unsigned nbMigrations
    eoSyncContinue cont;	
    selector <TYPE> & select;	
    replacement <TYPE> & replace;	
    Topology& topology;		
    peoData & source;
    peoData & destination;
    std :: queue< TYPE > imm;
    std :: queue< TYPE > em;
    std :: queue< Cooperative* > coop_em;
    sem_t sync;
    bool explicitPassive;
    bool standbyMigration;
    std :: vector< Cooperative* > in, out, all;
    unsigned nbMigrations;
  };


template< class EOT, class TYPE > peoSyncIslandMig< EOT,TYPE > :: peoSyncIslandMig(

  unsigned __frequency,
  selector <TYPE> & __select,
  replacement <TYPE> & __replace,
  Topology& __topology,
  peoData & __source,
  peoData & __destination

) : cont( __frequency ), select( __select ), replace( __replace ), topology( __topology ), source( __source ), destination( __destination )
{

  __topology.add( *this );
  sem_init( &sync, 0, 0 );
}


template< class EOT, class TYPE > void peoSyncIslandMig< EOT, TYPE > :: pack()
{
  ::pack( coop_em.front()->getKey() );
  em.front().pack();
  coop_em.pop();
  em.pop();
}

template< class EOT, class TYPE > void peoSyncIslandMig< EOT, TYPE > :: unpack()
{
  TYPE mig;
  mig.unpack();
  imm.push( mig );
  explicitPassive = true;
}

template< class EOT, class TYPE > void peoSyncIslandMig< EOT,TYPE > :: packSynchronizeReq()
{

  packSynchronRequest( all );
}

template< class EOT, class TYPE > void peoSyncIslandMig< EOT , TYPE > :: emigrate()
{

  for ( unsigned i = 0; i < out.size(); i ++ )
    {

      TYPE mig;
      select( mig );
      em.push( mig );
      coop_em.push( out[ i ] );
      send( out[ i ] );
      printDebugMessage( "peoSyncIslandMig: sending some emigrants." );
    }
}

template< class EOT, class TYPE > void peoSyncIslandMig< EOT , TYPE > :: immigrate()
{
  assert( imm.size() );

  while ( imm.size() )
    {
      replace( imm.front() ) ;
      imm.pop();
    }

  printDebugMessage( "peoSyncIslandMig: receiving some immigrants." );
}

template< class EOT, class TYPE > void peoSyncIslandMig< EOT , TYPE > :: operator()()
{

  if ( cont.check() )
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

template< class EOT, class TYPE > void peoSyncIslandMig< EOT , TYPE > :: notifySending()
{
  if ( !explicitPassive ) getOwner()->setPassive();
}

template< class EOT, class TYPE > void peoSyncIslandMig< EOT , TYPE > :: notifyReceiving()
{
  nbMigrations++;

  if ( nbMigrations == in.size() )
    {

      if ( standbyMigration ) getOwner()->setActive();
      sem_post( &sync );
    }
}

template< class EOT, class TYPE > void peoSyncIslandMig< EOT, TYPE > :: notifySendingSyncReq ()
{

  getOwner()->setPassive();
}

template< class EOT, class TYPE > void peoSyncIslandMig< EOT, TYPE > :: notifySynchronized ()
{

  standbyMigration = true;
  getOwner()->setActive();
  resume();
}


#endif
