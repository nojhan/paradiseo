/*
* <peoSyncDataTransfer.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Alexandru-Adrian TANTAR
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

#ifndef __peoSyncDataTransfer_h
#define __peoSyncDataTransfer_h


#include <queue>
#include <cassert>

#include <utils/eoUpdater.h>

#include "core/peoAbstractDefs.h"

#include "core/messaging.h"
#include "core/eoPop_mesg.h"
#include "core/eoVector_mesg.h"

#include "core/topology.h"
#include "core/thread.h"
#include "core/cooperative.h"
#include "core/peo_debug.h"

#include "rmc/mpi/synchron.h"


extern void wakeUpCommunicator();
extern int getNodeRank();


class peoSyncDataTransfer : public Cooperative, public eoUpdater
  {

  public:

    template< typename EndPointType >
    peoSyncDataTransfer(

      EndPointType& __endPoint,
      Topology& __topology

    ) : topology( __topology )
    {

      source = new MsgTransferQueue< EndPointType >( __endPoint );
      destination = new MsgTransferQueue< EndPointType >( __endPoint );
      __topology.add( *this );

      sem_init( &sync, 0, 0 );
    }

    template< typename EndPointType, typename FunctorType >
    peoSyncDataTransfer(

      EndPointType& __endPoint,
      Topology& __topology,
      FunctorType& externalFunctorRef

    ) : topology( __topology )
    {

      source = new MsgTransferQueue< EndPointType >( __endPoint, externalFunctorRef );
      destination = new MsgTransferQueue< EndPointType >( __endPoint, externalFunctorRef );
      __topology.add( *this );

      sem_init( &sync, 0, 0 );
    }

    template< typename SourceEndPointType, typename DestinationEndPointType >
    peoSyncDataTransfer(

      SourceEndPointType& __source,
      DestinationEndPointType& __destination,
      Topology& __topology

    ) : topology( __topology )
    {

      source = new MsgTransferQueue< SourceEndPointType >( __source );
      destination = new MsgTransferQueue< DestinationEndPointType >( __destination );
      __topology.add( *this );

      sem_init( &sync, 0, 0 );
    }

    template< typename SourceEndPointType, typename DestinationEndPointType, typename FunctorType >
    peoSyncDataTransfer(

      SourceEndPointType& __source,
      DestinationEndPointType& __destination,
      Topology& __topology,
      FunctorType& externalFunctorRef

    ) : topology( __topology )
    {

      source = new MsgTransferQueue< SourceEndPointType >( __source, externalFunctorRef );
      destination = new MsgTransferQueue< DestinationEndPointType >( __destination, externalFunctorRef );
      __topology.add( *this );

      sem_init( &sync, 0, 0 );
    }


    void operator()()
    {

      standbyTransfer = false;
      nbTransfersIn = nbTransfersOut = 0;
      
      topology.setNeighbors( this, in, out );
      all = topology;
      
      synchronizeCoopEx();
      stop();
      
      // sending data out
      sendData();
      // synchronizing
      sem_wait( &sync );
      // receiving data in
      receiveData();
      
      synchronizeCoopEx();
      stop();
    }
    

    void pack()
    {
      
      ::pack( coop_em.front()->getKey() );
      source->packMessage();
      coop_em.pop();
    }

    void unpack()
    {
      
      destination->unpackMessage();
    }
    

    void packSynchronizeReq()
    {
      
      packSynchronRequest( all );
    }
    

    void notifySending()
    {

      nbTransfersOut++;
      
      printDebugMessage( "peoSyncDataTransfer: notified of the completion of a transfer round." );
      
      getOwner()->setActive();
      if ( nbTransfersOut == out.size() && nbTransfersIn < in.size() )
	{
	  getOwner()->setPassive();
	}
    }


    void notifyReceiving()
    {
      
      nbTransfersIn++;
      printDebugMessage( "peoSyncIslandMig: notified of incoming data." );
      
      if ( standbyTransfer )
	{
	  getOwner()->setActive();
	  if ( nbTransfersOut == out.size() && nbTransfersIn < in.size() )
	    getOwner()->setPassive();
	}
      
      if ( nbTransfersIn == in.size() )
	{
	  
	  printDebugMessage( "peoSyncIslandMig: finished collecting incoming data." );
	  sem_post( &sync );
	}
    }
    

    void notifySendingSyncReq()
    {

      getOwner()->setPassive();
      printDebugMessage( "peoSyncIslandMig: synchronization request sent." );
    }

    void notifySynchronized()
    {

      printDebugMessage( "peoSyncIslandMig: cooperators synchronized." );
      
      standbyTransfer = true;
      getOwner()->setActive();
      resume();
    }


  private:

    void sendData()
    {
      
      for ( unsigned i = 0; i < out.size(); i ++ )
	{
	  
	  source->pushMessage();
	  
	  coop_em.push( out[ i ] );
	  send( out[ i ]);
	  
	  printDebugMessage( "peoSyncDataTransfer: sending data." );
	}
      
      wakeUpCommunicator();
    }

    void receiveData()
    {
      
      assert( !( destination->empty() ) );
      
      while ( !( destination->empty() ) )
	{
	  
	  printDebugMessage( "peoSyncDataTransfer: received data." );
	  destination->popMessage();
	  printDebugMessage( "peoSyncDataTransfer: done extracting received data." );
	}
    }
    
    Topology& topology; 			// neighboring topology

    // source and destination end-points
    AbstractMsgTransferQueue* source;
    AbstractMsgTransferQueue* destination;

    std :: queue< Cooperative* > coop_em;

    sem_t sync;

    bool standbyTransfer;

    std :: vector< Cooperative* > in, out, all;
    unsigned nbTransfersIn, nbTransfersOut;
  };

#endif
