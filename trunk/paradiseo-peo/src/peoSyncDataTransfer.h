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


  void operator()();

  void pack();

  void unpack();

  void packSynchronizeReq();

  void notifySending();

  void notifyReceiving();

  void notifySendingSyncReq();

  void notifySynchronized();


 private:

  void sendData();
  void receiveData();

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


void peoSyncDataTransfer :: pack() {

  ::pack( coop_em.front()->getKey() );
  source->packMessage();
  coop_em.pop();
}

void peoSyncDataTransfer :: unpack() {

  destination->unpackMessage();
}

void peoSyncDataTransfer :: packSynchronizeReq() {

  packSynchronRequest( all );
}

extern void wakeUpCommunicator();
extern int getNodeRank();

void peoSyncDataTransfer :: sendData() {

  for ( unsigned i = 0; i < out.size(); i ++ ) {

    source->pushMessage();
    
    coop_em.push( out[ i ] );
    send( out[ i ]);
    
    printDebugMessage( "peoSyncDataTransfer: sending data." );
  }
  
  wakeUpCommunicator();
}

void peoSyncDataTransfer :: receiveData() {
  
  assert( !( destination->empty() ) );

  while ( !( destination->empty() ) ) {
    
    printDebugMessage( "peoSyncDataTransfer: received data." );
    destination->popMessage();
    printDebugMessage( "peoSyncDataTransfer: done extracting received data." );
  }
}

void peoSyncDataTransfer :: operator()() {

  standbyTransfer = false;
  nbTransfersIn = nbTransfersOut = 0;
  
  topology.setNeighbors( this, in, out ); all = topology;
  
  synchronizeCoopEx(); stop();
  
  // sending data out
  sendData();
  // synchronizing
  sem_wait( &sync );
  // receiving data in
  receiveData();
  
  synchronizeCoopEx(); stop();
}

void peoSyncDataTransfer :: notifySending() {

  nbTransfersOut++;
  
  printDebugMessage( "peoSyncDataTransfer: notified of the completion of a transfer round." );
  
  getOwner()->setActive();
  if ( nbTransfersOut == out.size() && nbTransfersIn < in.size() ) {
    getOwner()->setPassive();
  }
}

void peoSyncDataTransfer :: notifyReceiving() {

  nbTransfersIn++;
  printDebugMessage( "peoSyncIslandMig: notified of incoming data." );
  
  if ( standbyTransfer ) {
    getOwner()->setActive();
    if ( nbTransfersOut == out.size() && nbTransfersIn < in.size() )
      getOwner()->setPassive();
  }
  
  if ( nbTransfersIn == in.size() ) {
    
    printDebugMessage( "peoSyncIslandMig: finished collecting incoming data." );
    sem_post( &sync );
  }
}

void peoSyncDataTransfer :: notifySendingSyncReq () {
  
  getOwner()->setPassive();
  printDebugMessage( "peoSyncIslandMig: synchronization request sent." );
}

void peoSyncDataTransfer :: notifySynchronized () {
  
  printDebugMessage( "peoSyncIslandMig: cooperators synchronized." );
  
  standbyTransfer = true;
  getOwner()->setActive();
  resume();
}


#endif
