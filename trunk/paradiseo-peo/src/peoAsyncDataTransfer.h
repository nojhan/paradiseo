#ifndef __peoAsyncDataTransfer_h
#define __peoAsyncDataTransfer_h


#include <queue>

#include <utils/eoUpdater.h>

#include "core/peoAbstractDefs.h"

#include "core/messaging.h"

#include "core/topology.h"
#include "core/thread.h"
#include "core/cooperative.h"
#include "core/peo_debug.h"


class peoAsyncDataTransfer : public Cooperative, public eoUpdater
  {

  public:

    template< typename EndPointType >
    peoAsyncDataTransfer(

      EndPointType& __endPoint,
      Topology& __topology

    ) : topology( __topology )
    {

      source = new MsgTransferQueue< EndPointType >( __endPoint );
      destination = new MsgTransferQueue< EndPointType >( __endPoint );
      __topology.add( *this );
    }

    template< typename EndPointType, typename FunctorType >
    peoAsyncDataTransfer(

      EndPointType& __endPoint,
      Topology& __topology,
      FunctorType& externalFunctorRef

    ) : topology( __topology )
    {

      source = new MsgTransferQueue< EndPointType >( __endPoint, externalFunctorRef );
      destination = new MsgTransferQueue< EndPointType >( __endPoint, externalFunctorRef );
      __topology.add( *this );
    }

    template< typename SourceEndPointType, typename DestinationEndPointType >
    peoAsyncDataTransfer(

      SourceEndPointType& __source,
      DestinationEndPointType& __destination,
      Topology& __topology

    ) : topology( __topology )
    {

      source = new MsgTransferQueue< SourceEndPointType >( __source );
      destination = new MsgTransferQueue< DestinationEndPointType >( __destination );
      __topology.add( *this );
    }

    template< typename SourceEndPointType, typename DestinationEndPointType, typename FunctorType >
    peoAsyncDataTransfer(

      SourceEndPointType& __source,
      DestinationEndPointType& __destination,
      Topology& __topology,
      FunctorType& externalFunctorRef

    ) : topology( __topology )
    {

      source = new MsgTransferQueue< SourceEndPointType >( __source, externalFunctorRef );
      destination = new MsgTransferQueue< DestinationEndPointType >( __destination, externalFunctorRef );
      __topology.add( *this );
    }

    ~peoAsyncDataTransfer()
    {
      delete source;
      delete destination;
    }


    void operator()();

    void pack();
    void unpack();

    void packSynchronizeReq();


  private:

    void sendData();
    void receiveData();


  private:

    // the neighboring topology
    Topology& topology;

    // source and destination end-points
    AbstractMsgTransferQueue* source;
    AbstractMsgTransferQueue* destination;

    std :: queue< Cooperative* > coop_em;
  };


void peoAsyncDataTransfer :: pack()
{

  lock ();

  ::pack( coop_em.front()->getKey() );
  source->packMessage();
  coop_em.pop();

  unlock();
}

void peoAsyncDataTransfer :: unpack()
{

  lock ();
  destination->unpackMessage();
  unlock();
}

void peoAsyncDataTransfer :: packSynchronizeReq()
{
}

void peoAsyncDataTransfer :: sendData()
{

  std :: vector< Cooperative* > in, out;
  topology.setNeighbors( this, in, out );

  for ( unsigned i = 0; i < out.size(); i++ )
    {

      source->pushMessage();

      coop_em.push( out[i] );
      send( out[i] );

      printDebugMessage( "peoAsyncDataTransfer: sending data." );
    }
}

void peoAsyncDataTransfer :: receiveData()
{

  lock ();
  {

    while ( !( destination->empty() ) )
      {

        printDebugMessage( "peoAsyncDataTransfer: received data." );
        destination->popMessage();
        printDebugMessage( "peoAsyncDataTransfer: done reading data." );
      }
  }
  unlock();
}

void peoAsyncDataTransfer :: operator()()
{

  sendData();	        // sending data
  receiveData();	// receiving data
}


#endif
