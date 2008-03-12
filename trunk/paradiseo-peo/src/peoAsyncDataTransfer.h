/*
* <peoAsyncDataTransfer.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2008
* (C) OPAC Team, LIFL, 2002-2008
*
* Clive Canape
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
