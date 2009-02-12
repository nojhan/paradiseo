/*
* <peoAbstractDefs.h>
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

#if !defined __peoAbstractDefs_h
#define __peoAbstractDefs_h


#include <queue>

#include "core/messaging.h"



template < typename Type > struct Entity;

struct AbstractEntity
  {

    virtual ~AbstractEntity() {}

    template < typename EntityType > operator EntityType& ()
    {

      return ( dynamic_cast< Entity< EntityType >& >( *this ) ).entity;
    }
  };

struct AbstractFunctor : virtual public AbstractEntity
  {

    virtual ~AbstractFunctor() {}

    virtual void operator()() {}
  };

struct AbstractUnaryFunctor : virtual public AbstractEntity
  {

    virtual ~AbstractUnaryFunctor() {}

    virtual void operator()( AbstractEntity& dataEntity ) {}
  };

struct AbstractBinaryFunctor : virtual public AbstractEntity
  {

    virtual ~AbstractBinaryFunctor() {}

    virtual void operator()( AbstractEntity& dataEntityA, AbstractEntity& dataEntityB ) {};
  };



template < typename EntityType > struct Entity : virtual public AbstractEntity
  {

    Entity( EntityType& externalEntityRef ) : entity( externalEntityRef ) {}

    EntityType& entity;
  };

template < typename FunctorType, typename DataType > struct FunctorEx : public Entity< DataType >, public AbstractFunctor
  {

    FunctorEx( FunctorType& externalFunctorRef, DataType& externalDataRef )
        : externalFunctor( externalFunctorRef ), Entity< DataType >( externalDataRef ) {}

    void operator()()
    {

      externalFunctor( Entity< DataType > :: entity );
    }

    FunctorType& externalFunctor;
  };

template < typename FunctorType > struct FunctorEx< FunctorType, void > : public Entity< AbstractEntity >, public AbstractFunctor
  {

    FunctorEx( FunctorType& externalFunctorRef )
        : externalFunctor( externalFunctorRef ), Entity< AbstractEntity >( *this ) {}

    void operator()()
    {

      externalFunctor();
    }

    FunctorType& externalFunctor;
  };

template < typename ReturnType, typename DataType > struct FnFunctorEx
      : public Entity< DataType >, public AbstractFunctor
  {

    FnFunctorEx( ReturnType (*externalFunctorRef)( DataType& ),  DataType& externalDataRef )
        : externalFunctor( externalFunctorRef ), Entity< DataType >( externalDataRef ) {}

    void operator()()
    {

      externalFunctor( Entity< DataType > :: entity );
    }

    ReturnType (*externalFunctor)( DataType& );
  };

template < typename ReturnType > struct FnFunctorEx< ReturnType, void >
      : public Entity< AbstractEntity >, public AbstractFunctor
  {

    FnFunctorEx( ReturnType (*externalFunctorRef)() )
        : externalFunctor( externalFunctorRef ), Entity< AbstractEntity >( *this ) {}

    void operator()()
    {

      externalFunctor();
    }

    ReturnType (*externalFunctor)();
  };



template < typename FunctorType > struct UnaryFunctor : public Entity< FunctorType >, public AbstractUnaryFunctor
  {

    UnaryFunctor( FunctorType& externalFunctorRef ) : Entity< FunctorType >( externalFunctorRef ) {}

    void operator()( AbstractEntity& dataEntity )
    {

      Entity< FunctorType > :: entity( dataEntity );
    }
  };

template < typename ReturnType, typename DataType > struct UnaryFnFunctor
      : public Entity< AbstractEntity >, public AbstractUnaryFunctor
  {

    UnaryFnFunctor( ReturnType (*externalFnRef)( DataType& ) ) : Entity< AbstractEntity >( *this ), externalFn( externalFnRef )
    {
    }

    void operator()( AbstractEntity& dataEntity )
    {

      externalFn( dataEntity );
    }

    ReturnType (*externalFn)( DataType& );
  };

template < typename FunctorType > struct BinaryFunctor : public Entity< FunctorType >, public AbstractBinaryFunctor
  {

    BinaryFunctor( FunctorType& externalFunctorRef ) : Entity< FunctorType >( externalFunctorRef ) {}

    void operator()( AbstractEntity& dataEntityA, AbstractEntity& dataEntityB )
    {

      Entity< FunctorType > :: entity( dataEntityA, dataEntityB );
    }
  };

struct AbstractMsgTransferQueue : virtual public AbstractEntity
  {

    virtual ~AbstractMsgTransferQueue() {}

    virtual void pushMessage() {}
    virtual void popMessage() {}

    virtual bool empty()
    {
      return true;
    }

    virtual void packMessage() {}
    virtual void unpackMessage() {}
  };

template < typename EntityType > struct MsgTransferQueue : public Entity< EntityType >, public AbstractMsgTransferQueue
  {

    MsgTransferQueue( EntityType& externalDataRef )
        : Entity< EntityType >( externalDataRef )
    {

      aggregationFunctor = new BinaryFunctor< AssignmentFunctor >( assignmentFunctor );
    }

    template < typename FunctorType >
    MsgTransferQueue( EntityType& externalDataRef, FunctorType& externalFunctorRef )
        : Entity< EntityType >( externalDataRef )
    {

      aggregationFunctor = new BinaryFunctor< FunctorType >( externalFunctorRef );
    }

    ~MsgTransferQueue()
    {
      delete aggregationFunctor;
    }

    void pushMessage()
    {

      transferQueue.push( Entity< EntityType > :: entity );
    }

    void popMessage()
    {

      Entity< EntityType > message( transferQueue.front() );
      aggregationFunctor->operator()( *this, message );

      transferQueue.pop();
    }

    bool empty()
    {
      return transferQueue.empty();
    }

    void packMessage()
    {

      pack( transferQueue.front() );
      transferQueue.pop();
    }

    void unpackMessage()
    {

      EntityType transferredData;
      unpack( transferredData );
      transferQueue.push( transferredData );
    }

    struct AssignmentFunctor
      {
        void operator()( EntityType& A, EntityType& B )
        {
          A = B;
        }
      } assignmentFunctor;

    std::queue< EntityType > transferQueue;
    AbstractBinaryFunctor* aggregationFunctor;
  };



#endif
