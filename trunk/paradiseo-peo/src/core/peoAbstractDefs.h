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
