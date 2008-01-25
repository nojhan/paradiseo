/*
* <peoMultiStart.h>
* Copyright (C) DOLPHIN Project-Team, INRIA Futurs, 2006-2007
* (C) OPAC Team, LIFL, 2002-2007
*
* Sebastien Cahon, Alexandru-Adrian Tantar
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
#ifndef __peoMultiStart_h
#define __peoMultiStart_h

#include <vector>

#include "core/service.h"
#include "core/messaging.h"


template < typename EntityType > class peoMultiStart : public Service
  {

  public:

    template < typename AlgorithmType > peoMultiStart( AlgorithmType& externalAlgorithm )
    {

      singularAlgorithm = new Algorithm< AlgorithmType >( externalAlgorithm );
      algorithms.push_back( singularAlgorithm );

      aggregationFunction = new NoAggregationFunction();
    }

    template < typename AlgorithmReturnType, typename AlgorithmDataType > peoMultiStart( AlgorithmReturnType (*externalAlgorithm)( AlgorithmDataType& ) )
    {

      singularAlgorithm = new FunctionAlgorithm< AlgorithmReturnType, AlgorithmDataType >( externalAlgorithm );
      algorithms.push_back( singularAlgorithm );

      aggregationFunction = new NoAggregationFunction();
    }

    template < typename AlgorithmType, typename AggregationFunctionType > peoMultiStart( std::vector< AlgorithmType* >& externalAlgorithms, AggregationFunctionType& externalAggregationFunction )
    {

      for ( unsigned int index = 0; index < externalAlgorithms.size(); index++ )
        {

          algorithms.push_back( new Algorithm< AlgorithmType >( *externalAlgorithms[ index ] ) );
        }

      aggregationFunction = new AggregationAlgorithm< AggregationFunctionType >( externalAggregationFunction );
    }

    template < typename AlgorithmReturnType, typename AlgorithmDataType, typename AggregationFunctionType >
    peoMultiStart( std::vector< AlgorithmReturnType (*)( AlgorithmDataType& ) >& externalAlgorithms,
                   AggregationFunctionType& externalAggregationFunction )
    {

      for ( unsigned int index = 0; index < externalAlgorithms.size(); index++ )
        {

          algorithms.push_back( new FunctionAlgorithm< AlgorithmReturnType, AlgorithmDataType >( externalAlgorithms[ index ] ) );
        }

      aggregationFunction = new AggregationAlgorithm< AggregationFunctionType >( externalAggregationFunction );
    }

    ~peoMultiStart()
    {

      for ( unsigned int index = 0; index < data.size(); index++ ) delete data[ index ];
      for ( unsigned int index = 0; index < algorithms.size(); index++ ) delete algorithms[ index ];

      delete aggregationFunction;
    }


    template < typename Type > void operator()( Type& externalData )
    {

      for ( typename Type::iterator externalDataIterator = externalData.begin(); externalDataIterator != externalData.end(); externalDataIterator++ )
        {

          data.push_back( new DataType< EntityType >( *externalDataIterator ) );
        }

      functionIndex = dataIndex = idx = num_term = 0;
      requestResourceRequest( data.size() * algorithms.size() );
      stop();
    }


    template < typename Type > void operator()( const Type& externalDataBegin, const Type& externalDataEnd )
    {

      for ( Type externalDataIterator = externalDataBegin; externalDataIterator != externalDataEnd; externalDataIterator++ )
        {

          data.push_back( new DataType< EntityType >( *externalDataIterator ) );
        }

      functionIndex = dataIndex = idx = num_term = 0;
      requestResourceRequest( data.size() * algorithms.size() );
      stop();
    }


    void packData();

    void unpackData();

    void execute();

    void packResult();

    void unpackResult();

    void notifySendingData();

    void notifySendingAllResourceRequests();


  private:

    template < typename Type > struct DataType;

    struct AbstractDataType
      {

        virtual ~AbstractDataType()
        { }

        template < typename Type > operator Type& ()
        {

          return ( dynamic_cast< DataType< Type >& >( *this ) ).data;
        }
      };

  template < typename Type > struct DataType : public AbstractDataType
      {

        DataType( Type& externalData ) : data( externalData )
        { }

        Type& data;
      };

    struct AbstractAlgorithm
      {

        virtual ~AbstractAlgorithm()
        { }

        virtual void operator()( AbstractDataType& dataTypeInstance )
        {}
      };

  template < typename AlgorithmType > struct Algorithm : public AbstractAlgorithm
      {

        Algorithm( AlgorithmType& externalAlgorithm ) : algorithm( externalAlgorithm )
        { }

        void operator()( AbstractDataType& dataTypeInstance )
        {
          algorithm( dataTypeInstance );
        }

        AlgorithmType& algorithm;
      };


  template < typename AlgorithmReturnType, typename AlgorithmDataType > struct FunctionAlgorithm : public AbstractAlgorithm
      {

        FunctionAlgorithm( AlgorithmReturnType (*externalAlgorithm)( AlgorithmDataType& ) ) : algorithm( externalAlgorithm )
        { }

        void operator()( AbstractDataType& dataTypeInstance )
        {
          algorithm( dataTypeInstance );
        }

        AlgorithmReturnType (*algorithm)( AlgorithmDataType& );
      };

    struct AbstractAggregationAlgorithm
      {

        virtual ~AbstractAggregationAlgorithm()
        { }

        virtual void operator()( AbstractDataType& dataTypeInstanceA, AbstractDataType& dataTypeInstanceB )
        {};
      };

  template < typename AggregationAlgorithmType > struct AggregationAlgorithm : public AbstractAggregationAlgorithm
      {

        AggregationAlgorithm( AggregationAlgorithmType& externalAggregationAlgorithm ) : aggregationAlgorithm( externalAggregationAlgorithm )
        { }

        void operator()( AbstractDataType& dataTypeInstanceA, AbstractDataType& dataTypeInstanceB )
        {

          aggregationAlgorithm( dataTypeInstanceA, dataTypeInstanceB );
        }

        AggregationAlgorithmType& aggregationAlgorithm;
      };

  struct NoAggregationFunction : public AbstractAggregationAlgorithm
      {

        void operator()( AbstractDataType& dataTypeInstanceA, AbstractDataType& dataTypeInstanceB )
        {

          static_cast< EntityType& >( dataTypeInstanceA ) = static_cast< EntityType& >( dataTypeInstanceB );
        }
      };



    AbstractAlgorithm* singularAlgorithm;

    std::vector< AbstractAlgorithm* > algorithms;
    AbstractAggregationAlgorithm* aggregationFunction;


    EntityType entityTypeInstance;
    std::vector< AbstractDataType* > data;

    unsigned idx;
    unsigned num_term;
    unsigned dataIndex;
    unsigned functionIndex;
  };


template < typename EntityType > void peoMultiStart< EntityType >::packData()
{

  pack( functionIndex );
  pack( idx );
  pack( ( EntityType& ) *data[ idx++ ]  );

  // done with functionIndex for the entire data set - moving to another
  //  function/algorithm starting all over with the entire data set ( idx is set to 0 )
  if ( idx == data.size() )
    {

      ++functionIndex;
      idx = 0;
    }
}

template < typename EntityType > void peoMultiStart< EntityType >::unpackData()
{

  unpack( functionIndex );
  unpack( dataIndex );
  unpack( entityTypeInstance );
}

template < typename EntityType > void peoMultiStart< EntityType >::execute()
{

  // wrapping the unpacked data - the definition of an abstract algorithm imposes
  // that its internal function operator acts only on abstract data types
  AbstractDataType* entityWrapper = new DataType< EntityType >( entityTypeInstance );
  algorithms[ functionIndex ]->operator()( *entityWrapper );

  delete entityWrapper;
}

template < typename EntityType > void peoMultiStart< EntityType >::packResult()
{

  pack( dataIndex );
  pack( entityTypeInstance );
}

template < typename EntityType > void peoMultiStart< EntityType >::unpackResult()
{

  unpack( dataIndex );
  unpack( entityTypeInstance );

  // wrapping the unpacked data - the definition of an abstract algorithm imposes
  // that its internal function operator acts only on abstract data types
  AbstractDataType* entityWrapper = new DataType< EntityType >( entityTypeInstance );
  aggregationFunction->operator()( *data[ dataIndex ], *entityWrapper );
  delete entityWrapper;

  num_term++;

  if ( num_term == data.size() * algorithms.size() )
    {

      getOwner()->setActive();
      resume();
    }
}

template < typename EntityType > void peoMultiStart< EntityType >::notifySendingData()
{}

template < typename EntityType > void peoMultiStart< EntityType >::notifySendingAllResourceRequests()
{

  getOwner()->setPassive();
}


#endif
